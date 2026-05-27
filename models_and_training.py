import os
import json
import math
import random
from dataclasses import dataclass, asdict, replace
from itertools import product
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import MD17
from torch_geometric.nn import radius_graph
from torch_scatter import scatter, scatter_mean, scatter_max, scatter_min


# =========================
# Reproducibility utilities
# =========================

def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================
# Config
# =========================
@dataclass
class TrainConfig:
    root: str = "./data/MD17"
    molecule: str = "aspirin"
    dataset_variant: str = "revised"  # 'revised' for rMD17 in PyG
    train_size: int = 1000
    val_size: int = 200
    test_size: int = 1000
    batch_size: int = 16
    hidden_dim: int = 128
    num_interactions: int = 4
    num_rbf: int = 50
    cutoff: float = 5.0
    lr: float = 1e-4
    weight_decay: float = 1e-6
    epochs: int = 100
    force_weight: float = 100.0
    energy_weight: float = 1.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    save_dir: str = "./output/mlff_project/runs"
    split_id: str = "default_split"


@dataclass
class ExperimentGrid:
    learning_rates: Tuple[float, float] = (1e-4, 5e-4)
    weight_decays: Tuple[float, float] = (1e-4, 1e-2)


# =========================
# Dataset
# =========================

def load_dataset(cfg: TrainConfig):
    try:
        dataset = MD17(root=cfg.root, name=cfg.molecule, revised=(cfg.dataset_variant == "revised"))
    except TypeError:
        dataset = MD17(root=cfg.root, name=cfg.molecule)
    return dataset


def make_split_indices(n: int, cfg: TrainConfig):
    required = cfg.train_size + cfg.val_size + cfg.test_size
    if required > n:
        raise ValueError(f"Requested {required} samples but dataset has only {n}.")
    g = torch.Generator().manual_seed(cfg.seed)
    perm = torch.randperm(n, generator=g).tolist()
    return {
        "train": perm[: cfg.train_size],
        "val": perm[cfg.train_size: cfg.train_size + cfg.val_size],
        "test": perm[cfg.train_size + cfg.val_size: required],
    }


def build_loaders(dataset, split_indices: Dict[str, List[int]], batch_size: int):
    train_set = Subset(dataset, split_indices["train"])
    val_set = Subset(dataset, split_indices["val"])
    test_set = Subset(dataset, split_indices["test"])
    return (
        DataLoader(train_set, batch_size=batch_size, shuffle=True),
        DataLoader(val_set, batch_size=batch_size, shuffle=False),
        DataLoader(test_set, batch_size=batch_size, shuffle=False),
    )


# =========================
# Basis / helpers
# =========================
class GaussianRBF(nn.Module):
    def __init__(self, num_rbf: int, cutoff: float):
        super().__init__()
        centers = torch.linspace(0.0, cutoff, num_rbf)
        widths = torch.full((num_rbf,), (cutoff / num_rbf))
        self.register_buffer("centers", centers)
        self.register_buffer("widths", widths)

    def forward(self, distances):
        diff = distances.unsqueeze(-1) - self.centers
        return torch.exp(-0.5 * (diff / self.widths) ** 2)


class ShiftedSoftplus(nn.Module):
    def forward(self, x):
        return F.softplus(x) - math.log(2.0)


class AtomEmbedding(nn.Module):
    def __init__(self, hidden_dim: int, max_z: int = 100):
        super().__init__()
        self.emb = nn.Embedding(max_z, hidden_dim)

    def forward(self, z):
        return self.emb(z)


class CFConvMessage(nn.Module):
    def __init__(self, hidden_dim: int, num_rbf: int):
        super().__init__()
        act = ShiftedSoftplus()
        self.filter_net = nn.Sequential(
            nn.Linear(num_rbf, hidden_dim),
            act,
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.dense = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, edge_index, rbf):
        row, col = edge_index
        filt = self.filter_net(rbf)
        xj = self.dense(x[col])
        m_ij = xj * filt
        return m_ij, row


class UpdateMLP(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        act = ShiftedSoftplus()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            act,
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, m):
        return self.net(m)


class SchNetInteraction(nn.Module):
    def __init__(self, hidden_dim: int, num_rbf: int):
        super().__init__()
        self.message = CFConvMessage(hidden_dim, num_rbf)
        self.update = UpdateMLP(hidden_dim)

    def forward(self, x, edge_index, rbf):
        m_ij, row = self.message(x, edge_index, rbf)
        m = scatter(m_ij, row, dim=0, dim_size=x.size(0), reduce="sum")
        return x + self.update(m)


class SchNetPlusInteraction(nn.Module):
    def __init__(self, hidden_dim: int, num_rbf: int):
        super().__init__()
        self.message = CFConvMessage(hidden_dim, num_rbf)
        self.update = UpdateMLP(hidden_dim)
        self.fuse = nn.Linear(hidden_dim * 4, hidden_dim)

    def forward(self, x, edge_index, rbf):
        m_ij, row = self.message(x, edge_index, rbf)
        m_sum = scatter(m_ij, row, dim=0, dim_size=x.size(0), reduce="sum")
        m_avg = scatter_mean(m_ij, row, dim=0, dim_size=x.size(0))
        m_max = scatter_max(m_ij, row, dim=0, dim_size=x.size(0))[0]
        m_min = scatter_min(m_ij, row, dim=0, dim_size=x.size(0))[0]

        h_sum = x + self.update(m_sum)
        h_avg = x + self.update(m_avg)
        h_max = x + self.update(m_max)
        h_min = x + self.update(m_min)
        fused = torch.cat([h_sum, h_avg, h_max, h_min], dim=-1)
        return self.fuse(fused)


class LogSumExpAggregator(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(hidden_dim))

    def forward(self, messages, row, num_nodes):
        alpha = self.alpha.unsqueeze(0)
        scaled = messages * alpha
        max_per_node = scatter_max(scaled, row, dim=0, dim_size=num_nodes)[0]
        stabilized = torch.exp(scaled - max_per_node[row])
        mean_exp = scatter_mean(stabilized, row, dim=0, dim_size=num_nodes)
        out = (torch.log(mean_exp + 1e-12) + max_per_node) / (alpha.squeeze(0) + 1e-8)
        near_zero = torch.abs(alpha.squeeze(0)) < 1e-4
        mean_msg = scatter_mean(messages, row, dim=0, dim_size=num_nodes)
        out[:, near_zero] = mean_msg[:, near_zero]
        return out


class SchNetPlusPlusInteraction(nn.Module):
    def __init__(self, hidden_dim: int, num_rbf: int):
        super().__init__()
        self.message = CFConvMessage(hidden_dim, num_rbf)
        self.update = UpdateMLP(hidden_dim)
        self.lse = LogSumExpAggregator(hidden_dim)
        self.fuse = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, x, edge_index, rbf):
        m_ij, row = self.message(x, edge_index, rbf)
        m_sum = scatter(m_ij, row, dim=0, dim_size=x.size(0), reduce="sum")
        m_lse = self.lse(m_ij, row, x.size(0))
        h_sum = x + self.update(m_sum)
        h_lse = x + self.update(m_lse)
        return self.fuse(torch.cat([h_sum, h_lse], dim=-1))


class EnergyHead(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        act = ShiftedSoftplus()
        self.atomwise = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            act,
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x, batch):
        e_i = self.atomwise(x).squeeze(-1)
        return scatter(e_i, batch, dim=0, reduce="sum")


class BaseSchNet(nn.Module):
    def __init__(self, cfg: TrainConfig, interaction_cls):
        super().__init__()
        self.cutoff = cfg.cutoff
        self.embedding = AtomEmbedding(cfg.hidden_dim)
        self.rbf = GaussianRBF(cfg.num_rbf, cfg.cutoff)
        self.interactions = nn.ModuleList([
            interaction_cls(cfg.hidden_dim, cfg.num_rbf)
            for _ in range(cfg.num_interactions)
        ])
        self.energy_head = EnergyHead(cfg.hidden_dim)

    def forward(self, z, pos, batch):
        edge_index = radius_graph(pos, r=self.cutoff, batch=batch, loop=False)
        row, col = edge_index
        dist = torch.norm(pos[row] - pos[col], dim=-1)
        rbf = self.rbf(dist)
        x = self.embedding(z)
        for inter in self.interactions:
            x = inter(x, edge_index, rbf)
        energy = self.energy_head(x, batch)
        return energy


class SchNetModel(BaseSchNet):
    def __init__(self, cfg: TrainConfig):
        super().__init__(cfg, SchNetInteraction)


class SchNetPlusModel(BaseSchNet):
    def __init__(self, cfg: TrainConfig):
        super().__init__(cfg, SchNetPlusInteraction)


class SchNetPlusPlusModel(BaseSchNet):
    def __init__(self, cfg: TrainConfig):
        super().__init__(cfg, SchNetPlusPlusInteraction)


# =========================
# Training / evaluation
# =========================

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compute_forces(energy, pos):
    return -torch.autograd.grad(energy.sum(), pos, create_graph=True)[0]


def step(model, batch, cfg: TrainConfig, optimizer=None):
    batch = batch.to(cfg.device)
    pos = batch.pos.clone().detach().requires_grad_(True)
    pred_e = model(batch.z, pos, batch.batch)
    pred_f = compute_forces(pred_e, pos)

    true_e = batch.energy.view(-1)
    true_f = batch.force

    loss_e = F.l1_loss(pred_e, true_e)
    loss_f = F.l1_loss(pred_f, true_f)
    loss = cfg.energy_weight * loss_e + cfg.force_weight * loss_f

    if optimizer is not None:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return {
        "loss": loss.detach().item(),
        "energy_mae": loss_e.detach().item(),
        "force_mae": loss_f.detach().item(),
    }


def run_epoch(model, loader, cfg: TrainConfig, optimizer=None):
    train = optimizer is not None
    model.train(train)
    metrics = {"loss": 0.0, "energy_mae": 0.0, "force_mae": 0.0}
    n = 0
    for batch in loader:
        with torch.enable_grad() if train else torch.set_grad_enabled(True):
            out = step(model, batch, cfg, optimizer if train else None)
        bs = batch.num_graphs
        for k, v in out.items():
            metrics[k] += v * bs
        n += bs
    for k in metrics:
        metrics[k] /= max(n, 1)
    return metrics


def train_model(model_name: str, model_cls, cfg: TrainConfig, split_indices: Dict[str, List[int]]):
    set_seed(cfg.seed)
    dataset = load_dataset(cfg)
    train_loader, val_loader, test_loader = build_loaders(dataset, split_indices, cfg.batch_size)

    model = model_cls(cfg).to(cfg.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    config_name = f"lr_{cfg.lr}_wd_{cfg.weight_decay}".replace(".", "p")
    run_dir = os.path.join(cfg.save_dir, config_name, model_name)
    os.makedirs(run_dir, exist_ok=True)

    history = {
        "model": model_name,
        "config": asdict(cfg),
        "split_indices": split_indices,
        "num_parameters": count_parameters(model),
        "train": [],
        "val": [],
        "test": None,
        "best_val_loss": float("inf"),
        "best_epoch": -1,
        "best_state_path": os.path.join(run_dir, "best_model.pt"),
        "final_state_path": os.path.join(run_dir, "final_model.pt"),
    }

    best_state = None
    for epoch in range(1, cfg.epochs + 1):
        train_metrics = run_epoch(model, train_loader, cfg, optimizer)
        val_metrics = run_epoch(model, val_loader, cfg, optimizer=None)
        train_metrics["epoch"] = epoch
        val_metrics["epoch"] = epoch
        history["train"].append(train_metrics)
        history["val"].append(val_metrics)

        if val_metrics["loss"] < history["best_val_loss"]:
            history["best_val_loss"] = val_metrics["loss"]
            history["best_epoch"] = epoch
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            torch.save(best_state, history["best_state_path"])

        print(
            f"[{config_name}]-[{model_name}]-[{history['num_parameters']}]: epoch #{epoch} - train_loss = {train_metrics['loss']} - val_loss = {val_metrics['loss']}")

    torch.save(model.state_dict(), history["final_state_path"])
    if best_state is not None:
        model.load_state_dict(best_state)
    history["test"] = run_epoch(model, test_loader, cfg, optimizer=None)
    print("===========================================================================================================")
    print(
        f"[{config_name}]-[{model_name}]-[{history['num_parameters']}]: finished training - test_loss = {history['test']}")

    with open(os.path.join(run_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    return history


def efficiency_score(model_mae: float, baseline_mae: float, model_params: int, baseline_params: int):
    rel_error_improvement = (baseline_mae - model_mae) / max(baseline_mae, 1e-12)
    rel_param_increase = (model_params - baseline_params) / max(baseline_params, 1e-12)
    return rel_error_improvement / (1.0 + max(rel_param_increase, 0.0))


def summarize_against_baseline(baseline_hist: Dict, candidate_hist: Dict):
    b_mae = baseline_hist["test"]["loss"]
    c_mae = candidate_hist["test"]["loss"]
    b_p = baseline_hist["num_parameters"]
    c_p = candidate_hist["num_parameters"]
    return {
        "baseline_force_mae": b_mae,
        "candidate_force_mae": c_mae,
        "baseline_params": b_p,
        "candidate_params": c_p,
        "relative_total_error_improvement": (b_mae - c_mae) / max(b_mae, 1e-12),
        "relative_param_increase": (c_p - b_p) / max(b_p, 1e-12),
        "efficiency_score": efficiency_score(c_mae, b_mae, c_p, b_p),
    }


def run_experiment_grid(base_cfg: TrainConfig, grid: ExperimentGrid):
    set_seed(base_cfg.seed)
    dataset = load_dataset(base_cfg)
    split_indices = make_split_indices(len(dataset), base_cfg)

    model_specs = [
        ("schnet", SchNetModel),
        ("schnet_plus", SchNetPlusModel),
        ("schnet_plusplus", SchNetPlusPlusModel),
    ]

    all_results = {}
    for lr, wd in product(grid.learning_rates, grid.weight_decays):
        cfg = replace(base_cfg, lr=lr, weight_decay=wd, split_id=f"seed_{base_cfg.seed}")
        config_name = f"lr_{lr}_wd_{wd}".replace('.', 'p')
        all_results[config_name] = {}
        for model_name, model_cls in model_specs:
            hist = train_model(model_name, model_cls, cfg, split_indices)
            all_results[config_name][model_name] = hist

        baseline = all_results[config_name]["schnet"]
        comparisons = {}
        for candidate in ["schnet_plus", "schnet_plusplus"]:
            comparisons[candidate] = summarize_against_baseline(baseline, all_results[config_name][candidate])
            print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            print(
                f"[{config_name}]-[{candidate}]-[{comparisons[candidate]['candidate_params']}]: efficiency score = {comparisons[candidate]['efficiency_score']}, relative total error improvement = {comparisons[candidate]['relative_total_error_improvement']}, relative param increase = {comparisons[candidate]['relative_param_increase']}")

        with open(os.path.join(cfg.save_dir, config_name, "comparisons.json"), "w") as f:
            json.dump(comparisons, f, indent=2)

    with open(os.path.join(base_cfg.save_dir, "all_results_index.json"), "w") as f:
        json.dump({k: list(v.keys()) for k, v in all_results.items()}, f, indent=2)
    return all_results


if __name__ == "__main__":
    cfg = TrainConfig()
    grid = ExperimentGrid()
    print("Uncomment below to run the full grid.")
    run_experiment_grid(cfg, grid)
