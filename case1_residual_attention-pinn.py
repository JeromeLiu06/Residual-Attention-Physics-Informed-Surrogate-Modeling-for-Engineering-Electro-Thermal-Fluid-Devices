
import os
import math
import time
import copy
import json
import signal
import random
import traceback
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


SEED = 20260324
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32
CASE_NAME = "Case10_boundary_internal_farfield_ResidualAttentionPINN"

DOMAIN_X_MIN = 0.0
DOMAIN_X_MAX = 1.0
DOMAIN_Y_MIN = 0.0
DOMAIN_Y_MAX = 1.0
NX = 41
NY = 41
TRAIN_RATIO = 0.7

N_ITERS = 200000
TRAIN_BATCH_SIZE = 128
BC_BATCH_SIZE = 96
LEARNING_RATE = 1.0e-3
WEIGHT_DECAY = 1.0e-8
GRAD_CLIP_NORM = 1.0
PRINT_EVERY = 1
VALIDATE_EVERY = 2000
SAVE_EVERY = 2000

LAMBDA_DATA = 20.0
LAMBDA_PDE = 1.0
LAMBDA_BC = 10.0

RHO = 1.0
NU = 0.02
LAMBDA_E = 0.80
LAMBDA_T = 0.60
ALPHA_T = 0.02
ETA_HEAT = 0.10
SIGMA_PHI = 0.08
EPS_PHI = 0.03
MU_PHI = 0.25
DELTA_PHI = 0.30

PHI_BL_WIDTH = 0.025
PHI_INNER_WIDTH = 0.030
T_BL_WIDTH = 0.030
T_INNER_WIDTH = 0.045

OUT_DIR = "outputs_case10_residual_attention_pinn_mms_ns_2d_autosave"
FIG_DIR = os.path.join(OUT_DIR, "figures")
DATA_DIR = os.path.join(OUT_DIR, "data")
LOG_DIR = os.path.join(OUT_DIR, "logs")
MODEL_DIR = os.path.join(OUT_DIR, "models")

STOP_REQUESTED = False
STOP_REASON = ""


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dirs() -> None:
    for path in [OUT_DIR, FIG_DIR, DATA_DIR, LOG_DIR, MODEL_DIR]:
        os.makedirs(path, exist_ok=True)


def request_stop(signum, _frame) -> None:
    global STOP_REQUESTED, STOP_REASON
    STOP_REQUESTED = True
    try:
        STOP_REASON = signal.Signals(signum).name
    except Exception:
        STOP_REASON = f"signal_{signum}"
    print(f"\\n[Signal] Received {STOP_REASON}. Will save checkpoint after current iteration and exit gracefully.")


for _sig in [getattr(signal, "SIGINT", None), getattr(signal, "SIGTERM", None)]:
    if _sig is not None:
        try:
            signal.signal(_sig, request_stop)
        except Exception:
            pass


def grad(outputs: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    return torch.autograd.grad(
        outputs,
        inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]


def sech2(z: torch.Tensor) -> torch.Tensor:
    return 1.0 / torch.cosh(z).pow(2)


def exact_fields(xy: torch.Tensor) -> torch.Tensor:
    x = xy[:, 0:1]
    y = xy[:, 1:2]

    phi_boundary = 0.55 * torch.exp(-x / PHI_BL_WIDTH) * (0.85 + 0.15 * torch.cos(math.pi * y))
    phi_centerline = 0.52 + 0.08 * torch.sin(math.pi * y)
    phi_internal = 0.28 * sech2((x - phi_centerline) / PHI_INNER_WIDTH)
    phi_farfield = (
        0.12 * torch.sin(math.pi * x) * torch.sin(0.5 * math.pi * y)
        + 0.05 * torch.cos(0.5 * math.pi * x) * torch.cos(math.pi * y)
    )
    phi = phi_boundary + phi_internal + phi_farfield

    T_boundary = 0.42 * torch.exp(-(1.0 - x) / T_BL_WIDTH) * (0.80 + 0.20 * torch.sin(math.pi * y))
    T_centerline = 0.60 + 0.06 * torch.cos(2.0 * math.pi * x)
    T_internal = 0.11 * sech2((y - T_centerline) / T_INNER_WIDTH)
    T_farfield = 1.00 + 0.14 * torch.cos(0.5 * math.pi * x) * torch.sin(math.pi * y) + 0.08 * x * (1.0 - x)
    T = T_farfield + T_boundary + T_internal

    p = (
        0.45 * torch.cos(math.pi * x) * torch.cos(0.5 * math.pi * y)
        + 0.18 * torch.sin(0.5 * math.pi * x + 0.30 * math.pi * y)
        + 0.08 * x * (1.0 - x) * y * (1.0 - y)
    )

    envelope = x**2 * (1.0 - x)**2 * y**2 * (1.0 - y)**2
    psi = envelope * (
        0.52 * torch.sin(math.pi * x) * torch.sin(math.pi * y)
        + 0.20 * p
        + 0.16 * (T - 1.0)
        + 0.14 * phi
        + 0.08 * torch.sin(2.0 * math.pi * x * y)
    )

    grad_psi = grad(psi, xy)
    u = grad_psi[:, 1:2]
    v = -grad_psi[:, 0:1]

    return torch.cat([u, v, p, T, phi], dim=1)


def pde_operator(fields: torch.Tensor, xy: torch.Tensor) -> torch.Tensor:
    u = fields[:, 0:1]
    v = fields[:, 1:2]
    p = fields[:, 2:3]
    T = fields[:, 3:4]
    phi = fields[:, 4:5]

    g_u = grad(u, xy)
    g_v = grad(v, xy)
    g_p = grad(p, xy)
    g_T = grad(T, xy)
    g_phi = grad(phi, xy)

    u_x, u_y = g_u[:, 0:1], g_u[:, 1:2]
    v_x, v_y = g_v[:, 0:1], g_v[:, 1:2]
    p_x, p_y = g_p[:, 0:1], g_p[:, 1:2]
    T_x, T_y = g_T[:, 0:1], g_T[:, 1:2]
    phi_x, phi_y = g_phi[:, 0:1], g_phi[:, 1:2]

    u_xx = grad(u_x, xy)[:, 0:1]
    u_yy = grad(u_y, xy)[:, 1:2]
    v_xx = grad(v_x, xy)[:, 0:1]
    v_yy = grad(v_y, xy)[:, 1:2]
    T_xx = grad(T_x, xy)[:, 0:1]
    T_yy = grad(T_y, xy)[:, 1:2]
    phi_xx = grad(phi_x, xy)[:, 0:1]
    phi_yy = grad(phi_y, xy)[:, 1:2]

    lap_u = u_xx + u_yy
    lap_v = v_xx + v_yy
    lap_T = T_xx + T_yy
    lap_phi = phi_xx + phi_yy

    r_cont = u_x + v_y
    r_mom_x = RHO * (u * u_x + v * u_y) + p_x - NU * lap_u - LAMBDA_E * phi_x - LAMBDA_T * T_x
    r_mom_y = RHO * (u * v_x + v * v_y) + p_y - NU * lap_v - LAMBDA_E * phi_y - LAMBDA_T * T_y
    r_energy = u * T_x + v * T_y - ALPHA_T * lap_T + ETA_HEAT * (u**2 + v**2) - SIGMA_PHI * (phi_x**2 + phi_y**2)
    r_phi = -EPS_PHI * lap_phi + MU_PHI * (u * phi_x + v * phi_y) + DELTA_PHI * T

    return torch.cat([r_cont, r_mom_x, r_mom_y, r_energy, r_phi], dim=1)


def build_full_dataset(device: str):
    x = torch.linspace(DOMAIN_X_MIN, DOMAIN_X_MAX, NX, device=device, dtype=DTYPE)
    y = torch.linspace(DOMAIN_Y_MIN, DOMAIN_Y_MAX, NY, device=device, dtype=DTYPE)
    xx, yy = torch.meshgrid(x, y, indexing="xy")
    xy = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)

    xy_req = xy.clone().detach().requires_grad_(True)
    fields = exact_fields(xy_req)
    sources = pde_operator(fields, xy_req)

    x_flat = xy[:, 0].detach().cpu().numpy()
    y_flat = xy[:, 1].detach().cpu().numpy()
    boundary_mask = (
        np.isclose(x_flat, DOMAIN_X_MIN)
        | np.isclose(x_flat, DOMAIN_X_MAX)
        | np.isclose(y_flat, DOMAIN_Y_MIN)
        | np.isclose(y_flat, DOMAIN_Y_MAX)
    )

    return (
        xy.detach().contiguous(),
        fields.detach().contiguous(),
        sources.detach().contiguous(),
        torch.from_numpy(boundary_mask.astype(np.bool_)).to(device),
    )



class ResidualAttentionBlock(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.fc1 = nn.Linear(width, width)
        self.fc2 = nn.Linear(width, width)
        self.attn = nn.Sequential(
            nn.Linear(width, width),
            nn.Tanh(),
            nn.Linear(width, width),
            nn.Sigmoid(),
        )
        self.norm = nn.LayerNorm(width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.tanh(self.fc1(x))
        h = torch.tanh(self.fc2(h))
        a = self.attn(x)
        out = x + a * h
        return self.norm(out)


class ResidualAttentionPINN(nn.Module):
    def __init__(self, in_dim: int = 2, width: int = 128, n_blocks: int = 6, out_dim: int = 5):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, width)
        self.blocks = nn.ModuleList([ResidualAttentionBlock(width) for _ in range(n_blocks)])
        self.head = nn.Sequential(
            nn.Linear(width, width),
            nn.Tanh(),
            nn.Linear(width, out_dim),
        )
        self._init_parameters()

    def _init_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        x = torch.tanh(self.in_proj(xy))
        for block in self.blocks:
            x = block(x)
        return self.head(x)



def sample_indices(indices: torch.Tensor, batch_size: int, device: str) -> torch.Tensor:
    if batch_size >= indices.numel():
        return indices
    selected = torch.randperm(indices.numel(), device=device)[:batch_size]
    return indices[selected]


def relative_l2(pred: np.ndarray, exact: np.ndarray) -> float:
    return float(np.linalg.norm(pred - exact) / (np.linalg.norm(exact) + 1.0e-14))


def compute_metrics(pred: np.ndarray, exact: np.ndarray):
    err = pred - exact
    mse = float(np.mean(err**2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(err)))
    abs_l2 = float(np.linalg.norm(err))
    rel_l2 = relative_l2(pred, exact)
    return mse, rmse, mae, abs_l2, rel_l2


def save_grid_txt(path: str, x: np.ndarray, y: np.ndarray, value: np.ndarray) -> None:
    arr = np.column_stack([x.reshape(-1), y.reshape(-1), value.reshape(-1)])
    np.savetxt(path, arr, fmt="%.12e", header="x y value")


def plot_field(xg, yg, value, title, save_png, cmap="jet"):
    plt.figure(figsize=(6.5, 5.2))
    plt.pcolormesh(xg, yg, value, shading="auto", cmap=cmap)
    plt.colorbar()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_png, dpi=300)
    plt.close()


def plot_losses(loss_history: np.ndarray) -> None:
    if loss_history.size == 0:
        return

    steps = loss_history[:, 0]
    train_total = loss_history[:, 1]
    train_data = loss_history[:, 2]
    train_pde = loss_history[:, 3]
    train_bc = loss_history[:, 4]
    val_data = loss_history[:, 5]

    plt.figure(figsize=(7.0, 5.0))
    plt.plot(steps, train_total, label="train_total")
    plt.plot(steps, train_data, label="train_data")
    plt.plot(steps, train_pde, label="train_pde")
    plt.plot(steps, train_bc, label="train_bc")
    plt.plot(steps, val_data, label="val_data")
    plt.xlabel("iteration")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "loss_linear.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(7.0, 5.0))
    plt.semilogy(steps, np.maximum(train_total, 1.0e-30), label="train_total")
    plt.semilogy(steps, np.maximum(train_data, 1.0e-30), label="train_data")
    plt.semilogy(steps, np.maximum(train_pde, 1.0e-30), label="train_pde")
    plt.semilogy(steps, np.maximum(train_bc, 1.0e-30), label="train_bc")
    plt.semilogy(steps, np.maximum(val_data, 1.0e-30), label="val_data")
    plt.xlabel("iteration")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "loss_log.png"), dpi=300)
    plt.close()


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    best_val: float,
    best_state,
    loss_records,
    reason: str,
) -> None:
    ensure_dirs()
    ckpt_path = os.path.join(MODEL_DIR, "latest_checkpoint.pt")
    state = {
        "case_name": CASE_NAME,
        "iteration": int(iteration),
        "best_val": float(best_val) if math.isfinite(best_val) else None,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "best_model_state": best_state,
        "loss_history": np.asarray(loss_records, dtype=np.float64),
        "reason": reason,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    torch.save(state, ckpt_path)
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, "last_model.pt"))
    if best_state is not None:
        torch.save(best_state, os.path.join(MODEL_DIR, "best_model_so_far.pt"))

    meta = {
        "case_name": CASE_NAME,
        "iteration": int(iteration),
        "best_val": float(best_val) if math.isfinite(best_val) else None,
        "reason": reason,
        "checkpoint": os.path.abspath(ckpt_path),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(LOG_DIR, "latest_checkpoint_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def save_loss_history_txt(loss_records) -> np.ndarray:
    if len(loss_records) == 0:
        with open(os.path.join(LOG_DIR, "loss_history.txt"), "w", encoding="utf-8") as f:
            f.write("# No loss history recorded.\n")
        return np.zeros((0, 6), dtype=np.float64)

    loss_history = np.array(loss_records, dtype=np.float64)
    np.savetxt(
        os.path.join(LOG_DIR, "loss_history.txt"),
        loss_history,
        fmt="%.12e",
        header="iteration train_total train_data train_pde train_bc val_data",
    )
    return loss_history


def main() -> None:
    start_time = time.time()
    set_seed(SEED)
    ensure_dirs()
    torch.set_default_dtype(DTYPE)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    xy_all, exact_all, source_all, boundary_mask = build_full_dataset(DEVICE)
    n_points = xy_all.shape[0]

    all_indices = torch.arange(n_points, device=DEVICE)
    perm = torch.randperm(n_points, device=DEVICE)
    n_train = int(TRAIN_RATIO * n_points)
    train_indices = perm[:n_train]
    val_indices = perm[n_train:]
    boundary_indices = all_indices[boundary_mask]

    val_xy = xy_all[val_indices]
    val_exact = exact_all[val_indices]

    model = ResidualAttentionPINN().to(DEVICE).to(DTYPE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    best_state = None
    best_val = float("inf")
    loss_records = []
    last_iter = 0
    finish_reason = "completed"

    training_start = time.time()

    try:
        for it in range(1, N_ITERS + 1):
            last_iter = it
            model.train()

            batch_idx = sample_indices(train_indices, TRAIN_BATCH_SIZE, DEVICE)
            bc_idx = sample_indices(boundary_indices, BC_BATCH_SIZE, DEVICE)

            batch_xy = xy_all[batch_idx].clone().detach().requires_grad_(True)
            batch_exact = exact_all[batch_idx]
            batch_source = source_all[batch_idx]

            pred = model(batch_xy)
            data_loss = torch.mean((pred - batch_exact) ** 2)

            residual = pde_operator(pred, batch_xy) - batch_source
            pde_loss = torch.mean(residual ** 2)

            bc_xy = xy_all[bc_idx]
            bc_exact = exact_all[bc_idx]
            bc_pred = model(bc_xy)
            bc_loss = torch.mean((bc_pred - bc_exact) ** 2)

            total_loss = LAMBDA_DATA * data_loss + LAMBDA_PDE * pde_loss + LAMBDA_BC * bc_loss

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()

            val_loss_value = float("nan")
            if it % VALIDATE_EVERY == 0 or it == 1 or it == N_ITERS:
                model.eval()
                with torch.no_grad():
                    val_pred = model(val_xy)
                    val_loss = torch.mean((val_pred - val_exact) ** 2)
                    val_loss_value = float(val_loss.item())
                    if val_loss_value < best_val:
                        best_val = val_loss_value
                        best_state = copy.deepcopy(model.state_dict())

            loss_records.append(
                [
                    float(it),
                    float(total_loss.item()),
                    float(data_loss.item()),
                    float(pde_loss.item()),
                    float(bc_loss.item()),
                    val_loss_value,
                ]
            )

            if it % PRINT_EVERY == 0:
                print(
                    f"Iter {it:06d} | total={total_loss.item():.6e} | "
                    f"data={data_loss.item():.6e} | pde={pde_loss.item():.6e} | "
                    f"bc={bc_loss.item():.6e} | val={val_loss_value:.6e}"
                )

            if it % SAVE_EVERY == 0:
                save_checkpoint(model, optimizer, it, best_val, best_state, loss_records, reason="periodic_save")

            if STOP_REQUESTED:
                finish_reason = f"graceful_stop_by_{STOP_REASON or 'signal'}"
                print(f"[Stop] {finish_reason}. Saving checkpoint and finishing post-processing...")
                save_checkpoint(model, optimizer, it, best_val, best_state, loss_records, reason=finish_reason)
                break

    except KeyboardInterrupt:
        finish_reason = "keyboard_interrupt"
        print("\n[Interrupt] KeyboardInterrupt captured. Saving checkpoint and finishing post-processing...")
        save_checkpoint(model, optimizer, last_iter, best_val, best_state, loss_records, reason=finish_reason)
    except Exception as exc:
        finish_reason = f"exception_{type(exc).__name__}"
        print(f"\n[Error] {exc}")
        traceback.print_exc()
        save_checkpoint(model, optimizer, last_iter, best_val, best_state, loss_records, reason=finish_reason)
        raise
    finally:
        training_end = time.time()
        save_checkpoint(model, optimizer, last_iter, best_val, best_state, loss_records, reason=finish_reason)

    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, "best_model.pt"))

    model.eval()
    eval_start = time.time()
    with torch.no_grad():
        pred_all = model(xy_all).detach().cpu().numpy()
    eval_end = time.time()

    xy_np = xy_all.detach().cpu().numpy()
    exact_np = exact_all.detach().cpu().numpy()
    x_np = xy_np[:, 0]
    y_np = xy_np[:, 1]
    xg = x_np.reshape(NY, NX)
    yg = y_np.reshape(NY, NX)

    field_names = ["u", "v", "p", "T", "phi"]
    metric_lines = []
    for i, name in enumerate(field_names):
        exact_field = exact_np[:, i].reshape(NY, NX)
        pred_field = pred_all[:, i].reshape(NY, NX)
        err_field = np.abs(pred_field - exact_field)

        save_grid_txt(os.path.join(DATA_DIR, f"{name}_exact.txt"), xg, yg, exact_field)
        save_grid_txt(os.path.join(DATA_DIR, f"{name}_pred.txt"), xg, yg, pred_field)
        save_grid_txt(os.path.join(DATA_DIR, f"{name}_abs_error.txt"), xg, yg, err_field)

        plot_field(xg, yg, exact_field, f"Exact {name}", os.path.join(FIG_DIR, f"{name}_exact.png"), cmap="jet")
        plot_field(xg, yg, pred_field, f"Predicted {name}", os.path.join(FIG_DIR, f"{name}_pred.png"), cmap="jet")
        plot_field(xg, yg, err_field, f"Absolute Error {name}", os.path.join(FIG_DIR, f"{name}_abs_error.png"), cmap="jet")

        mse, rmse, mae, abs_l2, rel_l2 = compute_metrics(pred_all[:, i], exact_np[:, i])
        metric_lines.append(
            f"Field: {name}\n"
            f"MSE      = {mse:.12e}\n"
            f"RMSE     = {rmse:.12e}\n"
            f"MAE      = {mae:.12e}\n"
            f"Abs_L2   = {abs_l2:.12e}\n"
            f"Rel_L2   = {rel_l2:.12e}\n"
        )

    all_mse, all_rmse, all_mae, all_abs_l2, all_rel_l2 = compute_metrics(pred_all.reshape(-1), exact_np.reshape(-1))
    metric_lines.append(
        "Field: all_fields_stacked\n"
        f"MSE      = {all_mse:.12e}\n"
        f"RMSE     = {all_rmse:.12e}\n"
        f"MAE      = {all_mae:.12e}\n"
        f"Abs_L2   = {all_abs_l2:.12e}\n"
        f"Rel_L2   = {all_rel_l2:.12e}\n"
    )

    with open(os.path.join(LOG_DIR, "metrics.txt"), "w", encoding="utf-8") as f:
        f.write("Residual Attention PINN metrics\n")
        f.write(f"Case = {CASE_NAME}\n")
        f.write("Rel_L2 denotes the relative L2 error.\n\n")
        for line in metric_lines:
            f.write(line + "\n")

    total_end = time.time()
    with open(os.path.join(LOG_DIR, "runtime.txt"), "w", encoding="utf-8") as f:
        f.write(f"case = {CASE_NAME}\n")
        f.write(f"device = {DEVICE}\n")
        f.write(f"dtype = {DTYPE}\n")
        f.write(f"finish_reason = {finish_reason}\n")
        f.write(f"training_seconds = {training_end - training_start:.6f}\n")
        f.write(f"evaluation_seconds = {eval_end - eval_start:.6f}\n")
        f.write(f"total_seconds = {total_end - start_time:.6f}\n")
        if math.isfinite(best_val):
            f.write(f"best_validation_mse = {best_val:.12e}\n")
        else:
            f.write("best_validation_mse = None\n")

    loss_history = save_loss_history_txt(loss_records)
    plot_losses(loss_history)

    with open(os.path.join(LOG_DIR, "pde_and_exact_solution.txt"), "w", encoding="utf-8") as f:
        f.write("Modified steady electrohydrodynamic PDE system solved by MMS\n\n")
        f.write(f"Case = {CASE_NAME}\n")
        f.write("Case 10 characteristic: boundary layer + internal thin layer + far-field slow variation coexist.\n\n")
        f.write("Unknowns: u(x,y), v(x,y), p(x,y), T(x,y), phi(x,y).\n\n")
        f.write("PDE system:\n")
        f.write("1) Continuity: u_x + v_y = s_c(x,y)\n")
        f.write("2) x-momentum: rho(u u_x + v u_y) + p_x - nu( u_xx + u_yy ) - lambda_E phi_x - lambda_T T_x = s_u(x,y)\n")
        f.write("3) y-momentum: rho(u v_x + v v_y) + p_y - nu( v_xx + v_yy ) - lambda_E phi_y - lambda_T T_y = s_v(x,y)\n")
        f.write("4) Energy: u T_x + v T_y - alpha_T( T_xx + T_yy ) + eta_heat(u^2+v^2) - sigma_phi(phi_x^2+phi_y^2) = s_T(x,y)\n")
        f.write("5) Potential: -eps_phi( phi_xx + phi_yy ) + mu_phi(u phi_x + v phi_y) + delta_phi T = s_phi(x,y)\n\n")
        f.write("All source terms s_c, s_u, s_v, s_T, s_phi are back-calculated from the manufactured exact solution below.\n\n")
        f.write("Manufactured exact solution (Case 10):\n")
        f.write("phi(x,y) = phi_left_boundary + phi_internal_layer + phi_farfield\n")
        f.write("phi_left_boundary = 0.55 exp(-x/eps_phi_bl) [0.85 + 0.15 cos(pi y)]\n")
        f.write("phi_internal_layer = 0.28 sech^2((x - (0.52 + 0.08 sin(pi y))) / eps_phi_inner)\n")
        f.write("phi_farfield = 0.12 sin(pi x) sin(0.5 pi y) + 0.05 cos(0.5 pi x) cos(pi y)\n\n")
        f.write("T(x,y) = T_farfield + T_right_boundary + T_internal_layer\n")
        f.write("T_right_boundary = 0.42 exp(-(1-x)/eps_T_bl) [0.80 + 0.20 sin(pi y)]\n")
        f.write("T_internal_layer = 0.11 sech^2((y - (0.60 + 0.06 cos(2 pi x))) / eps_T_inner)\n")
        f.write("T_farfield = 1 + 0.14 cos(0.5 pi x) sin(pi y) + 0.08 x(1-x)\n\n")
        f.write("p(x,y) = 0.45 cos(pi x) cos(0.5 pi y) + 0.18 sin(0.5 pi x + 0.30 pi y) + 0.08 x(1-x) y(1-y)\n\n")
        f.write("psi(x,y) = x^2(1-x)^2 y^2(1-y)^2 [0.52 sin(pi x) sin(pi y) + 0.20 p + 0.16 (T-1) + 0.14 phi + 0.08 sin(2 pi x y)]\n")
        f.write("u(x,y) = d psi / d y\n")
        f.write("v(x,y) = - d psi / d x\n\n")
        f.write(f"eps_phi_bl = {PHI_BL_WIDTH}, eps_phi_inner = {PHI_INNER_WIDTH}, eps_T_bl = {T_BL_WIDTH}, eps_T_inner = {T_INNER_WIDTH}.\n")
        f.write("\nAutosave:\n")
        f.write("- latest_checkpoint.pt is always refreshed.\n")
        f.write("- last_model.pt stores the latest model parameters.\n")
        f.write("- best_model_so_far.pt stores the best validation model seen during training.\n")
        f.write("- Press Ctrl+C to trigger graceful saving and post-processing.\n")

    print(f"All outputs have been saved to: {os.path.abspath(OUT_DIR)}")
    print("Autosave files: latest_checkpoint.pt, last_model.pt, best_model_so_far.pt")


if __name__ == "__main__":
    main()
