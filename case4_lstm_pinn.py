import os
import math
import time
import copy
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


SEED = 20260308
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32

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

LAMBDA_DATA = 20.0
LAMBDA_PDE = 1.0
LAMBDA_BC = 10.0

NUM_SCALES = 5
TOKEN_FEATURE_DIM = 12
TOKEN_EMBED_DIM = 24
CONV_CHANNELS = 24
LSTM_HIDDEN = 24
HEAD_HIDDEN = 32

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

SHOCK_BETA_1 = 14.0
SHOCK_BETA_2 = 12.0
SHOCK_BETA_3 = 16.0
SHOCK_BETA_4 = 20.0

REGION_SWITCH_1 = 0.33
REGION_SWITCH_2 = 0.68
REGION_SMOOTH_K = 18.0

OUT_DIR = "outputs_case14_lstm_pinn_mms_ns_2d"
FIG_DIR = os.path.join(OUT_DIR, "figures")
DATA_DIR = os.path.join(OUT_DIR, "data")
LOG_DIR = os.path.join(OUT_DIR, "logs")
MODEL_DIR = os.path.join(OUT_DIR, "models")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dirs() -> None:
    for path in [OUT_DIR, FIG_DIR, DATA_DIR, LOG_DIR, MODEL_DIR]:
        os.makedirs(path, exist_ok=True)


def grad(outputs: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    return torch.autograd.grad(
        outputs,
        inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]


def exact_fields(xy: torch.Tensor) -> torch.Tensor:
    x = xy[:, 0:1]
    y = xy[:, 1:2]


    left_mask = 0.5 * (1.0 - torch.tanh(REGION_SMOOTH_K * (x - REGION_SWITCH_1)))
    right_mask = 0.5 * (1.0 + torch.tanh(REGION_SMOOTH_K * (x - REGION_SWITCH_2)))
    mid_mask = 1.0 - left_mask - right_mask

    envelope = x**2 * (1.0 - x)**2 * y**2 * (1.0 - y)**2

    psi_left = 0.52 * left_mask * torch.exp(-1.8 * x) * (0.82 + 0.18 * torch.cos(math.pi * y))
    psi_mid = 0.22 * mid_mask * torch.tanh(
        SHOCK_BETA_1 * (x - 0.50 + 0.07 * torch.sin(2.0 * math.pi * y))
    )
    psi_right = 0.14 * right_mask * torch.cos(2.4 * math.pi * (x - 0.72)) * (
        0.78 + 0.22 * torch.sin(2.0 * math.pi * y)
    )
    psi_bg = 0.16 * torch.sin(math.pi * x) * torch.sin(math.pi * y)

    psi = envelope * (psi_left + psi_mid + psi_right + psi_bg)

    grad_psi = grad(psi, xy)
    u = grad_psi[:, 1:2]
    v = -grad_psi[:, 0:1]

    p = (
        0.46 * torch.cos(math.pi * x) * torch.cos(0.6 * math.pi * y)
        + 0.10 * left_mask * torch.exp(-1.3 * x) * (0.75 + 0.25 * torch.sin(math.pi * y))
        + 0.18 * mid_mask * torch.tanh(SHOCK_BETA_2 * (x - 0.51))
        + 0.08 * right_mask * torch.cos(2.1 * math.pi * (x - 0.74)) * (0.70 + 0.30 * y)
        + 0.05 * torch.sin(math.pi * x * y)
    )

    T = (
        1.00
        + 0.20 * left_mask * torch.exp(-1.5 * x) * (0.78 + 0.22 * torch.cos(math.pi * y))
        + 0.19 * mid_mask * torch.tanh(
            SHOCK_BETA_3 * (x - 0.49 - 0.05 * torch.cos(2.0 * math.pi * y))
        )
        + 0.11 * right_mask * torch.cos(2.0 * math.pi * (x - 0.73)) * (
            0.65 + 0.35 * torch.sin(math.pi * y)
        )
        + 0.07 * torch.sin(math.pi * x) * torch.sin(0.5 * math.pi * y)
    )

    phi = (
        0.24 * left_mask * torch.exp(-1.9 * x) * (0.86 + 0.14 * torch.cos(math.pi * y))
        + 0.30 * mid_mask * torch.tanh(
            SHOCK_BETA_4 * (x - 0.53 + 0.08 * torch.sin(2.0 * math.pi * y))
        )
        + 0.12 * right_mask * torch.cos(2.6 * math.pi * (x - 0.74)) * (
            0.72 + 0.28 * torch.cos(math.pi * y)
        )
        + 0.06 * torch.cos(0.5 * math.pi * x) * torch.sin(math.pi * y)
    )

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



class LSTMPINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer(
            "freqs",
            torch.tensor([1.0, 2.0, 4.0, 8.0, 12.0], dtype=DTYPE),
        )
        dirs = torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0 / math.sqrt(2.0), 1.0 / math.sqrt(2.0)],
                [1.0 / math.sqrt(2.0), -1.0 / math.sqrt(2.0)],
            ],
            dtype=DTYPE,
        )
        self.register_buffer("dirs", dirs)

        self.token_proj = nn.Linear(TOKEN_FEATURE_DIM, TOKEN_EMBED_DIM)
        self.lstm = nn.LSTM(
            input_size=TOKEN_EMBED_DIM,
            hidden_size=LSTM_HIDDEN,
            num_layers=1,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(LSTM_HIDDEN, HEAD_HIDDEN),
            nn.Tanh(),
            nn.Linear(HEAD_HIDDEN, 5),
        )

        self._init_parameters()

    def _init_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        for name, param in self.lstm.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def build_tokens(self, xy: torch.Tensor) -> torch.Tensor:
        directional_coords = xy @ self.dirs.T
        raw = torch.stack(
            [
                xy[:, 0],
                xy[:, 1],
                xy[:, 0] + xy[:, 1],
                xy[:, 0] - xy[:, 1],
            ],
            dim=1,
        )
        tokens = []
        for freq in self.freqs:
            ang = math.pi * freq * directional_coords
            token = torch.cat([torch.sin(ang), torch.cos(ang), raw], dim=1)
            tokens.append(token)
        return torch.stack(tokens, dim=1)

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        tokens = self.build_tokens(xy)
        emb = torch.tanh(self.token_proj(tokens))
        with torch.backends.cudnn.flags(enabled=False):
            lstm_out, _ = self.lstm(emb)
        z = lstm_out[:, -1, :]
        return self.head(z)


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

    model = LSTMPINN().to(DEVICE).to(DTYPE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    best_state = None
    best_val = float("inf")
    loss_records = []

    training_start = time.time()
    for it in range(1, N_ITERS + 1):
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

        print(
            f"Iter {it:06d} | total={total_loss.item():.6e} | "
            f"data={data_loss.item():.6e} | pde={pde_loss.item():.6e} | "
            f"bc={bc_loss.item():.6e} | val={val_loss_value:.6e}"
        )

    training_end = time.time()

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
        plot_field(
            xg,
            yg,
            err_field,
            f"Absolute Error {name} 为后续分析提供直观且具对比价值的可视化基础",
            os.path.join(FIG_DIR, f"{name}_abs_error.png"),
            cmap="jet",
        )

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
        f.write("MMS-based LSTM-PINN metrics\n")
        f.write("Rel_L2 denotes the relative L2 error.\n\n")
        for line in metric_lines:
            f.write(line + "\n")

    total_end = time.time()
    with open(os.path.join(LOG_DIR, "runtime.txt"), "w", encoding="utf-8") as f:
        f.write(f"device = {DEVICE}\n")
        f.write(f"dtype = {DTYPE}\n")
        f.write(f"training_seconds = {training_end - training_start:.6f}\n")
        f.write(f"evaluation_seconds = {eval_end - eval_start:.6f}\n")
        f.write(f"total_seconds = {total_end - start_time:.6f}\n")
        f.write(f"best_validation_mse = {best_val:.12e}\n")

    loss_history = np.array(loss_records, dtype=np.float64)
    np.savetxt(
        os.path.join(LOG_DIR, "loss_history.txt"),
        loss_history,
        fmt="%.12e",
        header="iteration train_total train_data train_pde train_bc val_data",
    )
    plot_losses(loss_history)

    with open(os.path.join(LOG_DIR, "pde_and_exact_solution.txt"), "w", encoding="utf-8") as f:
        f.write("Modified steady electrohydrodynamic PDE system solved by MMS\n\n")
        f.write("Unknowns: u(x,y), v(x,y), p(x,y), T(x,y), phi(x,y).\n\n")
        f.write("PDE system:\n")
        f.write("1) Continuity: u_x + v_y = s_c(x,y)\n")
        f.write("2) x-momentum: rho(u u_x + v u_y) + p_x - nu( u_xx + u_yy ) - lambda_E phi_x - lambda_T T_x = s_u(x,y)\n")
        f.write("3) y-momentum: rho(u v_x + v v_y) + p_y - nu( v_xx + v_yy ) - lambda_E phi_y - lambda_T T_y = s_v(x,y)\n")
        f.write("4) Energy: u T_x + v T_y - alpha_T( T_xx + T_yy ) + eta_heat(u^2+v^2) - sigma_phi(phi_x^2+phi_y^2) = s_T(x,y)\n")
        f.write("5) Potential: -eps_phi( phi_xx + phi_yy ) + mu_phi(u phi_x + v phi_y) + delta_phi T = s_phi(x,y)\n\n")
        f.write("All source terms s_c, s_u, s_v, s_T, s_phi are back-calculated from the manufactured exact solution below.\n\n")
        f.write("Manufactured exact solution:\n")
        f.write("left_mask = 0.5 * (1 - tanh(k(x-a))), right_mask = 0.5 * (1 + tanh(k(x-b))), mid_mask = 1 - left_mask - right_mask\n")
        f.write("psi(x,y) = x^2(1-x)^2 y^2(1-y)^2 [ 0.52 left_mask exp(-1.8 x)(0.82+0.18 cos(pi y)) + 0.22 mid_mask tanh(beta1(x-0.50+0.07 sin(2 pi y))) + 0.14 right_mask cos(2.4 pi (x-0.72))(0.78+0.22 sin(2 pi y)) + 0.16 sin(pi x) sin(pi y) ]\n")
        f.write("u(x,y) = d psi / d y\n")
        f.write("v(x,y) = - d psi / d x\n")
        f.write("p(x,y) = 0.46 cos(pi x) cos(0.6 pi y) + 0.10 left_mask exp(-1.3 x)(0.75+0.25 sin(pi y)) + 0.18 mid_mask tanh(beta2(x-0.51)) + 0.08 right_mask cos(2.1 pi (x-0.74))(0.70+0.30 y) + 0.05 sin(pi x y)\n")
        f.write("T(x,y) = 1.00 + 0.20 left_mask exp(-1.5 x)(0.78+0.22 cos(pi y)) + 0.19 mid_mask tanh(beta3(x-0.49-0.05 cos(2 pi y))) + 0.11 right_mask cos(2.0 pi (x-0.73))(0.65+0.35 sin(pi y)) + 0.07 sin(pi x) sin(0.5 pi y)\n")
        f.write("phi(x,y) = 0.24 left_mask exp(-1.9 x)(0.86+0.14 cos(pi y)) + 0.30 mid_mask tanh(beta4(x-0.53+0.08 sin(2 pi y))) + 0.12 right_mask cos(2.6 pi (x-0.74))(0.72+0.28 cos(pi y)) + 0.06 cos(0.5 pi x) sin(pi y)\n\n")
        f.write(f"beta1 = {SHOCK_BETA_1}, beta2 = {SHOCK_BETA_2}, beta3 = {SHOCK_BETA_3}, beta4 = {SHOCK_BETA_4}.\n")
        f.write(f"a = {REGION_SWITCH_1}, b = {REGION_SWITCH_2}, k = {REGION_SMOOTH_K}.\n")

    print(f"All outputs have been saved to: {os.path.abspath(OUT_DIR)}")


if __name__ == "__main__":
    main()
