import math
from dataclasses import dataclass
from typing import Callable, Dict
import numpy as np
import torch
from scipy.linalg import solve_banded

@dataclass
class TargetFunction:
    name: str
    domain_dim: int
    func: Callable[[torch.Tensor], torch.Tensor]

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.func(x)


def _ensure_d_input(x: torch.Tensor, d: int) -> torch.Tensor:
    """
    Ensure x has shape (N, d).
    If x has shape (d,), convert to (1, d).
    """
    if x.ndim == 1:
        x = x.unsqueeze(0)

    if x.ndim != 2:
        raise ValueError(f"Expected x to have 1 or 2 dimensions, got shape {tuple(x.shape)}")

    if x.shape[1] != d:
        raise ValueError(f"Expected input dimension {d}, got shape {tuple(x.shape)}")

    return x


# --------------------------------------------------------------------
# Parametric DEs
# --------------------------------------------------------------------


def _piston_core(x: torch.Tensor) -> torch.Tensor:
    x = _ensure_d_input(x, 7)

    M  = x[:, [0]]
    S  = x[:, [1]]
    V0 = x[:, [2]]
    k  = x[:, [3]]
    P0 = x[:, [4]]
    Ta = x[:, [5]]
    T0 = x[:, [6]];

    Aterm1 = P0 * S;
    Aterm2 = 19.62 * M;
    Aterm3 = -k*V0 / S;
    A = Aterm1 + Aterm2 + Aterm3;

    Vfact1 = S / (2*k);
    Vfact2 = torch.sqrt(A**2 + 4*k*(P0*V0/T0)*Ta);
    V = Vfact1 * (Vfact2 - A);

    fact1 = M;
    fact2 = k + (S**2)*(P0*V0/T0)*(Ta/(V**2));

    return 2 * math.pi * torch.sqrt(fact1 / fact2)



def piston_scaled_7d(x: torch.Tensor) -> torch.Tensor:
    """
    Map inputs from [-1, 1]^6 to the physical piston scales.
    """
    x = _ensure_d_input(x, 7)
    lower = torch.tensor(
        [30, 0.005, 0.002, 1000, 90000, 290, 340],
        device=x.device,
        dtype=x.dtype,
    )
    upper = torch.tensor(
        [60, 0.020, 0.010, 5000, 110000, 296, 360],
        device=x.device,
        dtype=x.dtype,
    )

    scale = (upper - lower) / 2.0
    center = (upper + lower) / 2.0
    x_scaled = x * scale + center

    return _piston_core(x_scaled)


def robotsq_8d(x: torch.Tensor) -> torch.Tensor:
    x = _ensure_d_input(x, 8)
    theta = x[:, :4]
    L = x[:, 4:]
    sumu = torch.zeros((x.shape[0], 1), device=x.device, dtype=x.dtype)
    sumv = torch.zeros((x.shape[0], 1), device=x.device, dtype=x.dtype)

    for i in range(4):
        sumtheta = torch.sum(theta[:, :i+1], dim=1, keepdim=True)
        Li = L[:, [i]]
        sumu = sumu + Li * torch.cos(sumtheta)
        sumv = sumv + Li * torch.sin(sumtheta)

    return sumu**2 + sumv**2


def robotsq_scaled_8d(x: torch.Tensor) -> torch.Tensor:
    x = _ensure_d_input(x, 8)

    range_tensor = torch.tensor(
        [
            [0.0, 2.0 * math.pi],
            [0.0, 2.0 * math.pi],
            [0.0, 2.0 * math.pi],
            [0.0, 2.0 * math.pi],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ],
        device=x.device,
        dtype=x.dtype,
    )

    sc1 = (range_tensor[:, 1] - range_tensor[:, 0]) / 2.0
    sc2 = (range_tensor[:, 1] + range_tensor[:, 0]) / 2.0

    x_scaled = x * sc1.unsqueeze(0) + sc2.unsqueeze(0)
    return robotsq_8d(x_scaled)



def _wingweight_core(x: torch.Tensor) -> torch.Tensor:
    x = _ensure_d_input(x, 10)

    Sw  = x[:, [0]]
    Wfw = x[:, [1]]
    A   = x[:, [2]]
    Lam = x[:, [3]]
    q   = x[:, [4]]
    lam = x[:, [5]]
    tc  = x[:, [6]]
    Nz  = x[:, [7]]
    Wdg = x[:, [8]]
    Wp  = x[:, [9]]
    Lam_rad = Lam * math.pi / 180.0

    term1 = 0.036 * Sw**0.758 * Wfw**0.0035
    term2 = (A / (torch.cos(Lam_rad)**2))**0.6
    term3 = q**0.006 * lam**0.04
    term4 = (100 * tc / torch.cos(Lam_rad))**(-0.3)
    term5 = (Nz * Wdg)**0.49

    return (term1 * term2 * term3 * term4 * term5 + Sw * Wp)/480


def wingweight_scaled_10d(x: torch.Tensor) -> torch.Tensor:
    x = _ensure_d_input(x, 10)

    lower = torch.tensor(
        [150, 220, 6, -10, 16, 0.5, 0.08, 2.5, 1700, 0.025],
        device=x.device, dtype=x.dtype
    )

    upper = torch.tensor(
        [200, 300, 10, 10, 45, 1, 0.18, 6, 2500, 0.08],
        device=x.device, dtype=x.dtype
    )

    sc1 = (upper - lower) / 2.0
    sc2 = (upper + lower) / 2.0

    x_scaled = x * sc1 + sc2   

    return _wingweight_core(x_scaled)

def wingweight_scaled_reduced(
    x: torch.Tensor,
) -> torch.Tensor:
    x = _ensure_d_input(x, 10)

    lower = torch.tensor(
        [150.0, 220.0, 6.0, -10.0, 16.0,
         0.5, 0.08, 2.5, 1700.0, 0.025],
        device=x.device,
        dtype=x.dtype,
    )

    upper = torch.tensor(
        [200.0, 300.0, 10.0, 10.0, 45.0,
         1.0, 0.18, 6.0, 2500.0, 0.08],
        device=x.device,
        dtype=x.dtype,
    )

    scale = (upper - lower) / 2.0
    center = (upper + lower) / 2.0

    # Scale all variables first
    x_scaled = x * scale + center

    # Replace the last five scaled physical values with 1
    fixed_ones = torch.ones_like(x_scaled[:, 4:])

    x_scaled = torch.cat(
        [x_scaled[:, :4], fixed_ones],
        dim=1,
    )

    return _wingweight_core(x_scaled)


def _circuit_core(x: torch.Tensor) -> torch.Tensor:
    x = _ensure_d_input(x, 6)
    Rb1  = x[:, [0]]
    Rb2 = x[:, [1]]
    Rf   = x[:, [2]]
    Rc1 = x[:, [3]]
    Rc2   = x[:, [4]]
    beta = x[:, [5]]
    
    Vb1 = 12*Rb2 / (Rb1+Rb2);
    term1a = (Vb1+0.74) * beta * (Rc2+9);
    term1b = beta*(Rc2+9) + Rf;
    term1 = term1a / term1b;

    term2a = 11.35 * Rf;
    term2b = beta*(Rc2+9) + Rf;
    term2 = term2a / term2b;

    term3a = 0.74 * Rf * beta * (Rc2+9);
    term3b = (beta*(Rc2+9)+Rf) * Rc1;
    term3 = term3a / term3b;
    
    return term1 + term2 + term3


def circuit_scaled_6d(x: torch.Tensor) -> torch.Tensor:
    x = _ensure_d_input(x, 6)

    lower = torch.tensor(
        [50, 25, 0.5, 1.2, 0.25, 50],
        device=x.device, dtype=x.dtype
    )

    upper = torch.tensor(
        [150, 70, 3, 2.5, 1.2, 300],
        device=x.device, dtype=x.dtype
    )

    sc1 = (upper - lower) / 2.0
    sc2 = (upper + lower) / 2.0

    x_scaled = x * sc1 + sc2   

    return _circuit_core(x_scaled)

# --------------------------------------------------------------------
# 1D functions
# --------------------------------------------------------------------

def sin_1d(x: torch.Tensor) -> torch.Tensor:
    x = _ensure_d_input(x, 1)
    return torch.sin(x[:, [0]])


def exp_1d(x: torch.Tensor) -> torch.Tensor:
    x = _ensure_d_input(x, 1)
    return torch.exp(x[:, [0]])


def tanh_1d(x: torch.Tensor) -> torch.Tensor:
    x = _ensure_d_input(x, 1)
    return torch.tanh(2.0 * x[:, [0]])


def bump_1d(x: torch.Tensor) -> torch.Tensor:
    x = _ensure_d_input(x, 1)
    z = x[:, [0]] - 0.5
    return torch.exp(-40.0 * z**2)


# --------------------------------------------------------------------
# 2D functions
# --------------------------------------------------------------------

def quadratic_2d(x: torch.Tensor) -> torch.Tensor:
    x = _ensure_d_input(x, 2)
    x1 = x[:, [0]]
    x2 = x[:, [1]]
    return x1**2 + 2.0 * x2**2


def gaussian_2d(x: torch.Tensor) -> torch.Tensor:
    x = _ensure_d_input(x, 2)
    x1 = x[:, [0]]
    x2 = x[:, [1]]
    return torch.exp(-5.0 * ((x1 - 0.5)**2 + (x2 - 0.5)**2))


def separable_sin_2d(x: torch.Tensor) -> torch.Tensor:
    x = _ensure_d_input(x, 2)
    x1 = x[:, [0]]
    x2 = x[:, [1]]
    return torch.sin(2.0 * math.pi * x1) * torch.sin(2.0 * math.pi * x2)


def mixed_exp_trig_2d(x: torch.Tensor) -> torch.Tensor:
    x = _ensure_d_input(x, 2)
    x1 = x[:, [0]]
    x2 = x[:, [1]]
    return torch.exp(x1) * torch.cos(2.0 * math.pi * x2)


# --------------------------------------------------------------------
# 3D functions
# --------------------------------------------------------------------

def radial_3d(x: torch.Tensor) -> torch.Tensor:
    x = _ensure_d_input(x, 3)
    r2 = torch.sum((x - 0.5)**2, dim=1, keepdim=True)
    return torch.exp(-8.0 * r2)


def polynomial_3d(x: torch.Tensor) -> torch.Tensor:
    x = _ensure_d_input(x, 3)
    x1 = x[:, [0]]
    x2 = x[:, [1]]
    x3 = x[:, [2]]
    return x1 + x2**2 - x1 * x3 + 0.5 * x3**3


def trig_3d(x: torch.Tensor) -> torch.Tensor:
    x = _ensure_d_input(x, 3)
    x1 = x[:, [0]]
    x2 = x[:, [1]]
    x3 = x[:, [2]]
    return (
        torch.sin(2.0 * math.pi * x1)
        + 0.5 * torch.cos(2.0 * math.pi * x2)
        + 0.25 * torch.sin(4.0 * math.pi * x3)
    )


# --------------------------------------------------------------------
# Higher dimension functions
# --------------------------------------------------------------------

def separable_sin_8d(x: torch.Tensor) -> torch.Tensor:
    x = _ensure_d_input(x, 8)
    return 1.0 + torch.sum(torch.sin(math.pi * x), dim=1, keepdim=True)

def gaussian_8d(x: torch.Tensor) -> torch.Tensor:
    x = _ensure_d_input(x, 8)
    r2 = torch.sum((x - 0.5)**2, dim=1, keepdim=True)
    return torch.exp(-5.0 * r2)

def separable_sin_8d_prod(x: torch.Tensor) -> torch.Tensor:
    x = _ensure_d_input(x, 8)
    return torch.prod(torch.sin(2.0 * math.pi * x), dim=1, keepdim=True)

def exp_sum_32d(x: torch.Tensor) -> torch.Tensor:
    x = _ensure_d_input(x, 32)
    return torch.exp(torch.sum(x, dim=1, keepdim=True) / 32.0)


def trig_sum_32d(x: torch.Tensor) -> torch.Tensor:
    x = _ensure_d_input(x, 32)
    return torch.sum(torch.sin(math.pi * x), dim=1, keepdim=True)


def mixed_exp_trig_32d(x: torch.Tensor) -> torch.Tensor:
    x = _ensure_d_input(x, 32)
    return torch.exp(x[:, [0]]) * torch.prod(torch.cos(math.pi * x[:, 1:]), dim=1, keepdim=True)


def weighted_trig_32d(x: torch.Tensor) -> torch.Tensor:
    x = _ensure_d_input(x, 32)
    out = torch.zeros((x.shape[0], 1), device=x.device, dtype=x.dtype)
    for j in range(32):
        out = out + (0.5 ** j) * torch.sin((j + 1) * math.pi * x[:, [j]])
    return out


def make_nonintegrable_singularity_delta_ia(d, del_a):

    def f(x: torch.Tensor) -> torch.Tensor:
        x_ = _ensure_d_input(x, d)

        idx = torch.arange(1, d + 1, device=x_.device, dtype=x_.dtype)
        delta = idx ** del_a

        numer = torch.sqrt(2.0 * delta + delta**2)
        denom = x_ + 1.0 + delta.unsqueeze(0)

        return torch.prod(numer.unsqueeze(0) / denom, dim=1, keepdim=True)

    return TargetFunction(
        f"nonintegrable_singularity_delta_ia_{d}d_a{del_a}",
        d,
        f
    )


def make_aniso_exp(d):

    def f(x: torch.Tensor) -> torch.Tensor:
        x_ = _ensure_d_input(x, d)

        idx = torch.arange(1, d + 1, device=x_.device, dtype=x_.dtype)
        weights = 1.0 / (2.0 * idx)

        return torch.exp(torch.sum(x_ * weights.unsqueeze(0), dim=1, keepdim=True))

    return TargetFunction(
        f"aniso_exp_{d}d",
        d,
        f
    )


# --------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------

FUNCTIONS: Dict[str, TargetFunction] = {
    "sin_1d": TargetFunction("sin_1d", 1, sin_1d),
    "exp_1d": TargetFunction("exp_1d", 1, exp_1d),
    "tanh_1d": TargetFunction("tanh_1d", 1, tanh_1d),
    "bump_1d": TargetFunction("bump_1d", 1, bump_1d),
    "quadratic_2d": TargetFunction("quadratic_2d", 2, quadratic_2d),
    "gaussian_2d": TargetFunction("gaussian_2d", 2, gaussian_2d),
    "separable_sin_2d": TargetFunction("separable_sin_2d", 2, separable_sin_2d),
    "mixed_exp_trig_2d": TargetFunction("mixed_exp_trig_2d", 2, mixed_exp_trig_2d),
    "radial_3d": TargetFunction("radial_3d", 3, radial_3d),
    "polynomial_3d": TargetFunction("polynomial_3d", 3, polynomial_3d),
    "trig_3d": TargetFunction("trig_3d", 3, trig_3d),
    "separable_sin_8d": TargetFunction("separable_sin_8d", 8, separable_sin_8d),
    "separable_sin_8d_prod": TargetFunction("separable_sin_8d_prod", 8, separable_sin_8d_prod),
    "exp_sum_32d": TargetFunction("exp_sum_32d", 32, exp_sum_32d),
    "trig_sum_32d": TargetFunction("trig_sum_32d", 32, trig_sum_32d),
    "mixed_exp_trig_32d": TargetFunction("mixed_exp_trig_32d", 32, mixed_exp_trig_32d),
    "weighted_trig_32d": TargetFunction("weighted_trig_32d", 32, weighted_trig_32d),
    "robotsq_scaled_8d": TargetFunction("robotsq_scaled_8d", 8, robotsq_scaled_8d),
    "wingweight_scaled_10d": TargetFunction("wingweight_scaled_10d", 10, wingweight_scaled_10d),
    "wingweight_scaled_reduced": TargetFunction("wingweight_scaled_reduced", 10, wingweight_scaled_reduced),
    "circuit_scaled_6d": TargetFunction("circuit_scaled_6d", 6, circuit_scaled_6d),
    "piston_scaled_7d": TargetFunction("piston_scaled_7d", 7, piston_scaled_7d)
}


def get_target_function(name: str) -> TargetFunction:
    if name not in FUNCTIONS:
        available = ", ".join(FUNCTIONS.keys())
        raise ValueError(f"Unknown target function '{name}'. Available: {available}")
    return FUNCTIONS[name]


def list_target_functions():
    return list(FUNCTIONS.keys())


# --------------------------------------------------------------------
# Parametric DEs
# --------------------------------------------------------------------


def eval_lognormal_ppde_batch(x: torch.Tensor, w, n=1023):
    x_np = x.detach().cpu().numpy()
    vals = np.zeros(x_np.shape[0])
    g_FEM = lambda x: 10.0 * np.ones_like(x, dtype=float)

    for i in range(x_np.shape[0]):
        z = x_np[i, :]
        a = define_diffusion_term_1D(z, w)
        vals[i] = FEM_1D_diffusion_QoI(a, g_FEM, n)
    return torch.tensor(vals, dtype=x.dtype, device=x.device).reshape(-1, 1)


def define_diffusion_term_1D(y, w):
    y = np.asarray(y, dtype=float).reshape(-1)
    w = np.asarray(w, dtype=float)

    # Preliminary checks
    if w.ndim == 0:
        if np.sum(y > w) + np.sum(y < -w) > 0:
            raise ValueError(f"Parameter value is not in [-{w}, {w}]^d")
    else:
        if np.sum(y > w) + np.sum(y < -w) > 0:
            raise ValueError("Parameter value is not in [-w, w]^d")

    # Extract parametric dimension
    d = len(y)
    
    beta_c = 1 / 8
    beta_p = max(1, 2 * beta_c)
    beta = beta_c / beta_p


    def zeta(i):
        return (np.sqrt(np.pi) * beta) ** 0.5 * np.exp(-((np.floor(i / 2) * np.pi * beta) ** 2) / 8)

    def a_exponent(x):
        val = 1 + y[0] * (np.sqrt(np.pi) * beta / 2) ** 0.5
        for i in range(2, d + 1):
            k = np.floor(i / 2)
            if i % 2 == 0:
                val += zeta(i) * np.sin(k * np.pi * x / beta_p) * y[i - 1]
            else:
                val += zeta(i) * np.cos(k * np.pi * x / beta_p) * y[i - 1]
        return val

    def a(x):
        return np.exp(a_exponent(x))

    return a



# def FEM_1D_diffusion_QoI(a, g, n):
#     h = 1.0 / (n + 1)
#     x_nodes = np.arange(0, n + 2) * h
#     a_vals  = a(x_nodes)

#     main_diag = (a_vals[:-2] + 2*a_vals[1:-1] + a_vals[2:]) / (2*h)
#     off_diag  = -(a_vals[1:-1] + a_vals[2:]) / (2*h)
#     off_diag  = off_diag[:-1]

#     ab = np.zeros((3, n))
#     ab[0, 1:]  = off_diag
#     ab[1, :]   = main_diag
#     ab[2, :-1] = off_diag

#     c = solve_banded((1, 1), ab, 10.0 * h * np.ones(n))
#     return c[(n - 1) // 2]


def FEM_1D_diffusion_QoI(a, g, n):
    h = 1 / (n + 1)

    x_nodes = np.arange(0, n + 2) * h
    a_vals  = a(x_nodes)
    g_vals  = g(x_nodes)

    Q = np.zeros((6, n + 1))
    i = np.arange(1, n + 1)
    Q[3, i - 1] = (a_vals[i] + a_vals[i - 1]) / (2 * h)
    Q[4, i - 1] = h * (2 * g_vals[i] + g_vals[i - 1]) / 6
    Q[5, i - 1] = h * (2 * g_vals[i] + g_vals[i + 1]) / 6
    Q[3, n]     = (a_vals[n + 1] + a_vals[n]) / (2 * h)

    alpha = np.zeros((n, 1))
    beta  = np.zeros((n - 1, 1))
    b     = np.zeros((n, 1))

    for i in range(1, n):
        alpha[i - 1] = Q[3, i - 1] + Q[3, i] + Q[1, i - 1] + Q[2, i - 1]
        beta[i - 1]  = Q[0, i - 1] - Q[3, i]
        b[i - 1]     = Q[4, i - 1] + Q[5, i - 1]

    alpha[n - 1] = Q[3, n - 1] + Q[3, n] + Q[1, n - 1] + Q[2, n - 1]
    b[n - 1]     = Q[4, n - 1] + Q[5, n - 1]

    ab = np.zeros((3, n))
    ab[0, 1:]  = beta.flatten()
    ab[1, :]   = alpha.flatten()
    ab[2, :-1] = beta.flatten()

    c = solve_banded((1, 1), ab, b)  # solve tridiagonal matrix algorithm
    QoI = c[(n - 1) // 2, 0]
    return QoI
