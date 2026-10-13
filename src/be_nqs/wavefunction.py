"""Multi-determinant VMC for the Be atom with an extensible CAS.

The determinant is

    D_CAS = D[2s]  +  c_p Sum_k D[2p_k]
                   +  c_d Sum_k D[3d_k]      (cas_level >= 2)
                   +  c_f Sum_k D[4f_k]      (cas_level >= 3)

where each group of (2l+1) orbitals enters with equal weight so that the state
remains a 1S singlet. Rather than building the 2l+1 orbitals explicitly, we use
the fact that in D[chi] the orbital appears exactly twice (once in the spin-up
determinant, once in the spin-down one), so the sum over the shell depends only
on the pair contraction

    S_l(i,j) = Sum_k chi_k(r_i) chi_k(r_j)
             = m(r_i) m(r_j) exp(-alpha (r_i + r_j)) * A_l(r_i, r_j)

with, for symmetric traceless Cartesian tensors of rank l,

    A_1 = (r_i . r_j)
    A_2 = (r_i . r_j)^2 - (1/3) r_i^2 r_j^2
    A_3 = (r_i . r_j)^3 - (3/5) r_i^2 r_j^2 (r_i . r_j)

each proportional to r_i^l r_j^l P_l(cos theta_ij) and therefore invariant under
a simultaneous rotation. Expanding the product of the two 2x2 determinants then
gives a single closed expression valid for any l (see shell_determinant).

STRUCTURAL INGREDIENTS

(1) Per-orbital radial modulation.  Each orbital is multiplied by

        m(r) = 1 + c1 t(r) + c2 t(r)^2 ,      t(r) = r^2 / (1 + r)

    Since t(0) = 0 and t'(0) = 0, the factor has unit value and zero slope at
    the origin, so BOTH cusp conditions survive for any values of c1, c2 and to
    any polynomial order. A modulation COMMON to all orbitals is a one-body
    Jastrow exp[sum_i ln m(r_i)] and leaves the nodes unchanged; what is nodal
    is only the modulation of each orbital RELATIVE to the 1s, i.e. the ratios
    m_nl / m_1s (see radial_profiles). m_1s therefore affects the node only
    through these ratios.

(2) Explicit angular channel in the three-body Jastrow: a fifth feature
    t_i t_j cos(theta_ij), regular at the nucleus because of the t_i t_j
    prefactor, symmetric under i <-> j, contributing no slope at coalescence.

PERSISTENCE FOR FIXED-NODE DMC

Everything needed to reconstruct the trial wave function downstream is written
to the .npz file: the flat parameter vector, the configuration (cas_level,
hidden_dim, number of features) and, critically, the CLIP BOUNDS. The bounds
enter through jnp.clip inside the wave function, so a parameter sitting at a
boundary means a different function if the bound is later changed; they are
saved and checked on reload.

The equilibrated walker positions are saved as well, so that the DMC run can
start from configurations already distributed according to |Psi_T|^2.

Use load_wavefunction(path) to obtain log_psi, sign_psi, grad_log_psi,
local_energy and kin_pot bound to the optimised parameters. The loader
recomputes the energy on the saved walkers and compares it with the stored
value as a self-check.
"""

from functools import partial
import time

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import flax.linen as nn
import matplotlib.pyplot as plt
import numpy as np

from . import data_path, results_path

jax.config.update("jax_enable_x64", True)

# Constants and reference energies.
Z_NUC = 4.0
N_ELEC = 4
N_DIM = 3 * N_ELEC

B_MIN, B_MAX = 0.05, 10.0                      # Pade parameter bounds
M1_MIN, M1_MAX = -0.15, 2.50                   # linear modulation bounds
M2_MIN, M2_MAX = -0.15, 0.80                   # quadratic modulation bounds

_i, _j = np.triu_indices(N_ELEC, k=1)          # Six unique electron pairs.
PAIR_I = jnp.array(_i)
PAIR_J = jnp.array(_j)

SPINS = np.array([0, 0, 1, 1])                 # Electrons 0-1 up, 2-3 down.
_parallel = (SPINS[_i] == SPINS[_j])

# Electron-electron cusp coefficients.
A_CUSP = jnp.where(jnp.array(_parallel), 0.25, 0.50)
# Network spin feature: +1 for parallel, -1 for antiparallel.
SPIN_F = jnp.where(jnp.array(_parallel), 1.0, -1.0)

N_FEAT = 5            # t_i+t_j, t_i t_j, t_ij, spin, t_i t_j cos(theta_ij)

# Reference energies (Eh). This is the single source of truth: dmc and
# the analysis scripts import them from here.
E_HF = -14.573023          # Hartree-Fock limit
E_FN_SD = -14.6570         # FN-DMC with single-determinant HF nodes
E_EXACT = -14.667356       # exact non-relativistic (Pachucki & Komasa)
E_CORR = E_EXACT - E_HF    # -0.094333

COL_W = 3.375        # REVTeX single-column width, inches (246 pt)
FULL_W = 7.00        # REVTeX full text width, inches (510 pt)

CAS_LABEL = {1: "2p", 2: "2p + 3d", 3: "2p + 3d + 4f"}

# Bounds are part of the wave-function definition and travel with the file.
BOUND_KEYS = ("B_MIN", "B_MAX", "M1_MIN", "M1_MAX", "M2_MIN", "M2_MAX")


def current_bounds():
    return dict(B_MIN=B_MIN, B_MAX=B_MAX,
                M1_MIN=M1_MIN, M1_MAX=M1_MAX,
                M2_MIN=M2_MIN, M2_MAX=M2_MAX)


# Helpers.
def norm_stable(x):
    return jnp.sqrt(jnp.sum(x ** 2, axis=-1) + 1e-12)


def t_cusp(r):
    """Return a smooth coordinate with zero slope at electron coalescence."""
    return r * r / (1.0 + r)


def radial_mod(t, c1, c2):
    """m(r) = 1 + c1 t + c2 t^2: unit value and zero slope at the nucleus."""
    return (1.0
            + jnp.clip(c1, M1_MIN, M1_MAX) * t
            + jnp.clip(c2, M2_MIN, M2_MAX) * t * t)


def shell_pair_matrix(R, r, alpha, mod, l):
    """S_l(i,j) summed over the 2l+1 real orbitals of a shell, shape (4,4)."""
    rad = jnp.exp(-alpha * r) * mod                # (4,)
    dot = R @ R.T                                  # (4,4)  r_i . r_j
    r2r2 = jnp.outer(r ** 2, r ** 2)               # (4,4)  r_i^2 r_j^2
    if l == 1:
        ang = dot
    elif l == 2:
        ang = dot ** 2 - r2r2 / 3.0
    elif l == 3:
        ang = dot ** 3 - 0.6 * r2r2 * dot
    else:
        raise ValueError("only l = 1, 2, 3 are implemented")
    return jnp.outer(rad, rad) * ang


def shell_determinant(S, phi_1s):
    """Sum_k det_up[1s,chi_k] * det_dn[1s,chi_k], expanded via S_l(i,j)."""
    return (phi_1s[0] * phi_1s[2] * S[1, 3]
            - phi_1s[0] * phi_1s[3] * S[1, 2]
            - phi_1s[1] * phi_1s[2] * S[0, 3]
            + phi_1s[1] * phi_1s[3] * S[0, 2])


# Three-body neural Jastrow.
class NeuralJastrow3Body(nn.Module):
    hidden_dim: int = 16
    envelope_alpha: float = 0.3

    @nn.compact
    def __call__(self, feats):
        # All features are symmetric under i <-> j.
        x = feats
        for _ in range(2):
            # Keep the network parameters in double precision.
            x = nn.Dense(self.hidden_dim, param_dtype=jnp.float64)(x)
            x = nn.silu(x)
        # Start near the baseline without disabling gradients.
        x = nn.Dense(1, param_dtype=jnp.float64,
                     kernel_init=nn.initializers.normal(1e-2),
                     bias_init=nn.initializers.zeros)(x)

        # feats[0] = t_i + t_j , feats[2] = t_ij
        env = jnp.exp(-self.envelope_alpha * (feats[0] + feats[2]))
        return x.squeeze(-1) * env


class OneBodyJastrow(nn.Module):
    """Explicit one-body Jastrow term w(r), summed over electrons.

    Input u = ln(1 + t(r)). Two properties follow from the choice of variable:
      - w'(r = 0) = 0 for any weights, since t'(0) = 0: the e-n cusp stays in
        phi_1s exactly as before;
      - a SiLU network is asymptotically linear in its input, so w ~ lambda ln r
        at large r: a power-law factor r^lambda, which can never make the wave
        function non-normalisable (unlike a term linear in r).
    ln m_1s(r) of the modulated models grows like 2 ln t, i.e. exactly this
    class of functions, so the network can take over the one-body work that
    the optimiser was doing through m_1s.
    """
    hidden_dim: int = 8

    @nn.compact
    def __call__(self, u):
        x = nn.Dense(self.hidden_dim, param_dtype=jnp.float64)(u)
        x = nn.silu(x)
        # Small but non-zero last layer: zero would kill every gradient upstream.
        x = nn.Dense(1, param_dtype=jnp.float64,
                     kernel_init=nn.initializers.normal(1e-2),
                     bias_init=nn.initializers.zeros)(x)
        return x.squeeze(-1)


def init_params(key, mlp, cas_level, j1=None):
    p_mlp = mlp.init(key, jnp.ones((N_FEAT,), dtype=jnp.float64))
    z = lambda v: jnp.asarray(v, dtype=jnp.float64)
    p_orb = {
        "a2s":    z(0.98),
        "a2p":    z(1.00),
        "b_par":  z(0.50),
        "b_anti": z(0.50),
        "cp":     z(0.00),        # Start from one determinant.
        # Per-orbital radial modulation, all starting from m(r) = 1.
        "m1s_1": z(0.00), "m1s_2": z(0.00),
        "m2s_1": z(0.00), "m2s_2": z(0.00),
        "m2p_1": z(0.00), "m2p_2": z(0.00),
    }
    if cas_level >= 2:
        # Hydrogenic guess Z/n = 4/3 for the 3d shell.
        p_orb["a3d"] = z(1.30)
        p_orb["cd"] = z(0.00)
        p_orb["m3d_1"] = z(0.00)
        p_orb["m3d_2"] = z(0.00)
    if cas_level >= 3:
        # Hydrogenic guess Z/n = 1 for the 4f shell.
        p_orb["a4f"] = z(1.00)
        p_orb["cf"] = z(0.00)
        p_orb["m4f_1"] = z(0.00)
        p_orb["m4f_2"] = z(0.00)
    out = {"mlp": p_mlp, "orb": p_orb}
    if j1 is not None:
        # A key derived from, not split off, the main one: models without the
        # one-body term are initialised exactly as before.
        out["j1"] = j1.init(jax.random.fold_in(key, 7),
                            jnp.ones((1,), dtype=jnp.float64))
    return out


def shell_names(cas_level):
    return ["1s", "2s", "2p"] + (["3d"] if cas_level >= 2 else []) \
                              + (["4f"] if cas_level >= 3 else [])


# ============================================================================
# Radial ingredients of the nodal surface.
#
# Since phi_1s > 0 (as long as m_1s > 0), the determinant factorises as
#
#     psi_det = Phi(R) * N(R),        Phi = prod_i phi_1s(r_i) > 0,
#     N = [f(r1) - f(r0)][f(r3) - f(r2)]
#         + sum_l c_l [K_l(0,2) - K_l(0,3) - K_l(1,2) + K_l(1,3)],
#     K_l(i,j) = h_l(r_i) h_l(r_j) A_l(i,j),
#
# with the ratios to the 1s orbital
#
#     f(r)   = phi_2s / phi_1s = (r - c) exp[(Z - a_2s) r] m_2s / m_1s ,
#     h_l(r) = exp[(Z - a_l) r] m_l / m_1s .
#
# The nodal surface {N = 0} depends on f, h_l and c_l only; m_1s enters only
# through the ratios m_nl / m_1s. These helpers evaluate those functions on a
# radial grid with exactly the clipping used inside the wave function.
# ============================================================================
EXP_CLIP = {"2s": (0.30, 3.00), "2p": (0.30, 3.00),
            "3d": (0.30, 5.00), "4f": (0.30, 5.00)}
MIX_KEY = {"2p": "cp", "3d": "cd", "4f": "cf"}
HIGH_SHELLS = ("2p", "3d", "4f")


def _clipped(po, key, lo, hi):
    return float(np.clip(float(po[key]), lo, hi))


def radial_profiles(po, cas_level, r):
    """Evaluate m_nl(r), f(r) and h_l(r) on a numpy grid r.

    Returns a dict with keys 'r', 't', 'm' (dict by shell), 'q' (m_nl/m_1s,
    dict by shell), 'f', 'h' (dict by shell l >= 1), 'c' (2s node radius) and
    the clipped exponents 'alpha' (dict by shell).
    """
    r = np.asarray(r, dtype=float)
    t = r * r / (1.0 + r)
    shells = shell_names(cas_level)

    m = {}
    for s in shells:
        c1 = _clipped(po, f"m{s}_1", M1_MIN, M1_MAX)
        c2 = _clipped(po, f"m{s}_2", M2_MIN, M2_MAX)
        m[s] = 1.0 + c1 * t + c2 * t * t

    alpha = {"1s": Z_NUC}
    for s in shells[1:]:
        lo, hi = EXP_CLIP[s]
        alpha[s] = _clipped(po, f"a{s}", lo, hi)

    c = 1.0 / (Z_NUC - alpha["2s"])
    q = {s: m[s] / m["1s"] for s in shells[1:]}
    f = (r - c) * np.exp((Z_NUC - alpha["2s"]) * r) * q["2s"]
    h = {s: np.exp((Z_NUC - alpha[s]) * r) * q[s]
         for s in shells if s in HIGH_SHELLS}
    return dict(r=r, t=t, m=m, q=q, f=f, h=h, c=c, alpha=alpha)


def radial_diagnostics(po, cas_level, R=None, r_max=15.0, n_grid=6001,
                       mass_tol=1e-5, verbose=True):
    """Check the assumptions behind reading the node as {N = 0}.

    The identity psi_det = Phi * N holds for any parameters. What needs
    checking is where the zeros of the radial factors fall. Any c2 < 0 makes
    m(r) -> -inf, so m has a zero at SOME finite radius r0; what matters is
    whether it lies where the electrons are (with walkers R: fraction of
    electrons beyond r0 above mass_tol; without R: any zero on [0, r_max]).

    (i)  m_1s: Phi = prod_i phi_1s(r_i) is then no longer positive, and the
         node is {N = 0} plus the spheres m_1s(r_i) = 0. -> out['ok'] = False
    (ii) m_nl, nl != 1s: the orbital acquires an extra radial node. The node is
         still {N = 0}, but f (for 2s) or h_l (for l >= 1) changes sign there:
         a genuine nodal feature permitted by the bounds, reported as a note.
    (iii) f(r) strictly increasing in the sampled region. With c_l = 0 the node
         is then exactly {r0 = r1} U {r2 = r3} whatever the radial parameters.
    (iv) raw coefficients sitting at, or beyond, their clip bounds.
    """
    r = np.linspace(0.0, r_max, n_grid)
    prof = radial_profiles(po, cas_level, r)
    r_el = None if R is None else np.linalg.norm(np.asarray(R), axis=-1).ravel()
    r_check = r_max if r_el is None else min(r_max, float(r_el.max()))

    def mass_beyond(r0):
        return None if r_el is None else float(np.mean(r_el > r0))

    out = {"ok": True, "r_check": r_check, "radial_nodes": []}
    for s, ms in prof["m"].items():
        i = int(np.argmin(ms))
        out[f"m_min_{s}"] = float(ms[i])
        out[f"m_argmin_{s}"] = float(r[i])
        neg = np.nonzero(ms <= 0.0)[0]
        r0 = float(r[neg[0]]) if neg.size else np.inf
        out[f"m_zero_{s}"] = r0
        out[f"m_zero_mass_{s}"] = mass_beyond(r0) if neg.size else 0.0
        if neg.size:
            relevant = (r0 <= r_check) if r_el is None else \
                       (out[f"m_zero_mass_{s}"] > mass_tol)
            if relevant:
                if s == "1s":
                    out["ok"] = False
                else:
                    out["radial_nodes"].append(s)

    df = np.diff(prof["f"])
    in_region = r[1:] <= r_check
    out["f_monotone"] = bool(np.all(df[in_region] > 0.0))
    if not out["f_monotone"]:
        out["f_first_turn"] = float(r[1:][in_region][np.argmax(
            df[in_region] <= 0.0)])

    at_bound = []
    for s in shell_names(cas_level):
        for k, lo, hi in ((f"m{s}_1", M1_MIN, M1_MAX),
                          (f"m{s}_2", M2_MIN, M2_MAX)):
            v = float(po[k])
            if v <= lo + 1e-6:
                at_bound.append(f"{k}=lower")
            elif v >= hi - 1e-6:
                at_bound.append(f"{k}=upper")
    out["at_bound"] = at_bound

    if verbose:
        print(f"  radial diagnostics (checked up to r = {r_check:.2f}):")
        for s in shell_names(cas_level):
            r0 = out[f"m_zero_{s}"]
            if np.isinf(r0):
                txt = "no zero on the grid"
            else:
                mb = out[f"m_zero_mass_{s}"]
                txt = f"first zero at r = {r0:.2f}" + (
                    "" if mb is None else f", electron fraction beyond = {mb:.1e}")
            print(f"    m_{s}: min {out[f'm_min_{s}']:+.4f} at "
                  f"r = {out[f'm_argmin_{s}']:.2f};  {txt}")
        print(f"    f(r) strictly increasing up to r = {r_check:.2f}: "
              f"{out['f_monotone']}"
              + ("" if out["f_monotone"]
                 else f"   <-- turns at r = {out['f_first_turn']:.2f}"))
        if at_bound:
            print(f"    coefficients at a bound: {', '.join(at_bound)}")
        for s in out["radial_nodes"]:
            print(f"    NOTE: the {s} orbital has an extra radial node at "
                  f"r = {out[f'm_zero_{s}']:.2f} inside the sampled region. The "
                  f"node is still {{N = 0}},")
            print("          but it now includes where this orbital's radial "
                  "factor changes sign.")
        if not out["ok"]:
            print("    WARNING: m_1s vanishes in the sampled region: Phi is not "
                  "positive and the node is {N = 0} plus the spheres "
                  "m_1s(r_i) = 0.")
    return out


# ============================================================================
# Wave function.  The determinant part and the Jastrow part are computed by a
# single internal routine so that the sign of Psi -- needed by the fixed-node
# constraint in DMC, and discarded by log_psi -- can be exposed without
# duplicating the algebra.  Since exp(J) > 0, sign(Psi) = sign(psi_det).
# ============================================================================
def make_wf_fns(mlp, cas_level, j1=None):

    def _parts(params, R):
        """Return (psi_det, J): the SIGNED determinant and the Jastrow exponent."""
        po = params["orb"]

        # Keep variational exponents in a stable range.
        a1s = Z_NUC                                    # Fixed by the cusp.
        a2s = jnp.clip(po["a2s"], 0.30, 3.00)          # must stay below Z
        a2p = jnp.clip(po["a2p"], 0.30, 3.00)
        b_par = jnp.clip(po["b_par"], B_MIN, B_MAX)
        b_anti = jnp.clip(po["b_anti"], B_MIN, B_MAX)

        r = norm_stable(R)                             # (4,)
        t = t_cusp(r)                                  # (4,)

        # The 2s prefactor enforces the electron-nucleus cusp; the modulation
        # m(r) leaves it untouched because t'(0) = 0. Orbitals with l >= 1
        # vanish at the nucleus and carry no cusp condition, and are orthogonal
        # to the s shells by angular symmetry.
        c = 1.0 / (Z_NUC - a2s)
        m_1s = radial_mod(t, po["m1s_1"], po["m1s_2"])
        m_2s = radial_mod(t, po["m2s_1"], po["m2s_2"])
        m_2p = radial_mod(t, po["m2p_1"], po["m2p_2"])

        phi_1s = jnp.exp(-a1s * r) * m_1s                          # (4,)
        phi_2s = (r - c) * jnp.exp(-a2s * r) * m_2s                # (4,)
        phi_2p = R * (jnp.exp(-a2p * r) * m_2p)[:, None]           # (4,3)

        def det_pair(orb):
            """Return the product of the spin-up and spin-down determinants."""
            d_up = phi_1s[0] * orb[1] - orb[0] * phi_1s[1]
            d_dn = phi_1s[2] * orb[3] - orb[2] * phi_1s[3]
            return d_up * d_dn

        # Equal p-orbital weights preserve rotational symmetry.
        psi_p = (det_pair(phi_2p[:, 0]) + det_pair(phi_2p[:, 1])
                 + det_pair(phi_2p[:, 2]))
        psi_det = det_pair(phi_2s) + po["cp"] * psi_p

        # Higher shells, added through the closed-form contraction.
        if cas_level >= 2:
            a3d = jnp.clip(po["a3d"], 0.30, 5.00)
            m_3d = radial_mod(t, po["m3d_1"], po["m3d_2"])
            S_d = shell_pair_matrix(R, r, a3d, m_3d, l=2)
            psi_det = psi_det + po["cd"] * shell_determinant(S_d, phi_1s)
        if cas_level >= 3:
            a4f = jnp.clip(po["a4f"], 0.30, 5.00)
            m_4f = radial_mod(t, po["m4f_1"], po["m4f_2"])
            S_f = shell_pair_matrix(R, r, a4f, m_4f, l=3)
            psi_det = psi_det + po["cf"] * shell_determinant(S_f, phi_1s)

        # Analytic Jastrow with exact electron-electron cusps.
        r_ij = norm_stable(R[PAIR_I] - R[PAIR_J])      # (6,)
        b = jnp.where(SPIN_F > 0.0, b_par, b_anti)
        u_pade = jnp.sum(A_CUSP * r_ij / (1.0 + b * r_ij))

        # Three-body neural correction.
        t_i = t[PAIR_I]
        t_j = t[PAIR_J]
        t_ij = t_cusp(r_ij)
        # Angular channel: regular at the nucleus thanks to the t_i t_j factor.
        cos_ij = (jnp.sum(R[PAIR_I] * R[PAIR_J], axis=-1)
                  / (r[PAIR_I] * r[PAIR_J]))
        feats = jnp.stack([t_i + t_j,
                           t_i * t_j,
                           t_ij,
                           SPIN_F,
                           t_i * t_j * cos_ij], axis=-1)           # (6,5)
        u_nn = jnp.sum(jax.vmap(lambda f: mlp.apply(params["mlp"], f))(feats))

        J = u_pade + u_nn
        if j1 is not None:
            u1 = jnp.log1p(t)[:, None]                             # (4,1)
            J = J + jnp.sum(jax.vmap(lambda x: j1.apply(params["j1"], x))(u1))
        return psi_det, J

    def log_psi(params, R):
        psi_det, J = _parts(params, R)
        return jnp.log(jnp.abs(psi_det) + 1e-300) + J

    def sign_psi(params, R):
        """sign(Psi) = sign(psi_det), since exp(J) > 0. Needed by fixed-node DMC."""
        psi_det, _ = _parts(params, R)
        return jnp.sign(psi_det)

    def log_psi_and_sign(params, R):
        psi_det, J = _parts(params, R)
        return jnp.log(jnp.abs(psi_det) + 1e-300) + J, jnp.sign(psi_det)

    # Expose (psi_det, J) for the nodal analysis without changing the return
    # signature used by existing callers.
    log_psi.parts = _parts

    return log_psi, sign_psi, log_psi_and_sign


def make_log_psi(mlp, cas_level, j1=None):
    """Backward-compatible wrapper returning only log|Psi|."""
    return make_wf_fns(mlp, cas_level, j1)[0]


def make_grad_log_psi(log_psi_fn):
    """v_D(R) = grad_R ln|Psi_T(R)|, shape (4,3): the DMC drift velocity."""
    def grad_log_psi(params, R):
        return jax.grad(lambda z: log_psi_fn(params, z))(R)
    return grad_log_psi


# Local energy and kinetic/potential components.
def make_energy_fns(log_psi_fn):

    def kin_pot(params, R):
        x = R.reshape(-1)
        f = lambda z: log_psi_fn(params, z.reshape(N_ELEC, 3))
        grad_fn = jax.grad(f)
        g = grad_fn(x)

        # Compute the Laplacian with JVPs instead of forming a full Hessian.
        def body(acc, i):
            e = jnp.zeros_like(x).at[i].set(1.0)
            return acc + jax.jvp(grad_fn, (x,), (e,))[1][i], None

        lap, _ = jax.lax.scan(body, jnp.array(0.0, dtype=x.dtype),
                              jnp.arange(N_DIM))
        T = -0.5 * (lap + jnp.sum(g ** 2))
        V = (-Z_NUC * jnp.sum(1.0 / norm_stable(R))
             + jnp.sum(1.0 / norm_stable(R[PAIR_I] - R[PAIR_J])))
        return T, V

    def local_energy(params, R):
        T, V = kin_pot(params, R)
        return T + V

    return local_energy, kin_pot


# MCMC with single-electron moves.
def make_mcmc(log_psi_fn):

    def sweeps(key, R, params, step, n_sweeps):
        logp0 = 2.0 * log_psi_fn(params, R)

        def one_move(carry, inp):
            R, logp = carry
            i, k = inp
            k1, k2 = jax.random.split(k)
            R_new = R.at[i].add(step * jax.random.normal(k1, (3,), dtype=R.dtype))
            logp_new = 2.0 * log_psi_fn(params, R_new)
            acc = jnp.log(jax.random.uniform(k2, dtype=R.dtype)) < (logp_new - logp)
            return (jnp.where(acc, R_new, R), jnp.where(acc, logp_new, logp)), acc

        keys = jax.random.split(key, n_sweeps * N_ELEC)
        idx = jnp.tile(jnp.arange(N_ELEC), n_sweeps)
        (R, _), accs = jax.lax.scan(one_move, (R, logp0), (idx, keys))
        return R, jnp.mean(accs)

    @partial(jax.jit, static_argnames=("n_sweeps",))
    def batch_sweeps(keys, R_batch, params, step, n_sweeps):
        R, a = jax.vmap(lambda k, r: sweeps(k, r, params, step, n_sweeps))(keys, R_batch)
        return R, jnp.mean(a)

    return batch_sweeps


def adapt_step(step, acc, target=0.55):
    """Adjust the move length toward the target acceptance rate."""
    if acc > target + 0.05:
        step *= 1.05
    elif acc < target - 0.05:
        step /= 1.05
    return float(np.clip(step, 0.02, 3.0))


# Energy, score matrix, and stochastic reconfiguration.
def make_trainer(log_psi_fn, unravel_fn):
    local_energy, kin_pot = make_energy_fns(log_psi_fn)

    @jax.jit
    def energy_and_scores(params_flat, R_batch):
        params = unravel_fn(params_flat)
        eloc = jax.vmap(lambda r: local_energy(params, r))(R_batch)
        score = lambda r: jax.grad(lambda pf: log_psi_fn(unravel_fn(pf), r))(params_flat)
        return eloc, jax.vmap(score)(R_batch)

    @jax.jit
    def virial(params_flat, R_batch):
        params = unravel_fn(params_flat)
        T, V = jax.vmap(lambda r: kin_pot(params, r))(R_batch)
        return jnp.mean(T), jnp.mean(V)

    @jax.jit
    def sr_update(params_flat, O, eloc, lr, gamma, max_norm, mask):
        # Frozen parameters: zeroing their score columns makes g and the
        # corresponding rows/columns of S vanish, so the regularised solve
        # returns exactly delta = 0 for them.
        O = O * mask[None, :]

        # Limit the influence of energetic outliers: clip at 5 MEAN absolute
        # deviations about the median (not the MAD in the strict sense).
        med = jnp.median(eloc)
        mad = jnp.mean(jnp.abs(eloc - med)) + 1e-9
        e = jnp.clip(eloc, med - 5.0 * mad, med + 5.0 * mad)
        e_c = e - jnp.mean(e)

        O_c = O - jnp.mean(O, axis=0, keepdims=True)
        g = 2.0 * jnp.mean(O_c * e_c[:, None], axis=0)

        S = (O_c.T @ O_c) / O.shape[0]
        # Scale damping by each parameter block's diagonal.
        S_reg = S + jnp.diag(gamma * jnp.diag(S) + 1e-8)
        delta = jnp.linalg.solve(S_reg, g)

        # Apply the trust region to the final update.
        update = lr * delta
        update = update * jnp.minimum(1.0, max_norm / (jnp.linalg.norm(update) + 1e-12))
        return params_flat - update, jnp.linalg.norm(update)   # POST-clip norm

    return energy_and_scores, sr_update, virial


def blocking_error(x):
    """Estimate the error using successive blocking."""
    y = np.asarray(x, dtype=float)
    errs = []
    while len(y) >= 8:
        errs.append(y.std(ddof=1) / np.sqrt(len(y)))
        y = y[:len(y) // 2 * 2].reshape(-1, 2).mean(axis=1)
    return max(errs)


# ============================================================================
# Reload interface for downstream fixed-node DMC.
# ============================================================================
def load_wavefunction(path, check=True, verbose=True):
    """Rebuild the optimised trial wave function from a saved .npz file.

    A bare file name is looked up in DATA_DIR; a path with a directory
    component is used as given.

    Returns a dictionary containing
        params          the parameter pytree
        params_flat     the flat parameter vector
        unravel         flat -> pytree
        log_psi         (params, R) -> ln|Psi_T(R)|
        sign_psi        (params, R) -> +/-1, needed by the fixed-node constraint
        log_psi_and_sign
        grad_log_psi    (params, R) -> (4,3), the DMC drift velocity
        local_energy    (params, R) -> E_L(R)
        kin_pot         (params, R) -> (T, V)
        R               equilibrated walkers, shape (n_walkers, 4, 3)
        step            the MCMC step size at the end of the VMC run
        meta            configuration and stored reference values

    If check=True the energy is recomputed on the saved walkers and compared
    with the stored value; a mismatch means the module-level bounds differ from
    those used when the file was written.
    """
    path = data_path(path)
    d = np.load(path)

    cas_level = int(d["cas_level"])
    hidden_dim = int(d["hidden_dim"])
    n_feat = int(d["n_feat"])

    if n_feat != N_FEAT:
        raise ValueError(f"file was written with N_FEAT={n_feat}, "
                         f"module has N_FEAT={N_FEAT}")

    cur = current_bounds()
    mismatched = [k for k in BOUND_KEYS
                  if k in d.files and abs(float(d[k]) - cur[k]) > 1e-12]
    if mismatched and verbose:
        print("  WARNING: clip bounds differ from those used at write time:")
        for k in mismatched:
            print(f"    {k}: file {float(d[k]):+.4f}  module {cur[k]:+.4f}")
        print("  Parameters sitting at a boundary now define a different "
              "function.")

    one_body = bool(d["one_body"]) if "one_body" in d.files else False
    mlp = NeuralJastrow3Body(hidden_dim=hidden_dim)
    j1 = OneBodyJastrow() if one_body else None
    template = init_params(jax.random.PRNGKey(0), mlp, cas_level, j1)
    _, unravel = ravel_pytree(template)
    params_flat = jnp.asarray(d["params_flat"])
    params = unravel(params_flat)

    log_psi, sign_psi, log_psi_and_sign = make_wf_fns(mlp, cas_level, j1)
    grad_log_psi = make_grad_log_psi(log_psi)
    local_energy, kin_pot = make_energy_fns(log_psi)

    R = jnp.asarray(d["R_final"])

    out = dict(params=params, params_flat=params_flat, unravel=unravel,
               mlp=mlp, j1=j1, one_body=one_body, cas_level=cas_level,
               log_psi=log_psi, sign_psi=sign_psi,
               log_psi_and_sign=log_psi_and_sign,
               parts=log_psi.parts,
               grad_log_psi=grad_log_psi,
               local_energy=local_energy, kin_pot=kin_pot,
               R=R, step=float(d["step"]),
               frozen=tuple(str(x) for x in d["frozen"]) if "frozen" in d.files
               else (),
               meta={k: d[k] for k in d.files
                     if d[k].ndim == 0 or k in BOUND_KEYS})

    if check:
        eloc = jax.vmap(lambda r: local_energy(params, r))(R)
        e_now = float(jnp.mean(eloc))
        e_ref = float(d["E"])
        dE = float(d["dE"])
        # The saved walkers are ONE configuration set, so the comparison is at
        # the level of a single-block error sqrt(Var(E_L)/N), not the blocked
        # error bar of the production run.
        sig_block = float(jnp.std(eloc) / jnp.sqrt(R.shape[0]))
        sgn = jax.vmap(lambda r: sign_psi(params, r))(R)
        frac_pos = float(jnp.mean(sgn > 0))
        # Antisymmetry: Psi(P_01 R) = -Psi(R) while |Psi|^2 is invariant, so
        # exactly half of |Psi|^2 lies in each sign domain. A significant
        # deviation from 1/2 signals non-ergodic sampling.
        sig_frac = 0.5 / np.sqrt(R.shape[0])
        z_frac = (frac_pos - 0.5) / sig_frac
        if verbose:
            print(f"  reload check: E(saved walkers) = {e_now:.5f}, "
                  f"stored E = {e_ref:.5f} +/- {dE:.5f}  "
                  f"(diff {(e_now - e_ref)/sig_block:+.1f} single-block sigma)")
            print(f"  sign(Psi): {frac_pos*100:.1f}% positive  "
                  f"({z_frac:+.1f} sigma from 50%)")
            if abs(z_frac) > 3.0:
                print("  WARNING: sign fraction far from 1/2; the walkers may "
                      "be trapped in one nodal domain.")
        if abs(e_now - e_ref) > max(5.0 * sig_block, 20.0 * dE):
            raise RuntimeError(
                f"reloaded wave function gives E = {e_now:.5f} against the "
                f"stored {e_ref:.5f}; the bounds or the module definition have "
                f"probably changed")
        out["E_reload"] = e_now
        out["frac_sign_positive"] = frac_pos
        out["sign_z"] = z_frac

    return out


# Figures.
def setup_plot_style():
    """Set typography for the paper figures."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 6.5,
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "legend.borderpad": 0.35,
        "legend.labelspacing": 0.3,
        "legend.handlelength": 1.6,
        "legend.handletextpad": 0.5,
        "lines.linewidth": 0.9,
        "axes.linewidth": 0.6,
        "grid.linewidth": 0.35,
        "grid.alpha": 0.3,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "figure.dpi": 300,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })


def _save(fig, fname):
    """Write both a vector PDF (for LaTeX) and a raster PNG (for previewing),
    inside RESULTS_DIR."""
    stem = results_path(fname).rsplit(".", 1)[0]
    fig.savefig(f"{stem}.pdf")
    fig.savefig(f"{stem}.png")
    print(f"[saved] {stem}.pdf / {stem}.png")


MIX_STYLE = {"cp": ("#7f3fbf", r"$c_p$"),
             "cd": ("#d68a00", r"$c_d$"),
             "cf": ("#0aa0a0", r"$c_f$")}


def plot_multidet(hist_e, hist_err, hist_var, mix_tracks=None,
                  fname="multidet_optimisation.pdf"):
    """Plot energy, variance, and the CAS mixing coefficients."""
    n = len(hist_e)
    ep = np.arange(1, n + 1)
    e = np.asarray(hist_e)
    err = np.asarray(hist_err)

    if not mix_tracks:
        fig, (a1, a2) = plt.subplots(
            2, 1, figsize=(COL_W, 3.9), sharex=True,
            gridspec_kw={"height_ratios": [3, 2], "hspace": 0.08})
        axes = (a1, a2)
        a3 = None
    else:
        fig, (a1, a2, a3) = plt.subplots(
            3, 1, figsize=(COL_W, 4.8), sharex=True,
            gridspec_kw={"height_ratios": [3, 2, 1.6], "hspace": 0.08})
        axes = (a1, a2, a3)

    # Energy.
    a1.axhline(E_HF, color="black", ls="--", lw=0.7)
    a1.axhline(E_FN_SD, color="crimson", ls="--", lw=0.7)
    a1.axhline(E_EXACT, color="black", ls=":", lw=0.9)

    a1.fill_between(ep, e - err, e + err, color="#2ca02c", alpha=0.25, lw=0)
    a1.plot(ep, e, color="#2ca02c", lw=0.7, label="(D+)")

    for y, txt, col in ((E_HF, " HF", "black"),
                        (E_FN_SD, " FN-DMC, HF nodes", "crimson"),
                        (E_EXACT, " exact", "black")):
        a1.text(0.985, y, txt, transform=a1.get_yaxis_transform(),
                ha="right", va="bottom", fontsize=6, color=col)

    a1.set_ylabel(r"$E$  ($E_h$)")
    a1.set_ylim(E_EXACT - 0.012, -14.55)
    a1.legend(loc="lower right")
    a1.grid(True)

    # Variance.
    a2.semilogy(ep, hist_var, color="#1f77b4", lw=0.7)
    a2.set_ylabel(r"Var($E_L$)")
    a2.grid(True, which="both")

    # Mixing coefficients.
    if a3 is not None:
        for name, track in mix_tracks.items():
            col, lab = MIX_STYLE[name]
            a3.plot(ep, track, color=col, lw=0.8, label=lab)
        a3.set_ylabel("mixing")
        a3.axhline(0.0, color="black", lw=0.4, ls=":")
        a3.legend(loc="lower left", ncol=3, columnspacing=0.8)
        a3.grid(True)

    axes[-1].set_xlabel("Epoch")

    _save(fig, fname)


def modulation_keys(cas_level):
    """Names of all radial-modulation coefficients, e.g. to freeze them at 0."""
    return tuple(f"m{s}_{k}" for s in shell_names(cas_level) for k in (1, 2))


# Training and production pipeline.
def run_pipeline(cas_level=1,
                 n_chains=4000,
                 n_epochs=3500,
                 n_sweeps=10,
                 n_blocks=200,
                 block_sweeps=150,
                 hidden_dim=16,
                 lr0=0.10,
                 gamma=5e-3,
                 max_norm=0.30,
                 seed=42,
                 make_plot=True,
                 tag=None,
                 freeze=(),
                 one_body=False):
    """Optimise and sample one trial wave function.

    freeze: names of orbital parameters (keys of params['orb']) kept at their
    initial value, e.g. freeze=modulation_keys(1) gives the unmodulated
    reference model with everything else identical.
    one_body: add the explicit one-body Jastrow term (OneBodyJastrow).
    """

    if cas_level not in (1, 2, 3):
        raise ValueError("cas_level must be 1 (2p), 2 (2p+3d) or 3 (2p+3d+4f)")
    tag = tag or f"cas{cas_level}"

    key = jax.random.PRNGKey(seed)
    setup_plot_style()

    mlp = NeuralJastrow3Body(hidden_dim=hidden_dim)
    j1 = OneBodyJastrow() if one_body else None
    key, sk = jax.random.split(key)
    params = init_params(sk, mlp, cas_level, j1)
    params_flat, unravel_fn = ravel_pytree(params)
    n_par = params_flat.size

    freeze = tuple(freeze)
    unknown = [k for k in freeze if k not in params["orb"]]
    if unknown:
        raise ValueError(f"cannot freeze unknown parameters: {unknown}")
    mask_tree = jax.tree_util.tree_map(jnp.ones_like, params)
    for k in freeze:
        mask_tree["orb"][k] = jnp.zeros_like(params["orb"][k])
    mask, _ = ravel_pytree(mask_tree)

    print(f"\n{'='*72}")
    print(f"  CAS level {cas_level}:  1s^2 2s^2  +  {CAS_LABEL[cas_level]}")
    print(f"{'='*72}")
    print(f"Variational parameters : {n_par}"
          + (f"  ({len(freeze)} frozen: {', '.join(freeze)})" if freeze else "")
          + ("  + one-body Jastrow" if one_body else ""))
    print(f"Walkers                : {n_chains}  (ratio {n_chains/n_par:.1f}x)")
    if n_chains < 4 * n_par:
        print("  WARNING: SR needs roughly 4-10 samples per parameter.")

    log_psi, sign_psi, _ = make_wf_fns(mlp, cas_level, j1)
    mcmc = make_mcmc(log_psi)
    energy_and_scores, sr_update, virial = make_trainer(log_psi, unravel_fn)

    # Start core electrons near the nucleus and valence electrons farther out.
    key, sk = jax.random.split(key)
    scale = jnp.array([0.4, 0.4, 1.8, 1.8])[None, :, None]
    R = jax.random.normal(sk, (n_chains, N_ELEC, 3), dtype=jnp.float64) * scale

    step = 0.35

    # Burn-in.
    print("\n=== MCMC burn-in ===")
    t0 = time.time()
    for _ in range(15):
        key, sk = jax.random.split(key)
        p = unravel_fn(params_flat)
        R, acc = mcmc(jax.random.split(sk, n_chains), R, p, step, 20)
        step = adapt_step(step, float(acc))
    print(f"acceptance = {float(acc)*100:.1f}%  step = {step:.3f}  "
          f"({time.time()-t0:.1f}s)\n")

    mix_names = ["cp"] + (["cd"] if cas_level >= 2 else []) \
                       + (["cf"] if cas_level >= 3 else [])

    # Training.
    print("=== Training (multi-determinant + three-body neural Jastrow, SR) ===")
    hist_e, hist_err, hist_var = [], [], []
    hist_mix = {name: [] for name in mix_names}
    t0 = time.time()

    for epoch in range(n_epochs):
        key, sk = jax.random.split(key)
        p = unravel_fn(params_flat)
        R, acc = mcmc(jax.random.split(sk, n_chains), R, p, step, n_sweeps)
        step = adapt_step(step, float(acc))

        eloc, O = energy_and_scores(params_flat, R)
        e_mean = float(jnp.mean(eloc))
        e_var = float(jnp.var(eloc))
        e_err = float(jnp.std(eloc) / jnp.sqrt(n_chains))

        hist_e.append(e_mean)
        hist_err.append(e_err)
        hist_var.append(e_var)

        # Decay the learning rate and trust-region radius together.
        d = 0.05 + 0.95 * 0.5 * (1.0 + np.cos(np.pi * epoch / n_epochs))
        params_flat, upd_norm = sr_update(params_flat, O, eloc,
                                          lr0 * d, gamma, max_norm * d, mask)

        po = unravel_fn(params_flat)["orb"]
        for name in mix_names:
            hist_mix[name].append(float(po[name]))

        if epoch == 0 or (epoch + 1) % 100 == 0:
            mix_str = "  ".join(f"{name} = {float(po[name]):+.3f}"
                                for name in mix_names)
            print(f"ep {epoch+1:5d} | E = {e_mean:10.5f} +/- {e_err:.5f} "
                  f"| var = {e_var:6.3f} | |upd| = {float(upd_norm):.4f} "
                  f"| {mix_str} | acc = {float(acc)*100:.0f}%")

    print(f"\nTraining in {time.time()-t0:.1f}s")

    # Production sampling.
    # T and V are accumulated over ALL blocks: their individual variances are
    # orders of magnitude larger than Var(E_L), because the divergences at
    # coalescence cancel in the sum but not in the separate terms. A single
    # snapshot gives <T> with an uncertainty of order 0.5 Eh and is useless.
    print(f"\n=== Final estimate ({n_blocks} blocks, {block_sweeps} sweeps apart) ===")
    p = unravel_fn(params_flat)
    blocks, vars_, blocks_T, blocks_V = [], [], [], []
    for _ in range(n_blocks):
        key, sk = jax.random.split(key)
        R, _ = mcmc(jax.random.split(sk, n_chains), R, p, step, block_sweeps)
        eloc, _ = energy_and_scores(params_flat, R)
        Tb, Vb = virial(params_flat, R)
        blocks.append(float(jnp.mean(eloc)))
        vars_.append(float(jnp.var(eloc)))
        blocks_T.append(float(Tb))
        blocks_V.append(float(Vb))

    E = float(np.mean(blocks))
    dE = blocking_error(blocks)
    var = float(np.mean(vars_))
    T = float(np.mean(blocks_T))
    V = float(np.mean(blocks_V))
    dT = blocking_error(blocks_T)
    dV = blocking_error(blocks_V)
    vt = V / T
    dvt = abs(vt) * np.sqrt((dT / T) ** 2 + (dV / V) ** 2)

    corr = (E - E_HF) / (E_EXACT - E_HF) * 100.0
    po = unravel_fn(params_flat)["orb"]

    # Fraction of walkers in each nodal pocket: a first look at the nodal
    # surface that the DMC run will be constrained to.
    sgn = jax.vmap(lambda r: sign_psi(p, r))(R)
    frac_pos = float(jnp.mean(sgn > 0))

    print(f"E_VMC = {E:.5f} +/- {dE:.5f} Eh")
    print(f"Var(E_loc)                     : {var:.4f}")
    print(f"Correlation energy recovered   : {corr:.2f}%")
    print(f"Distance from the exact value  : {E - E_EXACT:+.5f} Eh")
    print(f"Virial ratio V/T               : {vt:+.4f} +/- {dvt:.4f}   (exact -2)")
    print(f"  <T> = {T:.5f} +/- {dT:.5f}")
    print(f"  <V> = {V:.5f} +/- {dV:.5f}")
    print(f"sign(Psi) on final walkers     : {frac_pos*100:.1f}% positive")

    print("\nOptimised parameters:")
    print(f"  c_p    = {float(po['cp']):+.4f}   (2p^2 weight)")
    if cas_level >= 2:
        print(f"  c_d    = {float(po['cd']):+.4f}   (3d^2 weight)")
    if cas_level >= 3:
        print(f"  c_f    = {float(po['cf']):+.4f}   (4f^2 weight)")
    print(f"  a2s    = {float(po['a2s']):.4f}")
    print(f"  a2p    = {float(po['a2p']):.4f}")
    if cas_level >= 2:
        print(f"  a3d    = {float(po['a3d']):.4f}")
    if cas_level >= 3:
        print(f"  a4f    = {float(po['a4f']):.4f}")

    print("  radial modulation  m(r) = 1 + c1 t + c2 t^2")
    shells = shell_names(cas_level)
    for s in shells:
        c1 = float(np.clip(po[f"m{s}_1"], M1_MIN, M1_MAX))
        c2 = float(np.clip(po[f"m{s}_2"], M2_MIN, M2_MAX))
        flag = ""
        if c1 <= M1_MIN + 1e-6 or c1 >= M1_MAX - 1e-6:
            flag += "  <-- c1 AT BOUND"
        if c2 <= M2_MIN + 1e-6 or c2 >= M2_MAX - 1e-6:
            flag += "  <-- c2 AT BOUND"
        print(f"    {s}:  c1 = {c1:+.4f}   c2 = {c2:+.4f}{flag}")

    bp = float(np.clip(po["b_par"], B_MIN, B_MAX))
    ba = float(np.clip(po["b_anti"], B_MIN, B_MAX))
    print(f"  b_par  = {bp:.4f}"
          + ("   <-- AT BOUND" if bp >= B_MAX - 1e-6 else ""))
    print(f"  b_anti = {ba:.4f}"
          + ("   <-- AT BOUND" if ba >= B_MAX - 1e-6 else ""))

    diag = radial_diagnostics(po, cas_level, R=R)

    # One-body factor carried by the Jastrow (if any) next to the one carried
    # by m_1s: both are ln of a positive one-body factor, directly comparable.
    r_grid = np.linspace(0.0, 6.0, 121)
    t_grid = r_grid ** 2 / (1.0 + r_grid)
    ln_m1s = np.log(np.abs(1.0 + np.clip(float(po["m1s_1"]), M1_MIN, M1_MAX) * t_grid
                           + np.clip(float(po["m1s_2"]), M2_MIN, M2_MAX) * t_grid ** 2))
    if one_body:
        pj = unravel_fn(params_flat)["j1"]
        w = np.asarray(jax.vmap(lambda x: j1.apply(pj, x))(
            jnp.log1p(jnp.asarray(t_grid))[:, None]))
        w = w - w[0]
        print(f"  one-body Jastrow w(r) - w(0) at r = 1, 2, 4: "
              f"{np.interp(1, r_grid, w):+.3f} {np.interp(2, r_grid, w):+.3f} "
              f"{np.interp(4, r_grid, w):+.3f}   (ln m_1s: "
              f"{np.interp(1, r_grid, ln_m1s):+.3f} "
              f"{np.interp(2, r_grid, ln_m1s):+.3f} "
              f"{np.interp(4, r_grid, ln_m1s):+.3f})")
    else:
        w = np.zeros_like(r_grid)

    # Save diagnostics and results.
    if make_plot:
        plot_multidet(hist_e, hist_err, hist_var,
                      mix_tracks={k: np.array(v) for k, v in hist_mix.items()},
                      fname=f"multidet_{tag}.pdf")

    out = dict(E=E, dE=dE, var=var, T=T, V=V, dT=dT, dV=dV, vt=vt, dvt=dvt,
               n_par=n_par, cas_level=cas_level, corr=corr,
               a2s=float(po["a2s"]), a2p=float(po["a2p"]),
               cp=float(po["cp"]), b_par=bp, b_anti=ba,
               frac_sign_positive=frac_pos,
               hist_e=np.array(hist_e), hist_var=np.array(hist_var),
               hist_err=np.array(hist_err), blocks=np.array(blocks),
               blocks_T=np.array(blocks_T), blocks_V=np.array(blocks_V))
    for s in shells:
        out[f"m{s}_1"] = float(po[f"m{s}_1"])
        out[f"m{s}_2"] = float(po[f"m{s}_2"])
    if cas_level >= 2:
        out.update(cd=float(po["cd"]), a3d=float(po["a3d"]))
    if cas_level >= 3:
        out.update(cf=float(po["cf"]), a4f=float(po["a4f"]))
    for k, v in hist_mix.items():
        out[f"hist_{k}"] = np.array(v)

    # --- everything the DMC needs to rebuild this exact wave function -------
    out.update(params_flat=np.asarray(params_flat),
               R_final=np.asarray(R),
               step=step,
               hidden_dim=hidden_dim,
               n_feat=N_FEAT,
               n_chains=n_chains,
               n_epochs=n_epochs,
               seed=seed,
               frozen=np.array(freeze, dtype=str),
               one_body=bool(one_body),
               onebody_r=r_grid, onebody_w=w, onebody_ln_m1s=ln_m1s,
               radial_ok=bool(diag["ok"]),
               f_monotone=bool(diag["f_monotone"]),
               **{k: v for k, v in diag.items()
                  if k.startswith(("m_min_", "m_zero_")) and v is not None},
               **current_bounds())

    np.savez(data_path(f"multidet_{tag}.npz"), **out)
    print(f"[saved] multidet_{tag}.npz  "
          f"({n_par} params + {n_chains} equilibrated walkers)")

    return out


def main(cas_levels=(1,), n_chains=4000, n_epochs=3500, n_blocks=200,
         seed=42, tag_prefix="", freeze=(), one_body=False):
    """Run the requested CAS levels and summarise."""
    results = []
    for level in cas_levels:
        tag = f"{tag_prefix}cas{level}" if tag_prefix else f"cas{level}"
        fz = modulation_keys(level) if freeze == "modulation" else freeze
        results.append(run_pipeline(cas_level=level, n_chains=n_chains,
                                    n_epochs=n_epochs, n_blocks=n_blocks,
                                    seed=seed, tag=tag, freeze=fz,
                                    one_body=one_body))

    print(f"\n{'='*92}\n  SUMMARY\n{'='*92}")
    print(f"{'active space':<18}{'par.':>6}{'E (Eh)':>13}{'err':>9}"
          f"{'var':>9}{'corr %':>9}{'dE vs prev':>12}{'V/T':>11}{'err':>9}")
    print("-" * 92)
    prev = None
    for r in results:
        gain = "" if prev is None else f"{(r['E']-prev)*1000:+9.3f} mHa"
        print(f"{CAS_LABEL[r['cas_level']]:<18}{r['n_par']:>6}{r['E']:>13.5f}"
              f"{r['dE']:>9.5f}{r['var']:>9.4f}{r['corr']:>9.2f}"
              f"{gain:>12}{r['vt']:>11.4f}{r['dvt']:>9.4f}")
        prev = r["E"]
    print("-" * 92)
    print(f"{'exact':<18}{'':>6}{E_EXACT:>13.5f}")
    print("\nReference: Seth, Lopez Rios, Needs, JCP 134, 084105 (2011)")
    print("  VMC -14.66719(1), 99.82%, Var 0.00526 (multi-det + Jastrow + backflow)")
    print("  their multi-det Jastrow WITHOUT backflow has Var 0.01066")

    if len(results) > 1:
        np.savez(data_path("cas_scan_summary.npz"),
                 levels=np.array([r["cas_level"] for r in results]),
                 E=np.array([r["E"] for r in results]),
                 dE=np.array([r["dE"] for r in results]),
                 var=np.array([r["var"] for r in results]),
                 n_par=np.array([r["n_par"] for r in results]))
        print("[saved] cas_scan_summary.npz")
    return results


if __name__ == "__main__":
    main(cas_levels=(1,), tag_prefix="quad_")     # modulazione quadratica
    #main(cas_levels=(1,), tag_prefix="lin_")
    # Unmodulated reference (same Jastrow, same CAS, m(r) = 1 for every
    # orbital). Its FN-DMC energy isolates the NODAL effect of the modulation.
    #main(cas_levels=(1,), tag_prefix="nomod_", freeze="modulation")