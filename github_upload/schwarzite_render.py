"""
Load a CIF structure, print geometry summary, and render 3D atom positions.
Milestone 2 adds an explicit cutoff-based adjacency graph under PBC.
"""

import json
import math
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")  # No interactive window
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import plotly.graph_objects as go
from scipy.spatial import ConvexHull, HalfspaceIntersection
from ase.io import read
from ase import Atoms
from ase.neighborlist import neighbor_list

# Repo root (directory containing this file); keeps CIF resolvable after clone.
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
CIF_PATH = os.path.join(_REPO_ROOT, "data", "cif", "AFY_nozeolite_relaxed.cif")

# --- Mode-isolated output directories (set at runtime by main()) ---
FIG_DIR = "figures"
DATA_DIR = "data"

OUT_FIG = "structure_preview.png"
OUT_HTML = "graph_interactive.html"
EXPORT_INTERACTIVE = True  # set True by default; easy to toggle
CUT = 1.7  # Angstrom
CUT_SCAN = [1.2, 1.4, 1.6, 1.8, 2.0, 2.2]  # Angstrom
T_HOP = 1.0              # 1st-shell (NN) hopping amplitude
T_HOP_2NN = 1.0          # 2nd-nearest-neighbour hopping amplitude (nn+2nn model)
T_HOP_2NN_BDRY = 1.0     # 2NN boundary-only hopping amplitude (nn+2nn-bdry model)
T_HOP_4NN_BDRY = 1.0     # 4NN boundary-only hopping amplitude (nn+4nn-bdry model)
T_HOP_ALPHA    = 0.5     # geometric decay factor for nn+2nn+3nn+4nn model
MANUAL_T_HOP        = 1.0    # amplitude for manually-selected equivalent pairs
MANUAL_MATCH_TOL    = 1e-3   # Angstrom: tolerance for distance-equivalent matching
MANUAL_SHIFT_SEARCH = [-1, 0, 1]  # integer shifts searched per axis for min-image
ONSITE = 0.0             # uniform onsite energy epsilon
MAX_CUTOFF_2NN = 10.0    # hard cap on auto-expanded cutoff for 2NN/4NN search

# --- Manual cross-sheet hoppings ---
EXTRA_HOPS: list[dict] = []
EXTRA_HOPS_MODE = "unitcell"   # "unitcell": force S=(0,0,0), dr=pos[j]-pos[i]; "min_image": minimum-image S
EXTRA_HOPS_AMP = 0.3
EXTRA_HOPS_PAIRS = [(9, 52), (22, 83), (55, 68), (29, 17)]
SAVE_EVALS = True
SAVE_H = True
OUT_H_TEMPLATE = "H_cut{cut:.2f}.npy"
DO_BAND_STRUCTURE = True
OUT_BANDS = "bands_cut{cut:.2f}.png"
DO_PATHSUM_BANDS = False
PATH_L_MAX = 5
PATH_ALPHA = 0.6  # weight factor for length ℓ term: wℓ = alpha^(ℓ-1)
OUT_PATHSUM_BANDS = "bands_pathsum_L{L}_cut{cut:.2f}.png"
DO_SHELL_BANDS = False
SHELL_L_MAX = 5
SHELL_TOL = 1e-3  # fine deduplication tolerance (DO_SHELL_BANDS legacy)
SHELL_TOL_ABS = 5e-3  # absolute tolerance in Angstrom (DO_SHELL_BANDS)
SHELL_TOL_REL = 2e-3  # relative tolerance (DO_SHELL_BANDS)
SHELL_GAP_TOL  = 0.3  # minimum gap (Ang) separating two distinct bond-length shells
#   Within the NN shell of this schwarzite bond lengths span ~0.05-0.15 Ang;
#   the NN->2NN gap is ~0.7-0.9 Ang, so 0.3 Ang cleanly separates shells.
SHELL_CUTOFF = 6.0  # angstrom, ensure enough for 5 shells
SHELL_DECAY = 0.6
SHELL_MIN_HOPS = None  # auto: max(12, N//6); or set explicit threshold for NN shell selection
SHELL_ENERGY_WINDOW = 4.0  # Plot bands within [-window, +window] around E=0 (set None for all bands)
OUT_SHELL_BANDS = "bands_shells_L{L}_cut{cut:.2f}.png"
DO_KPATH_VIZ = True
# Overlay a legacy comparison loop on the k-path BZ figure (off by default; paper path only).
DO_KPATH_COMPARE_SECOND = False
OUT_KPATH_PNG = "kpath_reciprocal.png"
OUT_KPATH_HTML = "kpath_reciprocal.html"
OUT_BZ_RECIP_BASIS_PNG = "first_bz_reciprocal_basis.png"
OUT_BZ_RECIP_BASIS_HTML = "first_bz_reciprocal_basis.html"
# Wireframe first BZ + bold primitive reciprocal cell (PNG + interactive HTML), basis only — no k-path.
DO_BZ_WIREFRAME_VIZ = True
OUT_BZ_WIREFRAME_PNG = "bz_wireframe_basis.png"
OUT_BZ_WIREFRAME_HTML = "bz_wireframe_basis.html"

# --- True first Brillouin zone (Wigner–Seitz cell about Γ in reciprocal space) ---
USE_TRUE_FIRST_BZ = True
TRUE_BZ_G_SHELLS_HALFSPACE = 2   # planes from ±shell integer combos (matches render_true_bz.py)
TRUE_BZ_G_SHELLS_REDUCE = 3      # shells for iterative k → first-BZ reduction

# --- Topology diagnostics (gauge-invariant) ---
DO_TOPOLOGY = True
TOPO_MODE = "both"  # "slice", "weyl", or "both"
N_K = 21  # grid size per dimension for slice scans (odd helps centering)
SLICE_AXIS = "kz"  # slice normal axis: "kx", "ky", "kz"
SLICE_VALUE = 0.0  # fractional coord of reciprocal basis (0..1)
N_OCC = None  # number of occupied bands for multi-band Chern; None = single band near E_F
EF_MODE = "half_filling"  # "half_filling" or "fixed"
EF_FIXED = 0.0
CHERN_TOL = 0.1  # report near-integer if |C - round(C)| < tol
OUT_CHERN_TXT = "chern_results.txt"
OUT_BERRY_PNG = "berry_flux_slice.png"
N_K3 = 15  # coarse 3D scan resolution for Weyl search
GAP_BANDS = "near_EF"  # (m, m+1) tuple or "near_EF"
GAP_EF = 0.0
GAP_THRESH = 1e-2  # candidate if local direct gap < thresh
MAX_CANDIDATES = 30
SPHERE_RADIUS_FRAC = 0.05  # sphere radius as fraction of average |b|
N_THETA = 21
N_PHI = 2 * N_THETA
OUT_WEYL_TXT = "weyl_candidates.txt"

# --- Robust Weyl detection pipeline ---
WEYL_SCAN_MODE = "min_gap_window"  # "near_EF" (legacy), "min_gap_window" (new robust)
WEYL_BAND_WINDOW = 15              # DEPRECATED (kept for backwards compat; scan now covers all pairs)
WEYL_ENERGY_WINDOW = None           # optional energy filter: only keep candidates with |E_mid - EF| < this
WEYL_TOPK = 30                     # keep top-K smallest-gap k-points regardless of threshold
WEYL_THRESH_INIT = 3e-2            # initial exploration threshold for window mode
WEYL_REPORT = "weyl_scan_summary.txt"
WEYL_REFINED = "weyl_refined_candidates.txt"
N_K3_LIST = [15, 21, 31]           # resolution sweep grid sizes
WEYL_REFINE_SEEDS = 5              # number of coarse candidates to refine locally
WEYL_REFINE_LOCAL_N = 11           # local refinement grid per axis
WEYL_REFINE_DELTA = None           # auto: 0.5/nk3 if None
WEYL_SPHERE_TOP = 3                # number of refined candidates for sphere validation
WEYL_SPHERE_RADII = [0.08, 0.05, 0.03, 0.02]  # radius fracs for sweep
WEYL_SPHERE_NTHETA_LIST = [21, 27]             # angular resolutions for convergence
WEYL_NEG_CONTROL = True            # run NN-only negative control scan

# --- Rigorous Weyl verification pipeline (task 5) ---
VERIFY_RADII = [1e-2, 7e-3, 5e-3, 3e-3, 2e-3, 1e-3, 7e-4, 5e-4, 3e-4]
VERIFY_NTHETA_MIN = 15
VERIFY_NTHETA_ALPHA = 0.15       # n_theta = max(min, ceil(alpha/r_cart))
VERIFY_NTHETA_MAX = 200
VERIFY_Q_TOL = 0.05              # |Q(nt1)-Q(nt2)| tolerance
VERIFY_GAP_SURFACE_TOL = 5e-4    # min gap on sphere surface for reliability
VERIFY_MINIMIZE_GAP = True       # run scipy local minimiser to find k*
VERIFY_MINIMIZER_BOX_FRAC = 0.01 # coarse grid half-width (fractional)
VERIFY_MINIMIZER_BOUNDS_FRAC = 0.02  # scipy bounds half-width (fractional)
VERIFY_CUT_HALFWIDTH = 0.02      # dispersion cut half-width (fractional of avg |b|)

# --- Fermi-level zoom band plot ---
DO_BANDS_FERMI_ZOOM = True
N_SHOW = 8
OUT_BANDS_ZOOM = "bands_zoom_cut{cut:.2f}_N{nshow}.png"
GAP_TOL = 1e-3

# --- Reciprocal-space naming (plots / band-path ticks): primitive reciprocal rows **a**, **b**, **c**
# (numeric rows = ``reciprocal_lattice(cell)`` row 0,1,2). Half-way points X,Y,Z at ½**a**, ½**b**, ½**c**.
RECIP_VEC_LABEL_MPL = (r"$\mathbf{a}$", r"$\mathbf{b}$", r"$\mathbf{c}$")
RECIP_VEC_LABEL_PLAIN = ("a", "b", "c")
DEFAULT_KPATH_VERTEX_LABELS = ["Y", "Γ", "Z", "W₁", "W₂", "W₃", "W₄"]
KPATH_LEGEND_STD = "Y–Γ–Z–W₁–W₂–W₃–W₄"
KPATH_LEGEND_PRIMITIVE_LOOP = "Γ–a–a+b–b–Γ"

# High-symmetry + Weyl nodes (fractional reciprocal coords f, k_cart = f @ reciprocal rows).
# Matches external BZ analysis / Fig. 4 band path: Y → Γ → Z → W₁ → W₂ → W₃ → W₄.
REFERENCE_HS_POINTS: list[tuple[str, tuple[float, float, float]]] = [
    ("Γ", (0.0, 0.0, 0.0)),
    ("X", (0.5, 0.0, 0.0)),
    ("Y", (0.0, 0.5, 0.0)),
    ("Y₂", (0.0, -0.5, 0.0)),
    ("Z", (0.0, 0.0, 0.5)),
    ("V₂", (0.5, -0.5, 0.0)),
    ("U₂", (-0.5, 0.0, 0.5)),
    ("T₂", (0.0, -0.5, 0.5)),
    ("R₂", (-0.5, -0.5, 0.5)),
]
REFERENCE_WEYL_POINTS: list[tuple[str, tuple[float, float, float]]] = [
    ("W₁", (0.1224448, -0.2358748, -0.49708016)),
    ("W₂", (-0.12272477, 0.23539968, 0.49615932)),
    ("W₃", (0.07071221, 0.1808957, -0.25261623)),
    ("W₄", (-0.07033161, -0.18122039, 0.25396826)),
]
PAPER_KPATH_NODES: list[tuple[str, tuple[float, float, float]]] = [
    ("Y", (0.0, 0.5, 0.0)),
    ("Γ", (0.0, 0.0, 0.0)),
    ("Z", (0.0, 0.0, 0.5)),
    *REFERENCE_WEYL_POINTS,
]
DO_REFERENCE_WEYL_MARKERS = True
REFERENCE_WEYL_COLOR = "#e91e63"   # distinct magenta on wireframe


def _set_output_dirs(mode_name: str) -> None:
    """Redirect all output paths into results/<mode_name>/figures and results/<mode_name>/data."""
    global FIG_DIR, DATA_DIR
    global OUT_FIG, OUT_HTML, OUT_H_TEMPLATE, OUT_BANDS, OUT_PATHSUM_BANDS
    global OUT_SHELL_BANDS, OUT_KPATH_PNG, OUT_KPATH_HTML
    global OUT_BZ_RECIP_BASIS_PNG, OUT_BZ_RECIP_BASIS_HTML
    global OUT_BZ_WIREFRAME_PNG, OUT_BZ_WIREFRAME_HTML
    global OUT_CHERN_TXT, OUT_BERRY_PNG, OUT_WEYL_TXT
    global WEYL_REPORT, WEYL_REFINED, OUT_BANDS_ZOOM

    base = os.path.join("results", mode_name)
    FIG_DIR = os.path.join(base, "figures")
    DATA_DIR = os.path.join(base, "data")
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    OUT_FIG = os.path.join(FIG_DIR, "structure_preview.png")
    OUT_HTML = os.path.join(FIG_DIR, "graph_interactive.html")
    OUT_H_TEMPLATE = os.path.join(DATA_DIR, "H_cut{cut:.2f}.npy")
    OUT_BANDS = os.path.join(FIG_DIR, "bands_cut{cut:.2f}.png")
    OUT_PATHSUM_BANDS = os.path.join(FIG_DIR, "bands_pathsum_L{L}_cut{cut:.2f}.png")
    OUT_SHELL_BANDS = os.path.join(FIG_DIR, "bands_shells_L{L}_cut{cut:.2f}.png")
    OUT_KPATH_PNG = os.path.join(FIG_DIR, "kpath_reciprocal.png")
    OUT_KPATH_HTML = os.path.join(FIG_DIR, "kpath_reciprocal.html")
    OUT_BZ_RECIP_BASIS_PNG = os.path.join(FIG_DIR, "first_bz_reciprocal_basis.png")
    OUT_BZ_RECIP_BASIS_HTML = os.path.join(FIG_DIR, "first_bz_reciprocal_basis.html")
    OUT_BZ_WIREFRAME_PNG = os.path.join(FIG_DIR, "bz_wireframe_basis.png")
    OUT_BZ_WIREFRAME_HTML = os.path.join(FIG_DIR, "bz_wireframe_basis.html")
    OUT_CHERN_TXT = os.path.join(DATA_DIR, "chern_results.txt")
    OUT_BERRY_PNG = os.path.join(FIG_DIR, "berry_flux_slice.png")
    OUT_WEYL_TXT = os.path.join(DATA_DIR, "weyl_candidates.txt")
    WEYL_REPORT = os.path.join(DATA_DIR, "weyl_scan_summary.txt")
    WEYL_REFINED = os.path.join(DATA_DIR, "weyl_refined_candidates.txt")
    OUT_BANDS_ZOOM = os.path.join(FIG_DIR, "bands_zoom_cut{cut:.2f}_N{nshow}.png")

    print(f"Output directory: {base}/")


def add_manual_hop(
    atoms: Atoms,
    i: int,
    j: int,
    amplitude: float,
    S: tuple[int, int, int] | None = None,
) -> None:
    """Add a manual hopping between site i and j.

    Behaviour depends on EXTRA_HOPS_MODE:
      "unitcell" - force S=(0,0,0), dr = pos[j]-pos[i] (in-cell vector).
      "min_image" - if S is None, compute minimum-image S; dr = dr0 + S·cell.
    Stores S, dr, and amplitude for Bloch phase and visualization.
    """
    cell = atoms.get_cell().array
    pos = atoms.get_positions()
    dr0 = pos[j] - pos[i]

    if EXTRA_HOPS_MODE == "unitcell":
        S = (0, 0, 0)
        dr = dr0.copy()
    else:
        if S is None:
            df = np.linalg.solve(cell.T, dr0)
            S = tuple(int(x) for x in (-np.round(df)).tolist())
        sx, sy, sz = S
        dr = dr0 + float(sx) * cell[0] + float(sy) * cell[1] + float(sz) * cell[2]

    EXTRA_HOPS.append({
        "i": i,
        "j": j,
        "S": S,
        "dr": dr,
        "amplitude": amplitude,
    })
    print(f"Added manual hop [{EXTRA_HOPS_MODE}]: ({i} -> {j}), S={S}, "
          f"|dr|={np.linalg.norm(dr):.4f}, amp={amplitude}")


def load_cif(path: str) -> Atoms:
    """Load a CIF file with ASE. Exit with error if file is missing."""
    if not os.path.isfile(path):
        print(f"Error: CIF file not found: {path}")
        cif_dir = os.path.dirname(path)
        if os.path.isdir(cif_dir):
            print(f"Files in {cif_dir}/:")
            for f in sorted(os.listdir(cif_dir)):
                print(f"  {f}")
        else:
            print(f"Directory does not exist: {cif_dir}")
        sys.exit(1)
    return read(path)


def summarize(atoms: Atoms) -> None:
    """Print atom count, elements, PBC, cell matrix, lengths, and angles."""
    n = len(atoms)
    symbols = atoms.get_chemical_symbols()
    unique = sorted(set(symbols))
    print(f"Atom count: {n}")
    print(f"Elements: {unique}")

    pbc = atoms.get_pbc()
    print(f"PBC: {tuple(pbc)}")

    cell = atoms.get_cell()
    print("Cell matrix (Angstrom):")
    for i, row in enumerate(cell):
        print(f"  [{row[0]:12.6f} {row[1]:12.6f} {row[2]:12.6f}]")

    lengths = cell.lengths()
    angles = cell.angles()
    print(f"Cell lengths (a, b, c): {lengths[0]:.4f} {lengths[1]:.4f} {lengths[2]:.4f} Angstrom")
    print(f"Cell angles (alpha, beta, gamma): {angles[0]:.2f} {angles[1]:.2f} {angles[2]:.2f} deg")


def build_graph(atoms: Atoms, cutoff: float) -> dict[int, set[int]]:
    """
    Build an explicit cutoff-based adjacency graph under PBC using ASE neighbor_list.

    Notes:
    - This is purely geometric: pairs within `cutoff` (Angstrom) are connected.
    - No chemistry-based heuristics are used.
    """
    n = len(atoms)
    adj: dict[int, set[int]] = {i: set() for i in range(n)}

    ii, jj, _S = neighbor_list("ijS", atoms, cutoff)
    for i, j in zip(ii.tolist(), jj.tolist()):
        if i == j:
            continue
        adj[i].add(j)
        adj[j].add(i)
    return adj


def _count_connected_components(adj: dict[int, set[int]]) -> int:
    """Count connected components in an undirected graph via DFS."""
    visited: set[int] = set()
    n_comp = 0

    for start in adj.keys():
        if start in visited:
            continue
        n_comp += 1
        stack = [start]
        visited.add(start)
        while stack:
            v = stack.pop()
            for w in adj[v]:
                if w not in visited:
                    visited.add(w)
                    stack.append(w)
    return n_comp


def graph_stats(adj: dict[int, set[int]]) -> dict[str, object]:
    """
    Print basic graph diagnostics and return the computed stats.

    Prints:
    - N nodes, E edges (undirected)
    - degree list summary: min/mean/max
    - number of connected components
    """
    n = len(adj)
    degrees = [len(adj[i]) for i in range(n)]
    e = int(sum(degrees) // 2)
    deg_min = int(min(degrees)) if degrees else 0
    deg_mean = float(np.mean(degrees)) if degrees else 0.0
    deg_max = int(max(degrees)) if degrees else 0
    n_comp = _count_connected_components(adj)

    print(f"Graph: N={n} nodes, E={e} edges (undirected)")
    print(f"Degree: min/mean/max = {deg_min}/{deg_mean:.2f}/{deg_max}")
    print(f"Connected components: {n_comp}")

    return {
        "N": n,
        "E": e,
        "degrees": degrees,
        "deg_min": deg_min,
        "deg_mean": deg_mean,
        "deg_max": deg_max,
        "n_components": n_comp,
    }


def build_tb_hamiltonian(adj: dict[int, set[int]], t: float, onsite: float) -> np.ndarray:
    """
    Build a nearest-neighbor tight-binding Hamiltonian from an adjacency graph.

    H[i,j] = -t if (i,j) is an edge, H[i,i] = onsite.
    Returns a symmetric matrix.
    """
    n = len(adj)
    h = np.zeros((n, n), dtype=float)

    for i in range(n):
        h[i, i] = onsite
        for j in adj[i]:
            if i < j:  # process each undirected edge once
                h[i, j] = -t
                h[j, i] = -t

    return h


def spectrum_diagnostics(evals: np.ndarray) -> None:
    """
    Print basic eigenvalue diagnostics.

    Prints:
    - N eigenvalues
    - min/max
    - mean (should be ~onsite)
    - count of near-zero eigenvalues (within tol=1e-8)
    """
    n = len(evals)
    ev_min = float(np.min(evals))
    ev_max = float(np.max(evals))
    ev_mean = float(np.mean(evals))
    tol = 1e-8
    n_zero = int(np.sum(np.abs(evals) < tol))

    print(f"Spectrum: N={n} eigenvalues")
    print(f"  min/max/mean = {ev_min:.6f}/{ev_max:.6f}/{ev_mean:.6f}")
    if n_zero > 0:
        print(f"  near-zero (|E|<{tol:.0e}): {n_zero}")


def spacing_diagnostics(
    evals: np.ndarray,
    out_png: str,
    drop_frac: float = 0.2,
    *,
    cutoff: float | None = None,
) -> None:
    """
    Histogram of normalized nearest-neighbour spacings with GOE Wigner-surmise overlay.
    Uses bulk of spectrum only (drop edges). Deterministic; no fit.
    """
    evals = np.asarray(evals, dtype=float)
    if not np.all(np.diff(evals) >= -1e-12):
        evals = np.sort(evals)

    n = len(evals)
    k0 = int(drop_frac * n)
    bulk = evals[k0 : n - k0]

    if len(bulk) < 10:
        print("spacing_diagnostics: bulk length < 10, skipping.")
        return

    s = np.diff(bulk)
    s = s[s > 1e-12]
    if len(s) == 0:
        print("spacing_diagnostics: no positive spacings after filtering.")
        return

    s_norm = s / np.mean(s)

    # Adjacent gap ratio r (unfolding-free)
    s2 = np.diff(bulk)
    r = np.minimum(s2[1:], s2[:-1]) / np.maximum(s2[1:], s2[:-1])
    r = r[np.isfinite(r)]
    mean_r = float(np.mean(r)) if len(r) > 0 else 0.0
    print(f"Gap ratio <r> (bulk): {mean_r:.3f}  (GOE~0.536, Poisson~0.386)")

    fig, ax = plt.subplots()
    ax.hist(s_norm, bins=30, density=True, color="steelblue", alpha=0.7, label="data")
    ax.set_xlim(0, 3.0)
    ax.set_xlabel("s / <s>")
    ax.set_ylabel("density")

    x = np.linspace(0, 3, 400)
    p_goe = (np.pi / 2) * x * np.exp(-np.pi * x**2 / 4)
    ax.plot(x, p_goe, "k-", lw=2, label="GOE Wigner surmise")

    title = f"N={n}, bulk frac={1 - 2*drop_frac:.2f}"
    if cutoff is not None:
        title = f"cutoff={cutoff:.2f} Å, " + title
    ax.set_title(title)
    ax.legend()

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved spacing histogram: {out_png}")


def _print_cutoff_scan(atoms: Atoms, cutoffs: list[float]) -> None:
    """Build graphs for multiple cutoffs and print a compact stats table."""
    print("Cutoff scan (geometric adjacency under PBC):")
    print("  cut(Å)    E   deg_min  deg_mean  deg_max  n_comp")
    for c in cutoffs:
        adj = build_graph(atoms, c)
        degrees = [len(adj[i]) for i in range(len(adj))]
        e = int(sum(degrees) // 2)
        deg_min = int(min(degrees)) if degrees else 0
        deg_mean = float(np.mean(degrees)) if degrees else 0.0
        deg_max = int(max(degrees)) if degrees else 0
        n_comp = _count_connected_components(adj)
        print(f"  {c:5.2f}  {e:4d}  {deg_min:7d}  {deg_mean:8.2f}  {deg_max:7d}  {n_comp:6d}")


def export_interactive_plotly(atoms: Atoms, cutoff: float, out_html: str) -> None:
    """
    Export an interactive Plotly 3D visualization of the full cutoff-based graph.

    This is visualization-only: edges are defined purely by the existing geometric
    neighbor_list cutoff logic under PBC (no chemistry/bond inference).
    """
    atoms_vis = atoms.copy()
    if bool(np.any(atoms_vis.get_pbc())):
        atoms_vis.wrap()  # visualization only

    pos = atoms_vis.get_positions()
    cell = atoms_vis.get_cell().array

    ii, jj, SS = neighbor_list("ijS", atoms_vis, float(cutoff))

    # Deduplicate undirected edges while keeping a consistent periodic image shift.
    # Store as (i, j, Sx, Sy, Sz) with i < j. If we swap, also negate S.
    uniq_edges: set[tuple[int, int, int, int, int]] = set()
    for i, j, S in zip(ii.tolist(), jj.tolist(), SS.tolist()):
        if i == j:
            continue
        sx, sy, sz = int(S[0]), int(S[1]), int(S[2])
        if i < j:
            key = (int(i), int(j), sx, sy, sz)
        else:
            key = (int(j), int(i), -sx, -sy, -sz)
        uniq_edges.add(key)

    # Build edge coordinate arrays with None separators (single line trace).
    edge_x: list[float | None] = []
    edge_y: list[float | None] = []
    edge_z: list[float | None] = []
    for i, j, sx, sy, sz in sorted(uniq_edges):
        r_i = pos[i]
        r_j_img = pos[j] + (sx * cell[0] + sy * cell[1] + sz * cell[2])
        edge_x.extend([float(r_i[0]), float(r_j_img[0]), None])
        edge_y.extend([float(r_i[1]), float(r_j_img[1]), None])
        edge_z.extend([float(r_i[2]), float(r_j_img[2]), None])

    # Node colors (categorical) in a single Scatter3d trace.
    symbols = atoms_vis.get_chemical_symbols()
    unique_elements = sorted(set(symbols))
    palette = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    ]
    elem_to_color = {el: palette[i % len(palette)] for i, el in enumerate(unique_elements)}
    node_colors = [elem_to_color[s] for s in symbols]
    node_text = [f"i={i}, {symbols[i]}" for i in range(len(symbols))]

    nodes = go.Scatter3d(
        x=pos[:, 0],
        y=pos[:, 1],
        z=pos[:, 2],
        mode="markers",
        marker=dict(size=5, opacity=0.9, color=node_colors),
        text=node_text,
        hoverinfo="text",
        name="sites",
    )

    edges = go.Scatter3d(
        x=edge_x,
        y=edge_y,
        z=edge_z,
        mode="lines",
        line=dict(color="rgba(0,0,0,0.25)", width=1),
        hoverinfo="skip",
        name="edges",
    )

    # Extra hops trace (red, high visibility)
    extra_data = []
    if EXTRA_HOPS:
        ex_x: list[float | None] = []
        ex_y: list[float | None] = []
        ex_z: list[float | None] = []
        for hop in EXTRA_HOPS:
            hi = hop["i"]
            hj = hop["j"]
            r_i = pos[hi]
            if "S" in hop:
                sx_h, sy_h, sz_h = hop["S"]
                r_j_img = pos[hj] + float(sx_h) * cell[0] + float(sy_h) * cell[1] + float(sz_h) * cell[2]
            else:
                r_j_img = r_i + np.asarray(hop["dr"], dtype=float)
            ex_x.extend([float(r_i[0]), float(r_j_img[0]), None])
            ex_y.extend([float(r_i[1]), float(r_j_img[1]), None])
            ex_z.extend([float(r_i[2]), float(r_j_img[2]), None])
        extra_trace = go.Scatter3d(
            x=ex_x, y=ex_y, z=ex_z,
            mode="lines",
            line=dict(color="red", width=4),
            hoverinfo="skip",
            name="extra hops",
        )
        extra_data = [extra_trace]

    n_nodes = len(atoms_vis)
    n_edges = len(uniq_edges)
    n_extra = len(EXTRA_HOPS)
    fig = go.Figure(data=[edges] + extra_data + [nodes])
    extra_tag = f", extra_hops={n_extra}" if n_extra > 0 else ""
    fig.update_layout(
        title=f"Cutoff graph (cutoff={cutoff:.2f} Å) — N={n_nodes}, E={n_edges}{extra_tag}",
        scene=dict(
            aspectmode="data",
            xaxis_title="x (Å)",
            yaxis_title="y (Å)",
            zaxis_title="z (Å)",
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        showlegend=False,
    )

    os.makedirs(os.path.dirname(out_html), exist_ok=True)
    fig.write_html(out_html, include_plotlyjs=True)
    print(f"Saved interactive HTML: {out_html}")


def render_3d(atoms: Atoms, out_path: str, *, edge_cutoff: float | None = None, edge_sample_nodes: int = 10) -> None:
    """
    Render atoms as 3D scatter. Color by element and save PNG.

    Optional geometry-only sanity check:
    - If `edge_cutoff` is provided, draw edges (lines) for a small random subset of nodes
      using the same PBC-aware neighbor list. This is not a bond inference.
    """
    atoms_vis = atoms.copy()
    if bool(np.any(atoms_vis.get_pbc())):
        atoms_vis.wrap()  # visualization only: keep points inside the unit cell
        print("Note: wrapped positions into the unit cell for visualization only.")

    pos = atoms_vis.get_positions()
    symbols = atoms_vis.get_chemical_symbols()
    unique_elements = sorted(set(symbols))

    # Simple color map by element (distinct colors)
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(unique_elements), 1)))
    elem_to_color = {el: colors[i % len(colors)] for i, el in enumerate(unique_elements)}

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    for el in unique_elements:
        mask = [s == el for s in symbols]
        xyz = pos[mask]
        ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=[elem_to_color[el]], label=el, s=25)

    ax.set_xlabel("x (Å)")
    ax.set_ylabel("y (Å)")
    ax.set_zlabel("z (Å)")
    ax.legend()
    # Equal data scaling: 1 Angstrom same length in x, y, z
    ranges = np.ptp(pos, axis=0)
    ranges = np.where(ranges == 0.0, 1.0, ranges)
    try:
        ax.set_box_aspect((float(ranges[0]), float(ranges[1]), float(ranges[2])))
    except Exception:
        # Older matplotlib: fall back to equal-ish limits below.
        pass

    # Optional: draw edges for a small subset of nodes (geometry-only sanity check)
    if edge_cutoff is not None and edge_sample_nodes > 0:
        n = len(atoms_vis)
        k = int(min(edge_sample_nodes, n))
        rng = np.random.default_rng(0)  # deterministic subset for repeatable output
        subset = set(rng.choice(n, size=k, replace=False).tolist())

        ii, jj, SS = neighbor_list("ijS", atoms_vis, float(edge_cutoff))
        cell = atoms_vis.get_cell().array  # (3,3) with cell vectors as rows
        drawn = 0
        drawn_pairs: set[tuple[int, int]] = set()
        for i, j, S in zip(ii.tolist(), jj.tolist(), SS.tolist()):
            if i == j:
                continue
            if i not in subset:
                continue
            a, b = (i, j) if i < j else (j, i)
            if a == b or (a, b) in drawn_pairs:
                continue
            drawn_pairs.add((a, b))
            # Shift neighbor to the specific periodic image provided by neighbor_list.
            r_i = pos[i]
            r_j = pos[j] + (S[0] * cell[0] + S[1] * cell[1] + S[2] * cell[2])
            ax.plot([r_i[0], r_j[0]], [r_i[1], r_j[1]], [r_i[2], r_j[2]], color="k", linewidth=0.6, alpha=0.35)
            drawn += 1
        print(f"Edge overlay: drew {drawn} line segments for {k} sampled nodes at cutoff={edge_cutoff:.2f} Å.")

    # Draw manual extra hops in red (using stored S for correct periodic image)
    if EXTRA_HOPS:
        cell_vis = atoms_vis.get_cell().array
    for hop in EXTRA_HOPS:
        hi = hop["i"]
        hj = hop["j"]
        r_i = pos[hi]
        if "S" in hop:
            sx, sy, sz = hop["S"]
            r_j_img = pos[hj] + float(sx) * cell_vis[0] + float(sy) * cell_vis[1] + float(sz) * cell_vis[2]
        else:
            r_j_img = r_i + hop["dr"]
        ax.plot([r_i[0], r_j_img[0]], [r_i[1], r_j_img[1]], [r_i[2], r_j_img[2]],
                color="red", linewidth=2.0)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def view_hamiltonian(h: np.ndarray, cutoff: float) -> None:
    """
    Deterministic Hamiltonian viewing: sparsity pattern, heatmap, and top-left block.
    """
    h = np.asarray(h, dtype=float)
    out_sparsity = os.path.join(FIG_DIR, f"H_sparsity_cut{cutoff:.2f}.png")
    out_heatmap = os.path.join(FIG_DIR, f"H_heatmap_cut{cutoff:.2f}.png")
    os.makedirs(FIG_DIR, exist_ok=True)

    # Sparsity pattern: binary mask (|H| > 0)
    mask = np.abs(h) > 0
    fig1, ax1 = plt.subplots()
    ax1.imshow(mask.astype(float), cmap="binary", interpolation="none", aspect="equal")
    ax1.set_xlabel("j")
    ax1.set_ylabel("i")
    ax1.set_title(f"H sparsity (cutoff={cutoff:.2f} Å)")
    fig1.savefig(out_sparsity, dpi=200, bbox_inches="tight")
    plt.close(fig1)
    print(f"Saved sparsity: {out_sparsity}")

    # Heatmap: centered at 0, symmetric limits
    vmax = float(np.max(np.abs(h)))
    fig2, ax2 = plt.subplots()
    im = ax2.imshow(h, cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="none", aspect="equal")
    ax2.set_xlabel("j")
    ax2.set_ylabel("i")
    ax2.set_title(f"H heatmap (cutoff={cutoff:.2f} Å)")
    plt.colorbar(im, ax=ax2)
    fig2.savefig(out_heatmap, dpi=200, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved heatmap: {out_heatmap}")

    # Top-left 12×12 block to stdout
    n, m = h.shape
    block = h[: min(12, n), : min(12, m)]
    print("H top-left 12×12 block:")
    for row in block:
        print(" ", " ".join(f"{x:7.3f}" for x in row))


def build_edge_list_pbc(
    atoms: Atoms, cutoff: float
) -> list[tuple[int, int, tuple[float, float, float], tuple[int, int, int]]]:
    """
    Deduplicated undirected edge list with full bond vectors and shift vectors.

    Returns sorted list of (i, j, dr, shift_ints) with i < j, where
    dr = pos[j] - pos[i] + S·cell is the full bond vector from site i to site j,
    and shift_ints = (s0, s1, s2) is the integer PBC image shift.
    Self-edges excluded.
    """
    pos = atoms.get_positions()
    cell = atoms.get_cell().array
    ii, jj, SS = neighbor_list("ijS", atoms, float(cutoff))

    seen: dict[tuple[int, int, int, int, int], np.ndarray] = {}
    for i, j, S in zip(ii.tolist(), jj.tolist(), SS.tolist()):
        if i == j:
            continue
        sx, sy, sz = int(S[0]), int(S[1]), int(S[2])
        R = float(sx) * cell[0] + float(sy) * cell[1] + float(sz) * cell[2]
        if i < j:
            key = (int(i), int(j), sx, sy, sz)
            dr = (pos[j] - pos[i]) + R
        else:
            key = (int(j), int(i), -sx, -sy, -sz)
            dr = (pos[i] - pos[j]) - R
        if key not in seen:
            seen[key] = dr

    result = []
    for key in sorted(seen.keys()):
        i_can, j_can = key[0], key[1]
        shift = (key[2], key[3], key[4])
        dr = seen[key]
        result.append((i_can, j_can, (float(dr[0]), float(dr[1]), float(dr[2])), shift))
    return result


def classify_distance_shells(
    edge_list: list[tuple], tol: float = SHELL_GAP_TOL
) -> tuple[list[float], list[int]]:
    """Group edges into shells by bond length.

    A new shell is started wherever the gap between consecutive sorted unique
    distances exceeds `tol`.  Each edge is then assigned to its nearest shell
    representative (argmin), which is robust even when within-shell distances
    span more than `tol`.

    Parameters
    ----------
    edge_list : list of tuples with dr_cart at index 2
    tol       : minimum inter-shell gap (Angstrom).
                Default = SHELL_GAP_TOL (0.3 Ang), large enough to cleanly
                separate NN from 2NN in carbon schwarzites.

    Returns
    -------
    shell_distances : list of representative distances in ascending order
    shell_index     : per-edge shell id (same length as edge_list)
    """
    if not edge_list:
        return [], []

    lengths = np.array([np.linalg.norm(np.asarray(edge[2], dtype=float))
                        for edge in edge_list])
    unique_r = np.sort(np.unique(lengths))

    # Build one representative per shell: new shell when gap > tol.
    shells: list[float] = [float(unique_r[0])]
    for r in unique_r[1:]:
        if r - shells[-1] > tol:
            shells.append(float(r))

    # Assign each edge to its nearest shell representative.
    shell_arr = np.array(shells)
    shell_index = [int(np.argmin(np.abs(shell_arr - length))) for length in lengths]

    return shells, shell_index


def build_shell_model_edges(
    atoms: Atoms, shells_to_include: list[int]
) -> list[tuple]:
    """Build a 5-tuple edge list covering the specified bond-length shells.

    Automatically expands the cutoff until enough shells are detected,
    capped at MAX_CUTOFF_2NN. Shell 0 gets amplitude T_HOP,
    shell 1 gets T_HOP_2NN.

    Returns
    -------
    model_edges : list of (i, j, dr_tuple, shift_tuple, amp) 5-tuples
    """
    need_shells = max(shells_to_include) + 1
    max_cutoff = CUT * 2.5
    shell_amps = {0: T_HOP, 1: T_HOP_2NN}

    while max_cutoff <= MAX_CUTOFF_2NN:
        edge_list = build_edge_list_pbc(atoms, max_cutoff)
        shell_distances, shell_idx = classify_distance_shells(edge_list, SHELL_GAP_TOL)
        if len(shell_distances) >= need_shells:
            break
        max_cutoff *= 1.3
    else:
        print(f"WARNING: could not find {need_shells} shells within "
              f"{MAX_CUTOFF_2NN:.1f} Ang cutoff. Found {len(shell_distances)}.")

    print(f"nn+2nn: max_cutoff = {max_cutoff:.3f} Ang, "
          f"shells found: {[f'{d:.4f}' for d in shell_distances]}")
    for sid in sorted(shells_to_include):
        count = int(np.sum(np.array(shell_idx) == sid))
        d_rep = shell_distances[sid] if sid < len(shell_distances) else float("nan")
        amp   = shell_amps.get(sid, T_HOP)
        print(f"  shell {sid}: d={d_rep:.4f} Ang, {count} edges, amp={amp}")

    model_edges = []
    for edge, sid in zip(edge_list, shell_idx):
        if sid in shells_to_include:
            i, j, dr, sh = edge[0], edge[1], edge[2], edge[3]
            amp = shell_amps.get(sid, T_HOP)
            model_edges.append((i, j, dr, sh, amp))
    return model_edges


def build_boundary_model_edges(
    atoms: Atoms,
    target_shell: int,
    amp_boundary: float,
) -> list[tuple]:
    """Build a 5-tuple edge list: all NN (shell 0) + target_shell boundary-only.

    "Boundary-only" means PBC-shifted edges where shift_ints != (0,0,0).
    Shell numbering follows classify_distance_shells sorted ascending:
      0=NN, 1=2NN, 2=3NN, 3=4NN, ...

    Parameters
    ----------
    target_shell : int
        Shell index to include as boundary-only hops (1 → 2NN, 3 → 4NN).
    amp_boundary : float
        Hopping amplitude for boundary-shell edges.

    Returns
    -------
    model_edges : list of (i, j, dr_tuple, shift_tuple, amp) 5-tuples
    """
    need_shells = target_shell + 1
    max_cutoff = CUT * 2.5

    while max_cutoff <= MAX_CUTOFF_2NN:
        edge_list = build_edge_list_pbc(atoms, max_cutoff)
        shell_distances, shell_idx = classify_distance_shells(edge_list, SHELL_GAP_TOL)
        if len(shell_distances) >= need_shells:
            break
        max_cutoff *= 1.3
    else:
        print(f"ERROR: could not find {need_shells} shells within "
              f"{MAX_CUTOFF_2NN:.1f} Ang cutoff.")
        print(f"  Found only {len(shell_distances)} shells: "
              f"{[f'{d:.4f}' for d in shell_distances]}")
        print(f"  Try increasing MAX_CUTOFF_2NN (currently {MAX_CUTOFF_2NN:.1f} Ang).")
        raise RuntimeError(
            f"Not enough bond-length shells for nn+{target_shell}NN-bdry model.")

    # Print first max(4, need_shells) shells so the user can see the distances
    shell_idx_arr = np.array(shell_idx)
    n_show = max(4, need_shells)
    print(f"Shell detection: max_cutoff={max_cutoff:.3f} Ang, "
          f"{len(shell_distances)} shells total")
    for sid in range(min(n_show, len(shell_distances))):
        count = int(np.sum(shell_idx_arr == sid))
        bdry  = sum(1 for k, e in enumerate(edge_list)
                    if shell_idx[k] == sid and e[3] != (0, 0, 0))
        print(f"  shell {sid}: d={shell_distances[sid]:.4f} Ang, "
              f"{count} edges ({bdry} cross-boundary)")

    model_edges = []
    for edge, sid in zip(edge_list, shell_idx):
        i, j, dr, sh = edge[0], edge[1], edge[2], edge[3]
        if sid == 0:
            model_edges.append((i, j, dr, sh, T_HOP))
        elif sid == target_shell and sh != (0, 0, 0):
            model_edges.append((i, j, dr, sh, amp_boundary))

    n_nn   = sum(1 for e in model_edges if e[3] == (0, 0, 0))
    n_bdry = sum(1 for e in model_edges if e[3] != (0, 0, 0))
    print(f"  -> {n_nn} NN intra-cell + {n_nn + n_bdry - n_nn} NN cross-boundary "
          f"= {len([e for e in model_edges if e[4] == T_HOP])} NN edges (amp={T_HOP}), "
          f"{len([e for e in model_edges if e[4] == amp_boundary])} shell-{target_shell} "
          f"boundary edges (amp={amp_boundary})")
    return model_edges


def build_decay_model_edges(
    atoms: Atoms,
    n_shells: int,
    alpha: float,
) -> list[tuple]:
    """Build a 5-tuple edge list for shells 0 .. n_shells-1 with geometric decay.

    Shell i gets amplitude  T_HOP * alpha**i  (all edges, no boundary filter).
      shell 0 (NN)  : T_HOP * 1.0
      shell 1 (2NN) : T_HOP * alpha
      shell 2 (3NN) : T_HOP * alpha**2
      shell 3 (4NN) : T_HOP * alpha**3

    Parameters
    ----------
    n_shells : int   Number of shells to include (4 for NN+2NN+3NN+4NN).
    alpha    : float Geometric decay factor per shell (e.g. 0.5).
    """
    max_cutoff = CUT * 2.5

    while max_cutoff <= MAX_CUTOFF_2NN:
        edge_list = build_edge_list_pbc(atoms, max_cutoff)
        shell_distances, shell_idx = classify_distance_shells(edge_list, SHELL_GAP_TOL)
        if len(shell_distances) >= n_shells:
            break
        max_cutoff *= 1.3
    else:
        print(f"ERROR: could not find {n_shells} shells within "
              f"{MAX_CUTOFF_2NN:.1f} Ang cutoff.")
        print(f"  Found only {len(shell_distances)} shells: "
              f"{[f'{d:.4f}' for d in shell_distances]}")
        print(f"  Try increasing MAX_CUTOFF_2NN (currently {MAX_CUTOFF_2NN:.1f} Ang).")
        raise RuntimeError(f"Not enough bond-length shells (need {n_shells}).")

    shell_idx_arr = np.array(shell_idx)
    print(f"Shell detection: max_cutoff={max_cutoff:.3f} Ang, "
          f"{len(shell_distances)} shells total")
    for sid in range(min(n_shells, len(shell_distances))):
        count = int(np.sum(shell_idx_arr == sid))
        amp   = T_HOP * (alpha ** sid)
        print(f"  shell {sid}: d={shell_distances[sid]:.4f} Ang, "
              f"{count} edges, amp={amp:.4f}")

    model_edges = []
    for edge, sid in zip(edge_list, shell_idx):
        if sid < n_shells:
            i, j, dr, sh = edge[0], edge[1], edge[2], edge[3]
            amp = T_HOP * (alpha ** sid)
            model_edges.append((i, j, dr, sh, amp))

    print(f"  -> {len(model_edges)} total edges ({n_shells} shells, alpha={alpha})")
    return model_edges


def min_image_dr(
    atoms: Atoms, i: int, j: int
) -> tuple[np.ndarray, tuple[int, int, int], float]:
    """Return the minimum-image displacement from site i to site j.

    Searches all 27 shift combinations (sx,sy,sz) in MANUAL_SHIFT_SEARCH,
    choosing the one that minimises ||dr||.

    Returns
    -------
    dr_cart  : np.ndarray shape (3,)  Cartesian displacement (Angstrom)
    shift    : (sx, sy, sz) integer shift that gives the minimum image
    dist     : float  ||dr_cart||
    """
    pos  = atoms.get_positions()
    cell = atoms.get_cell().array
    ri, rj = pos[i], pos[j]

    best_dr   = None
    best_shift = (0, 0, 0)
    best_dist  = float("inf")

    for sx in MANUAL_SHIFT_SEARCH:
        for sy in MANUAL_SHIFT_SEARCH:
            for sz in MANUAL_SHIFT_SEARCH:
                dr   = (rj + sx*cell[0] + sy*cell[1] + sz*cell[2]) - ri
                dist = float(np.linalg.norm(dr))
                if dist < best_dist:
                    best_dist  = dist
                    best_dr    = dr
                    best_shift = (sx, sy, sz)

    return best_dr, best_shift, best_dist


def find_all_pairs_with_distance(
    atoms: Atoms, target_dist: float, tol: float
) -> list[tuple]:
    """Find all ordered pairs (p<q) whose minimum-image distance equals target_dist.

    Returns
    -------
    list of (p, q, dr_cart_tuple, shift_ints, dist)
    """
    n      = len(atoms)
    result = []
    for p in range(n):
        for q in range(p + 1, n):
            dr, shift, dist = min_image_dr(atoms, p, q)
            if abs(dist - target_dist) < tol:
                result.append((
                    p, q,
                    (float(dr[0]), float(dr[1]), float(dr[2])),
                    shift,
                    dist,
                ))
    return result


def build_manual_pair_model_edges(
    atoms: Atoms, i0: int, j0: int
) -> list[tuple]:
    """Build NN baseline + all distance-equivalent copies of the (i0,j0) hop.

    Steps:
    1. Build full NN edge list (shell 0, amplitude T_HOP).
    2. Compute minimum-image displacement for the seed pair (i0, j0).
    3. Find all pairs (p,q) with the same minimum-image distance (within MANUAL_MATCH_TOL).
    4. Add each equivalent pair as an extra edge with amplitude MANUAL_T_HOP,
       unless the same (p,q,shift) already exists in the NN set (no double-counting).

    Returns
    -------
    model_edges : list of (i, j, dr_tuple, shift_tuple, amp) 5-tuples
    """
    # 1. NN baseline
    edges_4  = build_edge_list_pbc(atoms, CUT)
    nn_edges = [(i, j, dr, sh, T_HOP) for (i, j, dr, sh) in edges_4]
    nn_keys  = {(e[0], e[1], e[3]) for e in nn_edges}   # (i, j, shift) lookup

    # 2. Seed hop
    dr0, shift0, d0 = min_image_dr(atoms, i0, j0)
    print(f"Chosen hop: ({i0} -> {j0}), shift={shift0}, |dr|={d0:.6f} Ang")

    # 3. Equivalent pairs
    eq_pairs = find_all_pairs_with_distance(atoms, d0, MANUAL_MATCH_TOL)
    print(f"Found {len(eq_pairs)} equivalent pairs (|dr| within {MANUAL_MATCH_TOL} Ang of {d0:.6f}):")
    for k, (p, q, dr, sh, dist) in enumerate(eq_pairs[:10]):
        print(f"  {k:3d}: ({p:3d},{q:3d}) shift={sh}  dist={dist:.6f} Ang")
    if len(eq_pairs) > 10:
        print(f"  ... ({len(eq_pairs) - 10} more pairs not shown)")

    # 4. Add non-duplicate manual edges
    manual_edges = []
    for (p, q, dr, sh, dist) in eq_pairs:
        if (p, q, sh) not in nn_keys:
            manual_edges.append((p, q, dr, sh, MANUAL_T_HOP))

    model_edges = nn_edges + manual_edges
    print(f"Model edges: {len(nn_edges)} NN (amp={T_HOP}) "
          f"+ {len(manual_edges)} manual-pair (amp={MANUAL_T_HOP}) "
          f"= {len(model_edges)} total  "
          f"({len(eq_pairs) - len(manual_edges)} pairs already in NN set, skipped)")
    return model_edges


def build_tb_hamiltonian_k(
    n: int,
    edges_pbc: list[tuple],
    k_cart: np.ndarray,
    t: float,
    onsite: float,
    include_extra_hops: bool = True,
    reciprocal_b: np.ndarray | None = None,
) -> np.ndarray:
    """
    Build the Bloch Hamiltonian H(k) for a tight-binding model.

    Uses the full bond vector dr = pos[j]-pos[i]+R for phase factors (Bloch
    convention), required for correct Berry-phase / topology calculations.

    Parameters
    ----------
    n : int
        Number of sites (matrix dimension).
    edges_pbc : list of tuples
        Deduplicated undirected edge list (i < j). Each entry is either:
          (i, j, dr_cart, shift_ints)         — 4-tuple, amplitude = t
          (i, j, dr_cart, shift_ints, amp)    — 5-tuple, per-edge amplitude
        The 5-tuple form is produced by build_shell_model_edges for nn+2nn.
    k_cart : np.ndarray, shape (3,)
        k-point in Cartesian (inverse-Angstrom) coordinates.
    t : float
        Default hopping amplitude (used for 4-tuple edges; ignored for 5-tuple).
    onsite : float
        Uniform on-site energy.
    include_extra_hops : bool
        If True (default), include EXTRA_HOPS manual hoppings.
        Set False for negative-control (NN-only) runs.
    reciprocal_b : np.ndarray or None, shape (3, 3)
        Reciprocal lattice vectors as rows (same convention as reciprocal_lattice).
        When USE_TRUE_FIRST_BZ is True, k_cart is folded into the true first BZ
        (Wigner–Seitz cell about Γ) before applying Bloch phases.

    Returns
    -------
    Hk : np.ndarray, shape (n, n), dtype complex
        Hermitian Bloch Hamiltonian at the given k-point.
    """
    k_cart = _effective_k_cart_for_tb(k_cart, reciprocal_b)

    hk = np.zeros((n, n), dtype=complex)

    for i in range(n):
        hk[i, i] = onsite

    for edge in edges_pbc:
        i, j, dr = edge[0], edge[1], edge[2]
        amp = edge[4] if len(edge) >= 5 else t
        dr_vec = np.array(dr, dtype=float)
        phase = np.exp(1j * np.dot(k_cart, dr_vec))
        hk[i, j] += -amp * phase
        hk[j, i] += -amp * np.conj(phase)

    # --- Manual extra hoppings (skipped for negative-control runs) ---
    if include_extra_hops:
        for hop in EXTRA_HOPS:
            hi = hop["i"]
            hj = hop["j"]
            dr_vec = np.asarray(hop["dr"], dtype=float)
            amplitude = hop["amplitude"]
            phase = np.exp(1j * np.dot(k_cart, dr_vec))
            hk[hi, hj] += -amplitude * phase
            hk[hj, hi] += -amplitude * np.conj(phase)

    assert np.allclose(hk, hk.conj().T), "H(k) must be Hermitian"
    return hk


def reciprocal_lattice(cell: np.ndarray) -> np.ndarray:
    """
    Compute reciprocal lattice vectors from a real-space cell matrix.

    Parameters
    ----------
    cell : np.ndarray, shape (3, 3)
        Real-space lattice vectors as rows: cell[i] = a_i.

    Returns
    -------
    b : np.ndarray, shape (3, 3)
        Reciprocal lattice vectors as rows, satisfying a_i . b_j = 2*pi*delta_ij.
    """
    cell = np.asarray(cell, dtype=float)
    return 2.0 * np.pi * np.linalg.inv(cell).T


def _generate_reciprocal_G_grid(b: np.ndarray, n_shells: int) -> np.ndarray:
    """G = n1*b1 + n2*b2 + n3*b3 for integers ni in [-n_shells, n_shells]."""
    ns = range(-int(n_shells), int(n_shells) + 1)
    b = np.asarray(b, dtype=float)
    pts = [
        float(n1) * b[0] + float(n2) * b[1] + float(n3) * b[2]
        for n1 in ns for n2 in ns for n3 in ns
    ]
    return np.array(pts, dtype=float)


_G_SORTED_CACHE: dict[tuple, np.ndarray] = {}


def _sorted_nonzero_G_vectors(b: np.ndarray, n_shells: int) -> np.ndarray:
    """Unique nonzero reciprocal vectors, shortest first (for deterministic reduction)."""
    b = np.asarray(b, dtype=float)
    key = (tuple(np.round(b.ravel(), 14)), int(n_shells))
    if key not in _G_SORTED_CACHE:
        G = _generate_reciprocal_G_grid(b, n_shells)
        nz = G[np.linalg.norm(G, axis=1) > 1e-10]
        order = np.argsort(np.sum(nz * nz, axis=1))
        _G_SORTED_CACHE[key] = nz[order]
    return _G_SORTED_CACHE[key]


def reduce_k_cartesian_to_first_bz(
    k_cart: np.ndarray,
    b: np.ndarray,
    *,
    tol: float = 1e-10,
    max_iter: int = 10000,
) -> np.ndarray:
    """Fold Cartesian k into the first BZ (Wigner–Seitz cell about Γ).

    Uses a centered primitive fractional fold followed by iterative subtraction
    of reciprocal lattice vectors G that violate k·G ≤ |G|²/2.
    """
    if not USE_TRUE_FIRST_BZ:
        return np.asarray(k_cart, dtype=float).reshape(3).copy()

    k = np.asarray(k_cart, dtype=float).reshape(3).copy()
    b = np.asarray(b, dtype=float)
    binv = np.linalg.inv(b)
    f = k @ binv
    f = np.mod(f + 0.5, 1.0) - 0.5
    k = f @ b

    G_list = _sorted_nonzero_G_vectors(b, TRUE_BZ_G_SHELLS_REDUCE)
    for _ in range(max_iter):
        moved = False
        for G in G_list:
            g2 = float(np.dot(G, G))
            if float(np.dot(k, G)) > 0.5 * g2 + tol:
                k -= G
                moved = True
                break
        if not moved:
            return k
    return k


def _effective_k_cart_for_tb(k_cart: np.ndarray, reciprocal_b: np.ndarray | None) -> np.ndarray:
    """Canonical k-point used for Bloch phases (optionally folded into true first BZ)."""
    k_cart = np.asarray(k_cart, dtype=float).reshape(3)
    if not USE_TRUE_FIRST_BZ or reciprocal_b is None:
        return k_cart.copy()
    return reduce_k_cartesian_to_first_bz(k_cart, reciprocal_b)


def compute_true_first_bz(
    b: np.ndarray,
    n_shells: int = TRUE_BZ_G_SHELLS_HALFSPACE,
) -> tuple[np.ndarray, ConvexHull, np.ndarray]:
    """First Brillouin zone as Wigner–Seitz cell via half-space intersection."""
    b = np.asarray(b, dtype=float)
    G_all = _generate_reciprocal_G_grid(b, n_shells)
    halfspaces: list[np.ndarray] = []
    for G in G_all:
        gn = float(np.linalg.norm(G))
        if gn < 1e-10:
            continue
        halfspaces.append(np.append(G, -0.5 * gn * gn))
    hs_arr = np.array(halfspaces)
    feasible = np.zeros(3)
    hs_int = HalfspaceIntersection(hs_arr, feasible)
    all_verts = hs_int.intersections
    hull = ConvexHull(all_verts)
    vertices = all_verts[hull.vertices]
    return vertices, hull, all_verts


def _undirected_edges_from_hull(hull: ConvexHull) -> list[tuple[int, int]]:
    """Unique undirected edges (vertex indices) on a 3D scipy ConvexHull."""
    edge_set: set[tuple[int, int]] = set()
    for tri in hull.simplices:
        for i in range(3):
            a = int(tri[i])
            vb = int(tri[(i + 1) % 3])
            if a > vb:
                a, vb = vb, a
            edge_set.add((a, vb))
    return sorted(edge_set)


def reference_weyl_points_kcart(b: np.ndarray) -> list[dict]:
    """Cartesian k and fractional coords for ``REFERENCE_WEYL_POINTS``."""
    b = np.asarray(b, dtype=float)
    out: list[dict] = []
    for label, f in REFERENCE_WEYL_POINTS:
        f_arr = np.asarray(f, dtype=float)
        k_cart = f_arr @ b
        out.append({
            "label": label,
            "k_frac": f_arr,
            "k_cart": k_cart,
        })
    return out


def render_bz_wireframe_basis_style(
    cell: np.ndarray,
    out_png: str,
    out_html: str | None = None,
) -> None:
    """
    Paper-style reciprocal-space figure: **wireframe** first BZ (black edges), a **bold**
    outline of the primitive reciprocal parallelepiped, reciprocal basis arrows
    **a**, **b**, **c** (rows of ``reciprocal_lattice(cell)``), and markers **X**, **Y**, **Z**
    at ½ **a**, ½ **b**, ½ **c**.

    When ``DO_REFERENCE_WEYL_MARKERS`` is True, overlays ``REFERENCE_WEYL_POINTS`` as
    distinct coloured markers (fractional coords → Cartesian via ``f @ b``).

    No k-paths — use this to read off faces / compare to standard BZ drawings, then align
    relabellings in code. Runs alongside ``plot_kpath_reciprocal`` from ``band_structure``.

    PNG: Matplotlib static. HTML: Plotly (rotate/zoom); same geometric content only.
    """
    cell = np.asarray(cell, dtype=float)
    b = reciprocal_lattice(cell)

    print("Reciprocal primitive rows a, b, c (Å⁻¹); real-space rows satisfy aᵢ · vⱼ = 2π δᵢⱼ "
          "with reciprocal primitive rows v ∈ {a, b, c}:")
    for i in range(3):
        nm = RECIP_VEC_LABEL_PLAIN[i]
        print(f"  {nm} = [{b[i, 0]:+.6f}, {b[i, 1]:+.6f}, {b[i, 2]:+.6f}]  "
              f"|{nm}| = {np.linalg.norm(b[i]):.6f}")

    origin = np.zeros(3)
    corners = [
        origin,
        b[0],
        b[1],
        b[2],
        b[0] + b[1],
        b[0] + b[2],
        b[1] + b[2],
        b[0] + b[1] + b[2],
    ]
    pp_edges = [
        (0, 1), (0, 2), (0, 3),
        (1, 4), (1, 5),
        (2, 4), (2, 6),
        (3, 5), (3, 6),
        (4, 7), (5, 7), (6, 7),
    ]

    bz_vertices = None
    bz_hull = None
    bz_all = None
    if USE_TRUE_FIRST_BZ:
        try:
            bz_vertices, bz_hull, bz_all = compute_true_first_bz(b)
            print(f"Wireframe plot: first BZ {len(bz_vertices)} vertices, "
                  f"{len(bz_hull.simplices)} faces, {len(_undirected_edges_from_hull(bz_hull))} edges")
        except Exception as exc:
            print(f"WARNING: could not mesh true first BZ ({exc}); drawing primitive cell + a,b,c only.")

    ref_weyl: list[dict] = []
    if DO_REFERENCE_WEYL_MARKERS:
        ref_weyl = reference_weyl_points_kcart(b)
        print(f"Reference Weyl markers ({len(ref_weyl)} points, fractional → Cartesian):")
        for wp in ref_weyl:
            kf = wp["k_frac"]
            kc = wp["k_cart"]
            print(f"  {wp['label']}: f=({kf[0]:+.6f},{kf[1]:+.6f},{kf[2]:+.6f})  "
                  f"k=({kc[0]:+.6f},{kc[1]:+.6f},{kc[2]:+.6f})")

    stacks = [np.array(corners), b, np.zeros((1, 3)), np.array([0.5 * b[i] for i in range(3)])]
    if ref_weyl:
        stacks.append(np.array([wp["k_cart"] for wp in ref_weyl], dtype=float))
    if bz_all is not None:
        stacks.append(np.asarray(bz_all, dtype=float))
    all_coords = np.vstack(stacks)
    mins = np.min(all_coords, axis=0)
    maxs = np.max(all_coords, axis=0)
    span = np.where(maxs - mins < 1e-12, 1.0, maxs - mins)
    margin = 0.08 * span

    # --- Matplotlib (wireframe + bold primitive cell) ---
    fig = plt.figure(figsize=(9.5, 9))
    ax = fig.add_subplot(111, projection="3d")

    if bz_all is not None and bz_hull is not None:
        for ia, ib in _undirected_edges_from_hull(bz_hull):
            p0 = bz_all[ia]
            p1 = bz_all[ib]
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]],
                    color="#0d0d0d", linewidth=1.35, alpha=0.95, zorder=4)

    for i0, i1 in pp_edges:
        c0, c1 = corners[i0], corners[i1]
        ax.plot([c0[0], c1[0]], [c0[1], c1[1]], [c0[2], c1[2]],
                color="#1a1a1a", linewidth=2.8, alpha=1.0, solid_capstyle="round", zorder=5)

    basis_arrow = "#156936"
    basis_lw = 2.35
    for i in range(3):
        ax.quiver(
            0, 0, 0, b[i, 0], b[i, 1], b[i, 2],
            arrow_length_ratio=0.09,
            color=basis_arrow,
            linewidth=basis_lw,
        )
        tip = b[i] * 1.07
        ax.text(
            float(tip[0]), float(tip[1]), float(tip[2]),
            RECIP_VEC_LABEL_MPL[i],
            fontsize=13,
            color="#111111",
            fontweight="bold",
        )

    ax.scatter([0], [0], [0], color="#1565c0", s=55, zorder=10, depthshade=False)
    ax.text(0, 0, 0, r"  $\Gamma$", fontsize=11, ha="left", va="bottom", color="#111111")

    hs_lbl = [r"$X$", r"$Y$", r"$Z$"]
    hs_col = ["#6a1b9a", "#c62828", "#1565c0"]
    for i in range(3):
        hp = 0.5 * b[i]
        ax.scatter([hp[0]], [hp[1]], [hp[2]], color=hs_col[i], s=40, zorder=9, depthshade=False)
        ax.text(
            float(hp[0]), float(hp[1]), float(hp[2]),
            f"  {hs_lbl[i]}",
            fontsize=11,
            ha="left",
            va="bottom",
            color=hs_col[i],
            fontweight="bold",
        )

    if ref_weyl:
        wc = REFERENCE_WEYL_COLOR
        for wp in ref_weyl:
            kc = wp["k_cart"]
            ax.scatter(
                [kc[0]], [kc[1]], [kc[2]],
                color=wc, s=90, marker="D", zorder=11, depthshade=False,
                edgecolors="#4a0028", linewidths=0.6,
            )
            ax.text(
                float(kc[0]), float(kc[1]), float(kc[2]),
                f"  {wp['label']}",
                fontsize=10,
                ha="left",
                va="bottom",
                color=wc,
                fontweight="bold",
            )

    ax.set_xlim(float(mins[0] - margin[0]), float(maxs[0] + margin[0]))
    ax.set_ylim(float(mins[1] - margin[1]), float(maxs[1] + margin[1]))
    ax.set_zlim(float(mins[2] - margin[2]), float(maxs[2] + margin[2]))

    ax.set_xlabel("kx (Å⁻¹)")
    ax.set_ylabel("ky (Å⁻¹)")
    ax.set_zlabel("kz (Å⁻¹)")
    ax.set_title(
        r"First BZ (wireframe) + $\mathbf{a},\mathbf{b},\mathbf{c}$ + X,Y,Z + ref. Weyl"
        if bz_all is not None else
        r"Primitive cell + $\mathbf{a},\mathbf{b},\mathbf{c}$ + X,Y,Z + ref. Weyl",
    )

    rng = (maxs + margin) - (mins - margin)
    try:
        ax.set_box_aspect((float(rng[0]), float(rng[1]), float(rng[2])))
    except Exception:
        pass

    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved wireframe BZ + basis PNG: {out_png}")

    # --- Plotly: same geometry, interactive, no paths ---
    if out_html is not None:
        traces: list = []

        if bz_all is not None and bz_hull is not None:
            wx: list[float | None] = []
            wy: list[float | None] = []
            wz: list[float | None] = []
            for ia, ib in _undirected_edges_from_hull(bz_hull):
                p0 = bz_all[ia]
                p1 = bz_all[ib]
                wx.extend([float(p0[0]), float(p1[0]), None])
                wy.extend([float(p0[1]), float(p1[1]), None])
                wz.extend([float(p0[2]), float(p1[2]), None])
            traces.append(go.Scatter3d(
                x=wx, y=wy, z=wz,
                mode="lines",
                line=dict(color="#111111", width=5),
                hoverinfo="skip",
                name="first BZ (wireframe)",
            ))

        px: list[float | None] = []
        py: list[float | None] = []
        pz: list[float | None] = []
        for i0, i1 in pp_edges:
            c0, c1 = corners[i0], corners[i1]
            px.extend([float(c0[0]), float(c1[0]), None])
            py.extend([float(c0[1]), float(c1[1]), None])
            pz.extend([float(c0[2]), float(c1[2]), None])
        traces.append(go.Scatter3d(
            x=px, y=py, z=pz,
            mode="lines",
            line=dict(color="#222222", width=9),
            hoverinfo="skip",
            name="primitive reciprocal cell",
        ))

        for i in range(3):
            shaft = b[i]
            traces.append(go.Scatter3d(
                x=[0.0, float(shaft[0])],
                y=[0.0, float(shaft[1])],
                z=[0.0, float(shaft[2])],
                mode="lines+text",
                line=dict(color=basis_arrow, width=8),
                text=["", RECIP_VEC_LABEL_PLAIN[i]],
                textposition="top center",
                textfont=dict(size=14, color="#111111", family="Arial Black"),
                name=RECIP_VEC_LABEL_PLAIN[i],
            ))
            traces.append(go.Cone(
                x=[float(shaft[0])],
                y=[float(shaft[1])],
                z=[float(shaft[2])],
                u=[float(shaft[0] * 0.12)],
                v=[float(shaft[1] * 0.12)],
                w=[float(shaft[2] * 0.12)],
                sizemode="absolute",
                sizeref=float(np.linalg.norm(shaft) * 0.05),
                colorscale=[[0, basis_arrow], [1, basis_arrow]],
                showscale=False,
                hoverinfo="skip",
                showlegend=False,
            ))

        traces.append(go.Scatter3d(
            x=[float(0.5 * b[i][0]) for i in range(3)],
            y=[float(0.5 * b[i][1]) for i in range(3)],
            z=[float(0.5 * b[i][2]) for i in range(3)],
            mode="markers+text",
            marker=dict(size=8, color=["#6a1b9a", "#c62828", "#1565c0"]),
            text=["X", "Y", "Z"],
            textposition="top center",
            textfont=dict(size=13, color="#111111", family="Arial Black"),
            name="X,Y,Z (½a,½b,½c)",
            hoverinfo="text",
            hovertext=[
                f"{lbl} = ½ {RECIP_VEC_LABEL_PLAIN[i]}"
                for i, lbl in enumerate(["X", "Y", "Z"])
            ],
        ))

        traces.append(go.Scatter3d(
            x=[0.0], y=[0.0], z=[0.0],
            mode="markers+text",
            marker=dict(size=7, color="#1565c0"),
            text=[r"Γ"],
            textposition="bottom center",
            textfont=dict(size=14, color="#111111"),
            name="Γ",
            hoverinfo="skip",
        ))

        if ref_weyl:
            traces.append(go.Scatter3d(
                x=[float(wp["k_cart"][0]) for wp in ref_weyl],
                y=[float(wp["k_cart"][1]) for wp in ref_weyl],
                z=[float(wp["k_cart"][2]) for wp in ref_weyl],
                mode="markers+text",
                marker=dict(
                    size=10,
                    color=REFERENCE_WEYL_COLOR,
                    symbol="diamond",
                    line=dict(width=1, color="#4a0028"),
                ),
                text=[wp["label"] for wp in ref_weyl],
                textposition="top center",
                textfont=dict(size=12, color=REFERENCE_WEYL_COLOR, family="Arial Black"),
                name="reference Weyl (Bo)",
                hoverinfo="text",
                hovertext=[
                    (f"{wp['label']}: f=({wp['k_frac'][0]:+.5f}, "
                     f"{wp['k_frac'][1]:+.5f}, {wp['k_frac'][2]:+.5f})<br>"
                     f"k=({wp['k_cart'][0]:+.5f}, {wp['k_cart'][1]:+.5f}, "
                     f"{wp['k_cart'][2]:+.5f})")
                    for wp in ref_weyl
                ],
            ))

        xr = [float(mins[0] - margin[0]), float(maxs[0] + margin[0])]
        yr = [float(mins[1] - margin[1]), float(maxs[1] + margin[1])]
        zr = [float(mins[2] - margin[2]), float(maxs[2] + margin[2])]

        fig_pl = go.Figure(data=traces)
        wire_title = (
            "First BZ (wireframe) + a,b,c + X,Y,Z + reference Weyl points"
            if bz_all is not None else
            "Primitive cell + a,b,c + X,Y,Z + reference Weyl points"
        )
        fig_pl.update_layout(
            title=wire_title,
            scene=dict(
                aspectmode="data",
                xaxis_title="kx (Å⁻¹)",
                yaxis_title="ky (Å⁻¹)",
                zaxis_title="kz (Å⁻¹)",
                xaxis=dict(range=xr, backgroundcolor="rgba(250,250,250,0.4)"),
                yaxis=dict(range=yr, backgroundcolor="rgba(250,250,250,0.4)"),
                zaxis=dict(range=zr, backgroundcolor="rgba(250,250,250,0.4)"),
            ),
            margin=dict(l=0, r=0, b=0, t=52),
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        )

        os.makedirs(os.path.dirname(out_html) or ".", exist_ok=True)
        fig_pl.write_html(out_html, include_plotlyjs=True)
        print(f"Saved wireframe BZ + basis HTML: {out_html}")


def make_k_path(
    b: np.ndarray,
    n_per_segment: int = 60,
) -> tuple[np.ndarray, list[int], list[str]]:
    """
    Paper-style band-structure k-path in Cartesian coordinates.

    Path (fractional reciprocal coords): **Y → Γ → Z → W₁ → W₂ → W₃ → W₄**
    (see ``PAPER_KPATH_NODES``; k_cart = f @ reciprocal_lattice rows).

    Parameters
    ----------
    b : np.ndarray, shape (3, 3)
        Reciprocal lattice vectors as rows.
    n_per_segment : int
        Number of k-points per path segment.

    Returns
    -------
    kpts : np.ndarray, shape (Nk, 3)
        k-points in Cartesian (inverse-Angstrom) coordinates.
    tick_idx : list[int]
        Indices of segment endpoints in kpts (for plot tick marks).
    tick_labels : list[str]
        Labels for each endpoint.
    """
    return _k_path_from_fractional_nodes(b, PAPER_KPATH_NODES, n_per_segment)


def _k_path_from_fractional_nodes(
    b: np.ndarray,
    nodes_frac: list[tuple[str, tuple[float, float, float]]],
    n_per_segment: int,
) -> tuple[np.ndarray, list[int], list[str]]:
    """Linear interpolation between labelled fractional reciprocal nodes."""
    b = np.asarray(b, dtype=float)
    labels = [lbl for lbl, _ in nodes_frac]
    nodes_cart = [np.asarray(f, dtype=float) @ b for _, f in nodes_frac]

    kpts_list: list[np.ndarray] = []
    tick_idx: list[int] = [0]

    for seg in range(len(nodes_cart) - 1):
        k_start = nodes_cart[seg]
        k_end = nodes_cart[seg + 1]
        for i in range(n_per_segment):
            frac = i / n_per_segment
            kpts_list.append(k_start + frac * (k_end - k_start))
    kpts_list.append(nodes_cart[-1])

    for seg in range(1, len(nodes_cart)):
        tick_idx.append(seg * n_per_segment)

    return np.array(kpts_list), tick_idx, labels


def make_k_path_legacy_gamma_xyz(
    b: np.ndarray,
    n_per_segment: int = 60,
) -> tuple[np.ndarray, list[int], list[str]]:
    """Legacy closed path Γ → X → Y → Z → Γ (X,Y,Z at ½ **a**, ½ **b**, ½ **c**)."""
    b = np.asarray(b, dtype=float)
    nodes_frac = [
        ("Γ", (0.0, 0.0, 0.0)),
        ("X", (0.5, 0.0, 0.0)),
        ("Y", (0.0, 0.5, 0.0)),
        ("Z", (0.0, 0.0, 0.5)),
        ("Γ", (0.0, 0.0, 0.0)),
    ]
    return _k_path_from_fractional_nodes(b, nodes_frac, n_per_segment)


def make_k_path_primitive_b1b2_loop(
    b: np.ndarray,
    n_per_segment: int = 45,
) -> tuple[np.ndarray, list[int], list[str]]:
    """
    Closed loop in reciprocal Cartesian coords:
        Γ → **a** → **a**+**b** → **b** → Γ
    (Edges of the primitive reciprocal parallelogram spanned by rows **a** and **b**.)

    Useful as a **reference path** overlaid on the true first BZ alongside the
    paper path (``make_k_path``).
    """
    b = np.asarray(b, dtype=float)
    gamma = np.zeros(3)
    nodes = [
        gamma,
        b[0],
        b[0] + b[1],
        b[1],
        gamma,
    ]
    labels = ["Γ", "a", "a+b", "b", "Γ"]

    kpts_list: list[np.ndarray] = []
    tick_idx: list[int] = [0]

    for seg in range(len(nodes) - 1):
        k_start = nodes[seg]
        k_end = nodes[seg + 1]
        for i in range(n_per_segment):
            frac = i / n_per_segment
            kpts_list.append(k_start + frac * (k_end - k_start))
    kpts_list.append(nodes[-1])

    for seg in range(1, len(nodes)):
        tick_idx.append(seg * n_per_segment)

    return np.array(kpts_list), tick_idx, labels


# ---------------------------------------------------------------------------
# Milestone: k-path visualization in reciprocal space
# ---------------------------------------------------------------------------

def render_first_bz_reciprocal_basis(
    cell: np.ndarray,
    out_png: str,
    out_html: str | None = None,
) -> None:
    """Plot the true first BZ with reciprocal basis vectors only.

    Draws the Wigner–Seitz cell (when ``USE_TRUE_FIRST_BZ``), a faint primitive
    reciprocal parallelepiped, and arrows **a**, **b**, **c** (rows of
    ``reciprocal_lattice(cell)``). Does **not** draw band-structure k-paths here.
    For **BZ + path in one figure**, use ``plot_first_bz_with_kpaths`` or run
    ``band_structure`` (which calls ``plot_kpath_reciprocal`` → combined plot).
    """
    cell = np.asarray(cell, dtype=float)
    b = reciprocal_lattice(cell)

    print("Reciprocal primitive rows a, b, c (Å⁻¹); real-space rows satisfy aᵢ · vⱼ = 2π δᵢⱼ:")
    for i in range(3):
        nm = RECIP_VEC_LABEL_PLAIN[i]
        print(f"  {nm} = [{b[i, 0]:+.6f}, {b[i, 1]:+.6f}, {b[i, 2]:+.6f}]  "
              f"|{nm}| = {np.linalg.norm(b[i]):.6f}")

    origin = np.zeros(3)
    corners = [
        origin,
        b[0],
        b[1],
        b[2],
        b[0] + b[1],
        b[0] + b[2],
        b[1] + b[2],
        b[0] + b[1] + b[2],
    ]
    pp_edges = [
        (0, 1), (0, 2), (0, 3),
        (1, 4), (1, 5),
        (2, 4), (2, 6),
        (3, 5), (3, 6),
        (4, 7), (5, 7), (6, 7),
    ]

    bz_vertices = None
    bz_hull = None
    bz_all = None
    if USE_TRUE_FIRST_BZ:
        try:
            bz_vertices, bz_hull, bz_all = compute_true_first_bz(b)
            print(f"True first BZ: {len(bz_vertices)} vertices, {len(bz_hull.simplices)} faces")
        except Exception as exc:
            print(f"WARNING: could not mesh true first BZ ({exc}); parallelepiped only.")

    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111, projection="3d")

    if bz_all is not None and bz_hull is not None:
        face_verts = [bz_all[s] for s in bz_hull.simplices]
        ax.add_collection3d(Poly3DCollection(
            face_verts,
            alpha=0.14,
            facecolor="#4cc9f0",
            edgecolor="#023e8a",
            linewidths=0.35,
        ))

    for i0, i1 in pp_edges:
        c0, c1 = corners[i0], corners[i1]
        ax.plot([c0[0], c1[0]], [c0[1], c1[1]], [c0[2], c1[2]],
                color="gray", linewidth=0.5, alpha=0.35, linestyle="--")

    arrow_colors = ["#d62728", "#2ca02c", "#1f77b4"]
    for i in range(3):
        ax.quiver(0, 0, 0, b[i, 0], b[i, 1], b[i, 2],
                  arrow_length_ratio=0.08, color=arrow_colors[i], linewidth=2.0)
        ax.text(b[i, 0] * 1.08, b[i, 1] * 1.08, b[i, 2] * 1.08,
                RECIP_VEC_LABEL_MPL[i], color=arrow_colors[i], fontsize=11, fontweight="bold")

    ax.scatter([0], [0], [0], color="black", s=45, zorder=7)
    ax.text(0, 0, 0, "  Γ", fontsize=10, ha="left", va="bottom")

    ax.set_xlabel("kx (Å⁻¹)")
    ax.set_ylabel("ky (Å⁻¹)")
    ax.set_zlabel("kz (Å⁻¹)")
    ax.set_title(r"First BZ + reciprocal basis $\mathbf{a},\mathbf{b},\mathbf{c}$ (no band k-path)")

    stacks = [np.array(corners), b]
    if bz_vertices is not None:
        stacks.append(bz_vertices)
    all_coords = np.vstack(stacks)
    ranges = np.ptp(all_coords, axis=0)
    ranges = np.where(ranges == 0.0, 1.0, ranges)
    try:
        ax.set_box_aspect((float(ranges[0]), float(ranges[1]), float(ranges[2])))
    except Exception:
        pass

    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved first-BZ reciprocal-basis PNG: {out_png}")

    if out_html is None:
        print("(Band-structure k-paths are plotted separately; arrows are not the path.)")
        return

    traces: list = []
    if bz_all is not None and bz_hull is not None:
        traces.append(go.Mesh3d(
            x=bz_all[:, 0],
            y=bz_all[:, 1],
            z=bz_all[:, 2],
            i=bz_hull.simplices[:, 0],
            j=bz_hull.simplices[:, 1],
            k=bz_hull.simplices[:, 2],
            opacity=0.12,
            color="#4cc9f0",
            name="first BZ",
        ))

    pp_x: list[float | None] = []
    pp_y: list[float | None] = []
    pp_z: list[float | None] = []
    for i0, i1 in pp_edges:
        c0, c1 = corners[i0], corners[i1]
        pp_x.extend([float(c0[0]), float(c1[0]), None])
        pp_y.extend([float(c0[1]), float(c1[1]), None])
        pp_z.extend([float(c0[2]), float(c1[2]), None])
    traces.append(go.Scatter3d(
        x=pp_x, y=pp_y, z=pp_z,
        mode="lines",
        line=dict(color="rgba(120,120,120,0.45)", width=2),
        hoverinfo="skip",
        name="primitive reciprocal cell",
    ))

    arrow_hex = ["#d62728", "#2ca02c", "#1f77b4"]
    for i in range(3):
        traces.append(go.Scatter3d(
            x=[0, float(b[i, 0])], y=[0, float(b[i, 1])], z=[0, float(b[i, 2])],
            mode="lines+text",
            line=dict(color=arrow_hex[i], width=5),
            text=["", RECIP_VEC_LABEL_PLAIN[i]],
            textposition="top center",
            textfont=dict(size=12, color=arrow_hex[i]),
            hoverinfo="text",
            hovertext=(
                f"{RECIP_VEC_LABEL_PLAIN[i]} = [{b[i, 0]:.5f}, {b[i, 1]:.5f}, {b[i, 2]:.5f}]"
            ),
            name=RECIP_VEC_LABEL_PLAIN[i],
        ))
        shaft = b[i]
        traces.append(go.Cone(
            x=[float(shaft[0])], y=[float(shaft[1])], z=[float(shaft[2])],
            u=[float(shaft[0] * 0.15)],
            v=[float(shaft[1] * 0.15)],
            w=[float(shaft[2] * 0.15)],
            sizemode="absolute",
            sizeref=float(np.linalg.norm(shaft) * 0.06),
            colorscale=[[0, arrow_hex[i]], [1, arrow_hex[i]]],
            showscale=False,
            hoverinfo="skip",
            showlegend=False,
        ))

    traces.append(go.Scatter3d(
        x=[0], y=[0], z=[0],
        mode="markers+text",
        marker=dict(size=6, color="black"),
        text=["Γ"],
        textposition="top center",
        name="Γ",
    ))

    fig_pl = go.Figure(data=traces)
    fig_pl.update_layout(
        title=r"First BZ + reciprocal basis $\mathbf{a},\mathbf{b},\mathbf{c}$ (no band k-path)",
        scene=dict(
            aspectmode="data",
            xaxis_title="kx (Å⁻¹)",
            yaxis_title="ky (Å⁻¹)",
            zaxis_title="kz (Å⁻¹)",
        ),
        margin=dict(l=0, r=0, b=0, t=48),
    )
    os.makedirs(os.path.dirname(out_html) or ".", exist_ok=True)
    fig_pl.write_html(out_html, include_plotlyjs=True)
    print(f"Saved first-BZ reciprocal-basis HTML: {out_html}")
    print("(Band-structure k-paths are plotted separately; arrows are not the path.)")


def plot_first_bz_with_kpaths(
    cell: np.ndarray,
    paths: list[dict],
    out_png: str,
    out_html: str | None = None,
) -> None:
    """
    Single 3D figure: true first Brillouin zone (when available), primitive reciprocal
    cell outline, reciprocal basis arrows, and one or more k-path polylines with
    labelled vertices.

    Each entry in ``paths`` is a dict with keys:
      - ``kpts`` : (Nk, 3) Cartesian k-points (Å⁻¹)
      - ``tick_idx`` : indices of labelled vertices along the polyline
      - ``tick_labels`` : str labels for those vertices
      - ``color`` : matplotlib line colour (e.g. "#111111")
      - ``name`` : legend name for this polyline
      - ``linestyle`` : optional, default "-" (matplotlib only; Plotly uses solid lines)

    Axis limits use all paths plus BZ vertices so the full Wigner–Seitz cell stays
    visible (avoids quadrant-only cropping when the path sits in k>0 octant).
    """
    if not paths:
        raise ValueError("paths must contain at least one path specification.")

    cell = np.asarray(cell, dtype=float)
    b = reciprocal_lattice(cell)

    print("Reciprocal primitive rows a, b, c (1/Angstrom), rows of reciprocal_lattice(cell):")
    for i in range(3):
        nm = RECIP_VEC_LABEL_PLAIN[i]
        print(f"  {nm} = [{b[i, 0]:+.6f}, {b[i, 1]:+.6f}, {b[i, 2]:+.6f}]  "
              f"|{nm}| = {np.linalg.norm(b[i]):.6f}")

    origin = np.zeros(3)
    corners = [
        origin,
        b[0],
        b[1],
        b[2],
        b[0] + b[1],
        b[0] + b[2],
        b[1] + b[2],
        b[0] + b[1] + b[2],
    ]
    pp_edges = [
        (0, 1), (0, 2), (0, 3),
        (1, 4), (1, 5),
        (2, 4), (2, 6),
        (3, 5), (3, 6),
        (4, 7), (5, 7), (6, 7),
    ]

    bz_vertices = None
    bz_hull = None
    bz_all = None
    if USE_TRUE_FIRST_BZ:
        try:
            bz_vertices, bz_hull, bz_all = compute_true_first_bz(b)
            print(f"True first BZ (combined plot): {len(bz_vertices)} vertices, "
                  f"{len(bz_hull.simplices)} faces")
        except Exception as exc:
            print(f"WARNING: could not mesh true first BZ ({exc}); "
                  f"plotting primitive parallelepiped only.")

    stacks_for_limits = [np.array(corners)]
    if bz_vertices is not None:
        stacks_for_limits.append(np.asarray(bz_vertices, dtype=float))
    for spec in paths:
        stacks_for_limits.append(np.asarray(spec["kpts"], dtype=float))

    all_lim = np.vstack(stacks_for_limits)
    mins = np.min(all_lim, axis=0)
    maxs = np.max(all_lim, axis=0)
    span = np.where(maxs - mins < 1e-12, 1.0, maxs - mins)
    margin = 0.06 * span

    # --- Matplotlib PNG ---
    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111, projection="3d")

    if bz_all is not None and bz_hull is not None:
        face_verts = [bz_all[s] for s in bz_hull.simplices]
        bz_poly = Poly3DCollection(
            face_verts,
            alpha=0.14,
            facecolor="#4cc9f0",
            edgecolor="#023e8a",
            linewidths=0.35,
        )
        ax.add_collection3d(bz_poly)

    for i0, i1 in pp_edges:
        c0, c1 = corners[i0], corners[i1]
        ax.plot([c0[0], c1[0]], [c0[1], c1[1]], [c0[2], c1[2]],
                color="gray", linewidth=0.5, alpha=0.35)

    arrow_colors = ["#d62728", "#2ca02c", "#1f77b4"]
    for i in range(3):
        ax.quiver(0, 0, 0, b[i, 0], b[i, 1], b[i, 2],
                  arrow_length_ratio=0.08, color=arrow_colors[i], linewidth=1.5)
        ax.text(b[i, 0] * 1.08, b[i, 1] * 1.08, b[i, 2] * 1.08,
                RECIP_VEC_LABEL_MPL[i], color=arrow_colors[i], fontsize=10, fontweight="bold")

    for pi, spec in enumerate(paths):
        kpts = np.asarray(spec["kpts"], dtype=float)
        tick_idx = spec["tick_idx"]
        tick_labels = spec["tick_labels"]
        color = spec.get("color", "#111111")
        name = spec.get("name", f"path {pi + 1}")
        linestyle = spec.get("linestyle", "-")

        ax.plot(kpts[:, 0], kpts[:, 1], kpts[:, 2],
                color=color, linewidth=2.0, alpha=0.9, linestyle=linestyle,
                label=name)

        node_pts = np.array([kpts[idx] for idx in tick_idx])
        ax.scatter(node_pts[:, 0], node_pts[:, 1], node_pts[:, 2],
                   color=color, s=42, zorder=6)
        label_offset = margin * (0.35 + 0.25 * pi)
        for idx_i, lbl in zip(tick_idx, tick_labels):
            pt = kpts[idx_i]
            ax.text(
                pt[0] + label_offset[0],
                pt[1] + label_offset[1],
                pt[2] + label_offset[2],
                f"{lbl}",
                fontsize=9,
                color=color,
                ha="left",
                va="bottom",
            )

    ax.set_xlim(float(mins[0] - margin[0]), float(maxs[0] + margin[0]))
    ax.set_ylim(float(mins[1] - margin[1]), float(maxs[1] + margin[1]))
    ax.set_zlim(float(mins[2] - margin[2]), float(maxs[2] + margin[2]))

    ax.set_xlabel("kx (Å⁻¹)")
    ax.set_ylabel("ky (Å⁻¹)")
    ax.set_zlabel("kz (Å⁻¹)")
    n_paths = len(paths)
    title_core = (
        f"First BZ + {n_paths} k-path(s)"
        if bz_all is not None else
        f"Primitive reciprocal cell + {n_paths} k-path(s)"
    )
    ax.set_title(title_core)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.88)

    ranges_plot = (maxs + margin) - (mins - margin)
    try:
        ax.set_box_aspect((
            float(ranges_plot[0]),
            float(ranges_plot[1]),
            float(ranges_plot[2]),
        ))
    except Exception:
        pass

    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved BZ + k-path PNG: {out_png}")

    # --- Plotly HTML (optional) ---
    if out_html is not None:
        pp_x: list[float | None] = []
        pp_y: list[float | None] = []
        pp_z: list[float | None] = []
        for i0, i1 in pp_edges:
            c0, c1 = corners[i0], corners[i1]
            pp_x.extend([float(c0[0]), float(c1[0]), None])
            pp_y.extend([float(c0[1]), float(c1[1]), None])
            pp_z.extend([float(c0[2]), float(c1[2]), None])

        trace_pp = go.Scatter3d(
            x=pp_x, y=pp_y, z=pp_z,
            mode="lines",
            line=dict(color="rgba(150,150,150,0.35)", width=2),
            hoverinfo="skip",
            name="primitive cell",
        )

        arrow_traces = []
        arrow_hex = ["#d62728", "#2ca02c", "#1f77b4"]
        for i in range(3):
            arrow_traces.append(go.Scatter3d(
                x=[0, float(b[i, 0])], y=[0, float(b[i, 1])], z=[0, float(b[i, 2])],
                mode="lines+text",
                line=dict(color=arrow_hex[i], width=4),
                text=["", RECIP_VEC_LABEL_PLAIN[i]],
                textposition="top center",
                textfont=dict(size=12, color=arrow_hex[i]),
                hoverinfo="text",
                hovertext=(
                    f"{RECIP_VEC_LABEL_PLAIN[i]} = [{b[i,0]:.4f}, {b[i,1]:.4f}, {b[i,2]:.4f}]"
                ),
                name=RECIP_VEC_LABEL_PLAIN[i],
            ))
            shaft = b[i]
            arrow_traces.append(go.Cone(
                x=[float(shaft[0])], y=[float(shaft[1])], z=[float(shaft[2])],
                u=[float(shaft[0] * 0.15)], v=[float(shaft[1] * 0.15)], w=[float(shaft[2] * 0.15)],
                sizemode="absolute",
                sizeref=float(np.linalg.norm(shaft) * 0.06),
                colorscale=[[0, arrow_hex[i]], [1, arrow_hex[i]]],
                showscale=False,
                hoverinfo="skip",
                name="",
                showlegend=False,
            ))

        traces_out: list = []
        if bz_all is not None and bz_hull is not None:
            traces_out.append(go.Mesh3d(
                x=bz_all[:, 0],
                y=bz_all[:, 1],
                z=bz_all[:, 2],
                i=bz_hull.simplices[:, 0],
                j=bz_hull.simplices[:, 1],
                k=bz_hull.simplices[:, 2],
                opacity=0.14,
                color="#4cc9f0",
                name="first BZ",
            ))

        traces_out.append(trace_pp)
        traces_out.extend(arrow_traces)

        for pi, spec in enumerate(paths):
            kpts = np.asarray(spec["kpts"], dtype=float)
            tick_idx = spec["tick_idx"]
            tick_labels = spec["tick_labels"]
            color = spec.get("color", "#111111")
            name = spec.get("name", f"path {pi + 1}")
            node_pts = np.array([kpts[idx] for idx in tick_idx])

            traces_out.append(go.Scatter3d(
                x=kpts[:, 0].tolist(),
                y=kpts[:, 1].tolist(),
                z=kpts[:, 2].tolist(),
                mode="lines",
                line=dict(color=color, width=4),
                hoverinfo="skip",
                name=name,
            ))
            traces_out.append(go.Scatter3d(
                x=node_pts[:, 0].tolist(),
                y=node_pts[:, 1].tolist(),
                z=node_pts[:, 2].tolist(),
                mode="markers+text",
                marker=dict(size=6, color=color),
                text=tick_labels,
                textposition="top center",
                textfont=dict(size=11, color=color),
                hoverinfo="text",
                hovertext=[
                    f"{lbl} ({node_pts[i,0]:.4f}, {node_pts[i,1]:.4f}, {node_pts[i,2]:.4f})"
                    for i, lbl in enumerate(tick_labels)
                ],
                name=f"{name} (nodes)",
            ))

        xr = [float(mins[0] - margin[0]), float(maxs[0] + margin[0])]
        yr = [float(mins[1] - margin[1]), float(maxs[1] + margin[1])]
        zr = [float(mins[2] - margin[2]), float(maxs[2] + margin[2])]

        fig_plotly = go.Figure(data=traces_out)
        fig_plotly.update_layout(
            title=title_core,
            scene=dict(
                aspectmode="data",
                xaxis_title="kx (Å⁻¹)",
                yaxis_title="ky (Å⁻¹)",
                zaxis_title="kz (Å⁻¹)",
                xaxis=dict(range=xr),
                yaxis=dict(range=yr),
                zaxis=dict(range=zr),
            ),
            margin=dict(l=0, r=0, b=0, t=48),
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        )

        os.makedirs(os.path.dirname(out_html) or ".", exist_ok=True)
        fig_plotly.write_html(out_html, include_plotlyjs=True)
        print(f"Saved BZ + k-path HTML: {out_html}")

    print("Note: shaded polyhedron is the true first Brillouin zone when available; "
          "gray edges outline the primitive reciprocal parallelepiped.")
    print("Note: Γ–X–Y–Z on the wireframe are ½ **a**, ½ **b**, ½ **c**; the band path "
          "follows Y–Γ–Z–W₁–… from ``PAPER_KPATH_NODES``.")


def plot_kpath_reciprocal(
    cell: np.ndarray,
    kpts: np.ndarray,
    tick_idx: list[int],
    tick_labels: list[str],
    out_png: str,
    out_html: str | None = None,
) -> None:
    """
    Plot k-path(s) together with the first Brillouin zone (``plot_first_bz_with_kpaths``).

    When ``DO_KPATH_COMPARE_SECOND`` is True, overlays the legacy Γ–a–a+b–b–Γ loop.
    """
    b = reciprocal_lattice(cell)
    paths: list[dict] = [{
        "kpts": kpts,
        "tick_idx": tick_idx,
        "tick_labels": tick_labels,
        "color": "#111111",
        "linestyle": "-",
        "name": KPATH_LEGEND_STD,
    }]
    if DO_KPATH_COMPARE_SECOND:
        k2, t2, l2 = make_k_path_primitive_b1b2_loop(b, n_per_segment=45)
        paths.append({
            "kpts": k2,
            "tick_idx": t2,
            "tick_labels": l2,
            "color": "#c0392b",
            "linestyle": "--",
            "name": KPATH_LEGEND_PRIMITIVE_LOOP,
        })
    plot_first_bz_with_kpaths(cell, paths, out_png, out_html)

def band_structure(
    atoms: Atoms,
    cutoff: float,
    t: float,
    onsite: float,
    model_edges: list[tuple] | None = None,
) -> None:
    """
    Compute and plot the tight-binding band structure along a simple k-path.

    Parameters
    ----------
    model_edges : optional list of (i, j, dr, shift, amp) 5-tuples from build_model().
        When provided, used for H(k) instead of the CUT-based NN edge list.
        Supports nn, nn+extra, and nn+2nn models with per-edge amplitudes.
        When None (default), falls back to build_edge_list_pbc(atoms, cutoff).
    """
    n = len(atoms)

    # Use caller-supplied model edges if available; otherwise build NN edges.
    if model_edges is not None:
        edges_pbc = model_edges
    else:
        edges_4 = build_edge_list_pbc(atoms, cutoff)
        edges_pbc = [(i, j, dr, sh, T_HOP) for (i, j, dr, sh) in edges_4]

    # Cross-boundary edge count
    ii_s, jj_s, SS_s = neighbor_list("ijS", atoms, float(cutoff))
    n_cross = int(np.sum(np.any(SS_s != 0, axis=1) & (ii_s != jj_s)))
    n_total_directed = int(np.sum(ii_s != jj_s))
    print(f"Cross-boundary hops: {n_cross}/{n_total_directed} directed "
          f"({100*n_cross/max(n_total_directed,1):.1f}%)")

    # Duplicate-dr sanity check: detect multi-image pairs
    pair_dists: dict[tuple[int, int], list[float]] = defaultdict(list)
    for edge in edges_pbc:
        i_e, j_e, dr_e = edge[0], edge[1], edge[2]
        pair_dists[(i_e, j_e)].append(np.linalg.norm(dr_e))
    dup_pairs = [(k, v) for k, v in pair_dists.items() if len(v) > 1]
    if dup_pairs:
        print(f"Multi-image pairs: {len(dup_pairs)} (i,j) pairs with >1 edge")
        for (i_p, j_p), dists in dup_pairs[:5]:
            print(f"  ({i_p},{j_p}): distances = {[f'{d:.4f}' for d in sorted(dists)]}")
    else:
        print("Multi-image pairs: 0 (no duplicate (i,j) with different S)")

    cell = atoms.get_cell().array
    b = reciprocal_lattice(cell)
    kpts, tick_idx, tick_labels = make_k_path(b, n_per_segment=60)
    nk = len(kpts)

    if DO_KPATH_VIZ:
        plot_kpath_reciprocal(cell, kpts, tick_idx, tick_labels,
                              OUT_KPATH_PNG, OUT_KPATH_HTML)
    if DO_BZ_WIREFRAME_VIZ:
        render_bz_wireframe_basis_style(
            cell,
            OUT_BZ_WIREFRAME_PNG,
            OUT_BZ_WIREFRAME_HTML if EXPORT_INTERACTIVE else None,
        )

    # Gamma-point sanity check: H(k=0) should match real-space H built from same edges
    h_gamma = build_tb_hamiltonian_k(n, edges_pbc, np.zeros(3), t, onsite,
                                      reciprocal_b=b)
    h_ref = np.zeros((n, n), dtype=float)
    for i_s in range(n):
        h_ref[i_s, i_s] = onsite
    for edge in edges_pbc:
        si, sj = edge[0], edge[1]
        amp_e = edge[4] if len(edge) >= 5 else t
        h_ref[si, sj] += -amp_e
        h_ref[sj, si] += -amp_e
    for hop in EXTRA_HOPS:
        hi, hj, amp = hop["i"], hop["j"], hop["amplitude"]
        h_ref[hi, hj] += -amp
        h_ref[hj, hi] += -amp
    max_diff = float(np.max(np.abs(h_gamma.real - h_ref)))
    if np.allclose(h_gamma.real, h_ref):
        extra_tag = f" (incl. {len(EXTRA_HOPS)} extra hops)" if EXTRA_HOPS else ""
        print(f"Gamma-point check PASSED{extra_tag} (max diff = {max_diff:.2e})")
    else:
        print(f"WARNING: Gamma-point check FAILED (max |H(Gamma).real - H_ref| = {max_diff:.2e})")

    # Periodicity check: eigenvalues at k must equal eigenvalues at k + G
    k_test = 0.1 * b[0]
    evals_k = np.linalg.eigvalsh(build_tb_hamiltonian_k(n, edges_pbc, k_test, t, onsite,
                                                         reciprocal_b=b))
    evals_kG = np.linalg.eigvalsh(build_tb_hamiltonian_k(n, edges_pbc, k_test + b[0], t, onsite,
                                                         reciprocal_b=b))
    max_diff_per = float(np.max(np.abs(evals_k - evals_kG)))
    if max_diff_per < 1e-8:
        print(f"Periodicity check PASSED: E(0.1*a) == E(0.1*a + a) (max diff = {max_diff_per:.2e})")
    else:
        print(f"WARNING: Periodicity check FAILED: max |E(k) - E(k+G)| = {max_diff_per:.2e}")

    # Wrapped phase diagnostic
    dr_array = np.array([edge[2] for edge in edges_pbc])
    raw_phases = kpts @ dr_array.T                          # (Nk, E)
    wrapped = (raw_phases + np.pi) % (2 * np.pi) - np.pi    # wrap to [-pi, pi]
    max_raw = float(np.max(np.abs(raw_phases)))
    max_wrapped = float(np.max(np.abs(wrapped)))
    print(f"Phase diagnostic: max |k·dr| raw = {max_raw:.4f} rad, "
          f"wrapped = {max_wrapped:.4f} rad ({np.degrees(max_wrapped):.1f} deg)")

    all_evals = np.empty((nk, n), dtype=float)
    for ik, k_cart in enumerate(kpts):
        hk = build_tb_hamiltonian_k(n, edges_pbc, k_cart, t, onsite,
                                    reciprocal_b=b)
        all_evals[ik] = np.linalg.eigvalsh(hk)

    # Bandwidth per band
    bandwidths = np.max(all_evals, axis=0) - np.min(all_evals, axis=0)
    order = np.argsort(bandwidths)
    print("10 flattest bands:  ", ", ".join(f"n={order[i]}:dE={bandwidths[order[i]]:.4f}" for i in range(min(10, n))))
    print("10 most dispersive: ", ", ".join(f"n={order[-(i+1)]}:dE={bandwidths[order[-(i+1)]]:.4f}" for i in range(min(10, n))))

    # Cumulative |delta-k| distance along the path for x-axis
    dk = np.linalg.norm(np.diff(kpts, axis=0), axis=1)
    x = np.zeros(nk)
    x[1:] = np.cumsum(dk)

    fig, ax = plt.subplots(figsize=(8, 5))
    for band in range(n):
        ax.plot(x, all_evals[:, band], color="steelblue", linewidth=0.6)

    for idx in tick_idx:
        ax.axvline(x[idx], color="gray", linewidth=0.5, linestyle="--")
    ax.set_xticks([x[idx] for idx in tick_idx])
    ax.set_xticklabels(tick_labels)
    ax.set_ylabel("Energy")
    ax.set_title(f"Band structure (cutoff={cutoff:.2f} Å, N={n} bands, Nk={nk})")
    ax.set_xlim(x[0], x[-1])

    out_path = OUT_BANDS.format(cut=cutoff)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Band structure: Nk={nk} points, {n} bands, saved: {out_path}")

    # --- Fermi-level zoom plot + gap diagnostics ---
    if DO_BANDS_FERMI_ZOOM:
        # Compute E_F_plot
        m = n // 2
        if n < 4:
            print("WARNING: n_sites < 4, skipping Fermi zoom plot.")
        elif m == 0:
            print("WARNING: n_sites//2 == 0, skipping Fermi zoom plot.")
        else:
            if EF_MODE == "fixed":
                ef_plot = float(EF_FIXED)
            else:
                ef_k = 0.5 * (all_evals[:, m - 1] + all_evals[:, m])
                ef_plot = float(np.median(ef_k))
            print(f"E_F_plot = {ef_plot:.6f} ({EF_MODE})")

            # Select N_SHOW closest bands
            n_show = min(N_SHOW, n)
            if N_SHOW >= n:
                print(f"WARNING: N_SHOW={N_SHOW} >= n_sites={n}, clamping to {n}.")
            d_n = np.array([float(np.min(np.abs(all_evals[:, band_i] - ef_plot)))
                            for band_i in range(n)])
            closest_idx = np.argsort(d_n)[:n_show]
            mean_e = np.array([float(np.mean(all_evals[:, bi])) for bi in closest_idx])
            closest_idx = closest_idx[np.argsort(mean_e)]

            print(f"Selected {n_show} bands closest to E_F:")
            for bi in closest_idx:
                print(f"  band {bi}: d_n={d_n[bi]:.6f}, <E>={np.mean(all_evals[:, bi]):.6f}")

            # Gap diagnostics at half filling
            gap_k = all_evals[:, m] - all_evals[:, m - 1]
            gap_min = float(np.min(gap_k))
            gap_med = float(np.median(gap_k))
            gap_argmin = int(np.argmin(gap_k))
            e_lo = float(all_evals[gap_argmin, m - 1])
            e_hi = float(all_evals[gap_argmin, m])
            print(f"Direct gap at half filling: gap_min={gap_min:.6e}, "
                  f"gap_med={gap_med:.6e}")
            print(f"  Worst k (idx={gap_argmin}): "
                  f"E_{{m-1}}={e_lo:.6f}, E_m={e_hi:.6f}")
            if gap_min < GAP_TOL:
                print("Result: no clean bulk gap at half filling along this "
                      "k-path (gap_min < GAP_TOL).")
            else:
                print("Result: finite direct gap at half filling along this "
                      "k-path (gap_min >= GAP_TOL).")

            # Zoom plot
            fig_z, ax_z = plt.subplots(figsize=(8, 5))
            for bi in closest_idx:
                ax_z.plot(x, all_evals[:, bi], linewidth=0.8)
            ax_z.axhline(ef_plot, color="black", linewidth=1.0, linestyle="--",
                         label=f"E_F = {ef_plot:.4f}")

            if gap_min >= GAP_TOL:
                ax_z.fill_between(x, all_evals[:, m - 1], all_evals[:, m],
                                  color="yellow", alpha=0.15, label="direct gap")

            for idx in tick_idx:
                ax_z.axvline(x[idx], color="gray", linewidth=0.5, linestyle="--")
            ax_z.set_xticks([x[idx] for idx in tick_idx])
            ax_z.set_xticklabels(tick_labels)
            ax_z.set_ylabel("Energy")
            ax_z.set_title(f"Bands near E_F (cutoff={cutoff:.2f} Å, Nk={nk}, "
                           f"N_show={n_show}, E_F={ef_plot:.4f})")
            ax_z.set_xlim(x[0], x[-1])

            sel_evals = all_evals[:, closest_idx]
            e_min_sel = float(np.min(sel_evals))
            e_max_sel = float(np.max(sel_evals))
            pad = 0.1 * (e_max_sel - e_min_sel + 1e-12)
            ax_z.set_ylim(e_min_sel - pad, e_max_sel + pad)
            ax_z.legend(loc="best", fontsize=8)

            out_zoom = OUT_BANDS_ZOOM.format(cut=cutoff, nshow=n_show)
            os.makedirs(os.path.dirname(out_zoom), exist_ok=True)
            fig_z.savefig(out_zoom, dpi=200, bbox_inches="tight")
            plt.close(fig_z)
            print(f"Saved zoom plot: {out_zoom}")


def build_Hk_from_couplings(
    n: int,
    couplings: dict[tuple[int, int, tuple[float, float, float]], float],
    k_cart: np.ndarray,
    onsite: float,
    reciprocal_b: np.ndarray | None = None,
) -> np.ndarray:
    """
    Build Bloch Hamiltonian H(k) from a coupling dictionary.

    Parameters
    ----------
    n : int
        Number of sites.
    couplings : dict
        Keys: (i, j, R_tuple) where R_tuple = (Rx, Ry, Rz).
        Values: coupling amplitude A.
    k_cart : np.ndarray, shape (3,)
        k-point in Cartesian coordinates.
    onsite : float
        On-site energy.
    reciprocal_b : np.ndarray or None
        Reciprocal rows b_i; enables true first-BZ folding when USE_TRUE_FIRST_BZ.

    Returns
    -------
    Hk : np.ndarray, shape (n, n), dtype complex
        Hermitian Bloch Hamiltonian.
    """
    k_cart = _effective_k_cart_for_tb(k_cart, reciprocal_b)

    hk = np.zeros((n, n), dtype=complex)

    for i in range(n):
        hk[i, i] = onsite

    for (i, j, r_key), amplitude in couplings.items():
        r_vec = np.array(r_key, dtype=float)
        phase = np.exp(1j * np.dot(k_cart, r_vec))
        hk[i, j] += amplitude * phase

    # Enforce Hermiticity
    hk = 0.5 * (hk + hk.conj().T)
    assert np.allclose(hk, hk.conj().T, atol=1e-10), "H(k) must be Hermitian"
    return hk


def band_structure_pathsum(
    atoms: Atoms,
    cutoff: float,
    t: float,
    onsite: float,
    L: int,
    alpha: float,
) -> None:
    """
    Compute and plot band structure using path-sum expansion up to length L.

    For each site pair (i, j), enumerate all paths of length ℓ = 1..L,
    accumulate weighted couplings with weight alpha^(ℓ-1), and build H(k).
    """
    n = len(atoms)
    cell = atoms.get_cell().array
    pos = atoms.get_positions()

    # Build directed edge list with real-space vectors
    ii, jj, SS = neighbor_list("ijS", atoms, float(cutoff))
    directed_edges: list[tuple[int, int, np.ndarray]] = []
    for i, j, S in zip(ii.tolist(), jj.tolist(), SS.tolist()):
        if i == j:
            continue
        sx, sy, sz = float(S[0]), float(S[1]), float(S[2])
        r_vec = (pos[j] - pos[i]) + sx * cell[0] + sy * cell[1] + sz * cell[2]
        directed_edges.append((i, j, r_vec))

    # Build adjacency for path enumeration
    adj_out: dict[int, list[tuple[int, np.ndarray]]] = {i: [] for i in range(n)}
    for i, j, r_vec in directed_edges:
        adj_out[i].append((j, r_vec))

    # Enumerate paths up to length L and accumulate couplings
    couplings: dict[tuple[int, int, tuple[float, float, float]], float] = {}

    for ell in range(1, L + 1):
        weight = alpha ** (ell - 1)
        if ell == 1:
            # Single hop: amplitude = -t * weight
            for i, j, r_vec in directed_edges:
                r_key = tuple(np.round(r_vec, 10).tolist())
                key = (i, j, r_key)
                couplings[key] = couplings.get(key, 0.0) + (-t * weight)
        else:
            # Multi-hop paths: enumerate via BFS/DFS up to length ell
            # For each starting site, enumerate all ell-step paths
            # This is exponentially expensive but deterministic
            for start in range(n):
                # paths: list of (current_node, total_R, path_nodes)
                current_paths = [(start, np.zeros(3), [start])]
                for step in range(ell):
                    next_paths = []
                    for current_node, r_total, path_nodes in current_paths:
                        for neighbor, r_edge in adj_out[current_node]:
                            next_paths.append((neighbor, r_total + r_edge, path_nodes + [neighbor]))
                    current_paths = next_paths

                # Now current_paths contains all ell-step paths from start
                for end_node, r_total, path_nodes in current_paths:
                    # Coupling from start to end_node via this path
                    r_key = tuple(np.round(r_total, 10).tolist())
                    key = (start, end_node, r_key)
                    # Amplitude: (-t)^ell * weight, but we already included -t in weight calculation
                    # Actually: each hop contributes -t, so ell hops give (-t)^ell
                    # Weight is alpha^(ell-1), so total is (-t)^ell * alpha^(ell-1)
                    amplitude = ((-t) ** ell) * weight
                    couplings[key] = couplings.get(key, 0.0) + amplitude

    # Compute reciprocal lattice and k-path
    b = reciprocal_lattice(cell)
    kpts, tick_idx, tick_labels = make_k_path(b, n_per_segment=60)
    nk = len(kpts)

    # Compute bands
    all_evals = np.empty((nk, n), dtype=float)
    for ik, k_cart in enumerate(kpts):
        hk = build_Hk_from_couplings(n, couplings, k_cart, onsite, reciprocal_b=b)
        all_evals[ik] = np.linalg.eigvalsh(hk)

    # L=1 consistency check: must match NN bands
    if L == 1:
        edges_pbc = build_edge_list_pbc(atoms, cutoff)
        all_evals_nn = np.empty((nk, n), dtype=float)
        for ik, k_cart in enumerate(kpts):
            hk_nn = build_tb_hamiltonian_k(n, edges_pbc, k_cart, t, onsite,
                                           reciprocal_b=b)
            all_evals_nn[ik] = np.linalg.eigvalsh(hk_nn)
        max_abs_diff = float(np.max(np.abs(all_evals - all_evals_nn)))
        print(f"Pathsum L=1 check: max |DeltaE| = {max_abs_diff:.2e}")
        if max_abs_diff >= 1e-8:
            print(f"WARNING: L=1 consistency check FAILED (threshold 1e-8). Not saving plot.")
            return

    # Cumulative distance for x-axis
    dk = np.linalg.norm(np.diff(kpts, axis=0), axis=1)
    x = np.zeros(nk)
    x[1:] = np.cumsum(dk)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    for band in range(n):
        ax.plot(x, all_evals[:, band], color="steelblue", linewidth=0.6)

    for idx in tick_idx:
        ax.axvline(x[idx], color="gray", linewidth=0.5, linestyle="--")
    ax.set_xticks([x[idx] for idx in tick_idx])
    ax.set_xticklabels(tick_labels)
    ax.set_ylabel("Energy")
    ax.set_title(f"Path-sum bands (L={L}, alpha={alpha:.2f}, cutoff={cutoff:.2f} Angstrom)")
    ax.set_xlim(x[0], x[-1])

    out_path = OUT_PATHSUM_BANDS.format(L=L, cut=cutoff)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Path-sum bands: Nk={nk}, L={L}, alpha={alpha:.2f}, {len(couplings)} coupling terms, saved: {out_path}")


def build_directed_hops_with_dr(
    atoms: Atoms, cutoff: float
) -> list[tuple[int, int, np.ndarray, float]]:
    """
    Return directed hops (i->j) from neighbor_list up to cutoff,
    with dr = (pos[j]-pos[i]) + S·cell and dist = ||dr||.
    """
    pos = atoms.get_positions()
    cell = atoms.get_cell().array
    ii, jj, SS = neighbor_list("ijS", atoms, float(cutoff))
    
    directed_hops = []
    for i, j, S in zip(ii.tolist(), jj.tolist(), SS.tolist()):
        if i == j:
            continue
        sx, sy, sz = float(S[0]), float(S[1]), float(S[2])
        dr = (pos[j] - pos[i]) + sx * cell[0] + sy * cell[1] + sz * cell[2]
        dist = float(np.linalg.norm(dr))
        directed_hops.append((i, j, dr, dist))
    
    return directed_hops


def distance_shells(
    dists: np.ndarray,
    tol_abs: float = 5e-3,
    tol_rel: float = 2e-3,
) -> list[float]:
    """
    Cluster distances into shells using greedy clustering with combined tolerance.
    
    Uses max(tol_abs, tol_rel * center) as the tolerance criterion.
    Returns sorted shell centers.
    """
    dists = np.asarray(dists)
    if len(dists) == 0:
        return []
    
    sorted_dists = np.sort(dists)
    shells = []
    current_shell = [sorted_dists[0]]
    
    for d in sorted_dists[1:]:
        if len(current_shell) > 0:
            center = np.mean(current_shell)
            tol_combined = max(tol_abs, tol_rel * center)
            if abs(d - center) <= tol_combined:
                current_shell.append(d)
            else:
                shells.append(float(np.mean(current_shell)))
                current_shell = [d]
    
    if len(current_shell) > 0:
        shells.append(float(np.mean(current_shell)))
    
    return shells


def assign_shell(
    dist: float,
    shells: list[float],
    tol_abs: float = 5e-3,
    tol_rel: float = 2e-3,
) -> int | None:
    """
    Return shell index (0-based) if within combined tolerance of a shell center, else None.
    
    Uses max(tol_abs, tol_rel * shell_center) as the tolerance criterion.
    """
    for idx, shell_center in enumerate(shells):
        tol_combined = max(tol_abs, tol_rel * shell_center)
        if abs(dist - shell_center) <= tol_combined:
            return idx
    return None


def build_Hk_shells(
    n: int,
    directed_hops: list[tuple[int, int, np.ndarray, float]],
    shells: list[float],
    t_list: list[float],
    k_cart: np.ndarray,
    onsite: float,
    tol_abs: float,
    tol_rel: float,
    reciprocal_b: np.ndarray | None = None,
) -> np.ndarray:
    """
    Build H(k) from shell-based hoppings with proper deduplication.

    Deduplicates directed hops into canonical undirected set before filling H(k).
    No post-hoc symmetrization; fills both H[i,j] and H[j,i] explicitly.
    """
    k_cart = _effective_k_cart_for_tb(k_cart, reciprocal_b)

    hk = np.zeros((n, n), dtype=complex)
    
    for i in range(n):
        hk[i, i] = onsite
    
    # Deduplicate hops: canonical key is (min(i,j), max(i,j), rounded_dr_canonical)
    canonical_hops: dict[tuple[int, int, tuple[float, ...]], tuple[int, int, np.ndarray, int]] = {}
    
    for i, j, dr, dist in directed_hops:
        shell_idx = assign_shell(dist, shells, tol_abs, tol_rel)
        if shell_idx is None or shell_idx >= len(t_list):
            continue
        
        # Canonicalize: i < j
        if i < j:
            i_can, j_can, dr_can = i, j, dr
        else:
            i_can, j_can, dr_can = j, i, -dr
        
        dr_key = tuple(np.round(dr_can, 10).tolist())
        key = (i_can, j_can, dr_key)
        
        if key not in canonical_hops:
            canonical_hops[key] = (i_can, j_can, dr_can, shell_idx)
    
    # Fill H[k] explicitly for both (i,j) and (j,i)
    for i_can, j_can, dr_can, shell_idx in canonical_hops.values():
        amplitude = -t_list[shell_idx]
        phase = np.exp(1j * np.dot(k_cart, dr_can))
        hk[i_can, j_can] += amplitude * phase
        hk[j_can, i_can] += amplitude * np.conj(phase)
    
    assert np.allclose(hk, hk.conj().T, atol=1e-10), "H(k) must be Hermitian"
    return hk


def band_structure_shells(
    atoms: Atoms,
    cutoff: float,
    onsite: float,
    L_max: int,
    t0: float = 1.0,
    decay: float = 0.6,
    tol_abs: float = 5e-3,
    tol_rel: float = 2e-3,
    min_hops: int | None = None,
    energy_window: float | None = None,
) -> None:
    """
    Compute and plot shell-based tight-binding bands for L=1..L_max.
    
    Automatically selects NN shell s0 as the first shell with hop_count >= min_hops.
    Then uses shells s0..(s0+L-1) for the L-shell model.
    
    min_hops: Minimum hop count to qualify as NN shell. Default: max(12, N//6).
    energy_window: If not None, plot only bands within [-window, +window] around E=0.
    """
    n = len(atoms)
    cell = atoms.get_cell().array
    
    if min_hops is None:
        # Default: max(12, N//6). For well-connected periodic systems,
        # NN shell should have at least ~12-20 hops to ensure reasonable coordination.
        # Scale with system size but keep a minimum threshold.
        min_hops = max(12, n // 6)
    
    # Build all directed hops up to large cutoff
    directed_hops = build_directed_hops_with_dr(atoms, cutoff)
    if len(directed_hops) == 0:
        print(f"Warning: no hops found within cutoff={cutoff:.2f}")
        return
    
    print(f"Total directed hops within cutoff={cutoff:.2f} Å: {len(directed_hops)}")
    
    # Compute distance shells with combined tolerance
    all_dists = np.array([hop[3] for hop in directed_hops])
    shells = distance_shells(all_dists, tol_abs=tol_abs, tol_rel=tol_rel)
    
    if len(shells) == 0:
        print("Warning: no shells found")
        return
    
    # ISSUE 2: Precompute shell index for every directed hop ONCE
    # Use -1 for hops that don't belong to any shell
    hop_shell_idx_list = []
    for hop in directed_hops:
        s = assign_shell(hop[3], shells, tol_abs, tol_rel)
        hop_shell_idx_list.append(s if s is not None else -1)
    hop_shell_idx = np.array(hop_shell_idx_list, dtype=int)
    
    # Compute hop counts per shell
    shell_counts = []
    for s_idx in range(len(shells)):
        count = int(np.sum(hop_shell_idx == s_idx))
        shell_counts.append(count)
    
    # Print detailed shell diagnostics with precomputed assignments
    print(f"Distance shells (first {min(10, len(shells))}):")
    for s_idx in range(min(10, len(shells))):
        shell_center = shells[s_idx]
        hop_mask = (hop_shell_idx == s_idx)
        hop_dists = [directed_hops[i][3] for i in range(len(directed_hops)) if hop_mask[i]]
        count = len(hop_dists)
        if count > 0:
            d_min = min(hop_dists)
            d_max = max(hop_dists)
            print(f"  Shell {s_idx}: center={shell_center:.4f} Å, {count} hops, range=[{d_min:.4f}, {d_max:.4f}]")
        else:
            print(f"  Shell {s_idx}: center={shell_center:.4f} Å, 0 hops")
    
    # ISSUE 1: Automatic NN shell selection (first shell with hop_count >= MIN_HOPS)
    s0 = None
    for s_idx in range(len(shells)):
        if shell_counts[s_idx] >= min_hops:
            s0 = s_idx
            break
    
    if s0 is None:
        print(f"ERROR: No shell found with hop_count >= {min_hops}. Aborting.")
        return
    
    print(f"\nAutomatic NN shell selection: MIN_HOPS = {min_hops}")
    print(f"  Chosen NN shell: s0 = {s0}, distance = {shells[s0]:.4f} Å, hop_count = {shell_counts[s0]}")
    
    # Compute reciprocal lattice and k-path
    b = reciprocal_lattice(cell)
    kpts, tick_idx, tick_labels = make_k_path(b, n_per_segment=60)
    nk = len(kpts)
    
    # For each L=1..L_max, use shells s0..(s0+L-1)
    for L in range(1, min(L_max, len(shells) - s0) + 1):
        shell_indices = list(range(s0, s0 + L))
        t_list = [t0 * (decay ** s) for s in range(L)]
        
        # Filter hops: only those in shells s0..(s0+L-1)
        hop_mask = np.isin(hop_shell_idx, shell_indices)
        relevant_hops = [directed_hops[i] for i in range(len(directed_hops)) if hop_mask[i]]
        relevant_shell_idx = [hop_shell_idx[i] - s0 for i in range(len(directed_hops)) if hop_mask[i]]
        
        print(f"\nL={L}: using shells {shell_indices}, t_list = {[f'{t:.3f}' for t in t_list]}")
        print(f"  Total hops in shells s{s0}..s{s0+L-1}: {len(relevant_hops)}")
        
        # Count unique canonical hops
        canonical_hops_L: dict[tuple[int, int, tuple[float, ...]], int] = {}
        for (i, j, dr, dist), rel_s in zip(relevant_hops, relevant_shell_idx):
            if i < j:
                i_can, j_can, dr_can = i, j, dr
            else:
                i_can, j_can, dr_can = j, i, -dr
            dr_key = tuple(np.round(dr_can, 10).tolist())
            key = (i_can, j_can, dr_key)
            if key not in canonical_hops_L:
                canonical_hops_L[key] = rel_s
        
        dup_count = len(relevant_hops) - len(canonical_hops_L)
        if dup_count > 0:
            print(f"  Deduplicating: {dup_count} duplicate hops removed, {len(canonical_hops_L)} unique")
        
        # Compute bands using shell method
        all_evals = np.empty((nk, n), dtype=float)
        for ik, k_cart in enumerate(kpts):
            hk = np.zeros((n, n), dtype=complex)
            for i in range(n):
                hk[i, i] = onsite
            
            # Fill using canonical hops
            k_eff = _effective_k_cart_for_tb(k_cart, b)
            for (i_can, j_can, dr_key), rel_s in canonical_hops_L.items():
                dr_can = np.array(dr_key)
                amplitude = -t_list[rel_s]
                phase = np.exp(1j * np.dot(k_eff, dr_can))
                hk[i_can, j_can] += amplitude * phase
                hk[j_can, i_can] += amplitude * np.conj(phase)
            
            assert np.allclose(hk, hk.conj().T, atol=1e-10), "H(k) must be Hermitian"
            all_evals[ik] = np.linalg.eigvalsh(hk)
        
        # ISSUE 3: L=1 exact check using SAME hop set from shell s0
        if L == 1:
            print(f"L=1 consistency check: comparing shell-model vs exact NN-only reference")
            
            all_evals_ref = np.empty((nk, n), dtype=float)
            for ik, k_cart in enumerate(kpts):
                hk_ref = np.zeros((n, n), dtype=complex)
                for i in range(n):
                    hk_ref[i, i] = onsite
                
                # Use EXACT same canonical hops as shell model
                k_eff = _effective_k_cart_for_tb(k_cart, b)
                for (i_can, j_can, dr_key), rel_s in canonical_hops_L.items():
                    dr_can = np.array(dr_key)
                    amplitude = -t0  # L=1 uses only t0
                    phase = np.exp(1j * np.dot(k_eff, dr_can))
                    hk_ref[i_can, j_can] += amplitude * phase
                    hk_ref[j_can, i_can] += amplitude * np.conj(phase)
                
                all_evals_ref[ik] = np.linalg.eigvalsh(hk_ref)
            
            max_abs_diff = float(np.max(np.abs(all_evals - all_evals_ref)))
            print(f"  Shell model hops: {len(canonical_hops_L)}, Reference hops: {len(canonical_hops_L)}")
            print(f"  max |DeltaE| across all k: {max_abs_diff:.2e}")
            
            if max_abs_diff >= 1e-8:
                # Detailed diagnostics at Gamma
                hk_shell_gamma = np.zeros((n, n), dtype=complex)
                hk_ref_gamma = np.zeros((n, n), dtype=complex)
                for i in range(n):
                    hk_shell_gamma[i, i] = onsite
                    hk_ref_gamma[i, i] = onsite
                
                for (i_can, j_can, dr_key), rel_s in canonical_hops_L.items():
                    dr_can = np.array(dr_key)
                    phase = np.exp(1j * np.dot(np.zeros(3), dr_can))  # k=0
                    hk_shell_gamma[i_can, j_can] += -t_list[rel_s] * phase
                    hk_shell_gamma[j_can, i_can] += -t_list[rel_s] * np.conj(phase)
                    hk_ref_gamma[i_can, j_can] += -t0 * phase
                    hk_ref_gamma[j_can, i_can] += -t0 * np.conj(phase)
                
                max_h_diff = float(np.max(np.abs(hk_shell_gamma - hk_ref_gamma)))
                print(f"  max |H_shell(Gamma) - H_ref(Gamma)| elementwise: {max_h_diff:.2e}")
                print(f"WARNING: L=1 consistency check FAILED (threshold 1e-8). Not saving plots.")
                return
            else:
                print(f"  L=1 check PASSED")
        
        # Cumulative distance for x-axis
        dk = np.linalg.norm(np.diff(kpts, axis=0), axis=1)
        x = np.zeros(nk)
        x[1:] = np.cumsum(dk)
        
        # Plot bands within energy window for clearer visualization of symmetry features
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot only bands within energy window (if specified)
        bands_plotted = 0
        for band in range(n):
            band_evals = all_evals[:, band]
            # Check if band is within energy window for any k-point
            if energy_window is None or np.any(np.abs(band_evals) <= energy_window):
                ax.plot(x, band_evals, color="steelblue", linewidth=0.8, alpha=0.9)
                bands_plotted += 1
        
        # Add horizontal line at E=0 (Fermi level reference)
        ax.axhline(y=0, color="black", linewidth=1.0, linestyle="-", alpha=0.3, zorder=1)
        
        # Add vertical lines at high-symmetry points
        for idx in tick_idx:
            ax.axvline(x[idx], color="gray", linewidth=0.5, linestyle="--", alpha=0.5)
        
        ax.set_xticks([x[idx] for idx in tick_idx])
        ax.set_xticklabels(tick_labels, fontsize=11)
        ax.set_ylabel("Energy (t)", fontsize=11)
        ax.set_title(f"Shell-based bands (L={L}, shells s{s0}..s{s0+L-1})", fontsize=12)
        ax.set_xlim(x[0], x[-1])
        
        # Set y-axis limits based on energy window
        if energy_window is not None:
            ax.set_ylim(-energy_window, energy_window)
            print(f"  Plotted {bands_plotted} bands within energy window ±{energy_window:.1f}")
        else:
            print(f"  Plotted all {bands_plotted} bands")
        
        ax.grid(True, alpha=0.2, linestyle=":")
        
        out_path = OUT_SHELL_BANDS.format(L=L, cut=CUT)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Milestone: Topology diagnostics (gauge-invariant)
# ---------------------------------------------------------------------------


def kfrac_to_kcart(f: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Convert fractional reciprocal coordinates f to Cartesian k-space.

    k_cart = f[0]*b[0] + f[1]*b[1] + f[2]*b[2]  where b has rows = b_i.
    """
    f = np.asarray(f, dtype=float)
    b = np.asarray(b, dtype=float)
    return f @ b


def kcart_to_kfrac(k_cart: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Convert Cartesian k-space coordinate to fractional reciprocal coords."""
    k_cart = np.asarray(k_cart, dtype=float)
    b = np.asarray(b, dtype=float)
    return k_cart @ np.linalg.inv(b)


def _eigpairs(hk: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return sorted eigenvalues and column-eigenvectors of Hermitian hk.

    Returns (evals shape (n,), evecs shape (n, n)) with evecs[:,i] the i-th
    eigenvector, sorted by ascending eigenvalue.
    """
    evals, evecs = np.linalg.eigh(hk)
    return evals, evecs


def _determine_fermi_energy(
    evals_samples: np.ndarray,
    n_sites: int,
    ef_mode: str,
    ef_fixed: float,
) -> float:
    """Determine Fermi energy from a sample of eigenvalue arrays.

    evals_samples: shape (n_samples, n_sites).
    """
    if ef_mode == "fixed":
        return ef_fixed
    # half_filling: E_F sits between band n_sites//2 - 1 and n_sites//2
    all_sorted = np.sort(evals_samples.ravel())
    n_total = len(all_sorted)
    idx = n_total // 2
    return 0.5 * (all_sorted[max(idx - 1, 0)] + all_sorted[min(idx, n_total - 1)])


def _select_occupied_evecs(
    evals: np.ndarray,
    evecs: np.ndarray,
    n_occ: int | None,
    ef: float,
    degeneracy_tol: float = 1e-8,
) -> tuple[np.ndarray, int, list[str]]:
    """Select occupied eigenvectors and return them with warnings.

    Returns (U_occ shape (n, n_occ), n_occ_used, warnings list).
    If n_occ is None, select bands below ef (single-band-near-EF fallback is
    handled at the caller level).
    """
    warnings_list: list[str] = []
    n = len(evals)

    if n_occ is not None:
        if n_occ >= n:
            warnings_list.append(f"n_occ={n_occ} >= n_bands={n}, clamping to {n-1}")
            n_occ = n - 1
        # Check gap between n_occ-1 and n_occ bands
        gap = evals[n_occ] - evals[n_occ - 1]
        if abs(gap) < degeneracy_tol:
            warnings_list.append(
                f"Degeneracy at Fermi level: gap={gap:.2e} between bands "
                f"{n_occ-1} and {n_occ}"
            )
        return evecs[:, :n_occ], n_occ, warnings_list

    # n_occ is None: count bands below ef
    n_occ_auto = int(np.sum(evals < ef))
    if n_occ_auto == 0:
        n_occ_auto = 1
        warnings_list.append("No bands below E_F, using n_occ=1")
    if n_occ_auto >= n:
        n_occ_auto = n - 1
        warnings_list.append(f"All bands below E_F, clamping n_occ={n_occ_auto}")
    gap = evals[n_occ_auto] - evals[n_occ_auto - 1]
    if abs(gap) < degeneracy_tol:
        warnings_list.append(
            f"Degeneracy at auto Fermi cut: gap={gap:.2e} between bands "
            f"{n_occ_auto-1} and {n_occ_auto}"
        )
    return evecs[:, :n_occ_auto], n_occ_auto, warnings_list


def _link_variable_multiband(
    U_occ_k: np.ndarray,
    U_occ_kmu: np.ndarray,
) -> tuple[complex, bool]:
    """Compute gauge-invariant link variable between two k-points (multi-band).

    U_mu = det(M) / |det(M)|  where M = U_occ(k)^dag U_occ(k+mu).
    Returns (link, ok) where ok=False if |det(M)| is too small.
    """
    M = U_occ_k.conj().T @ U_occ_kmu
    det_M = np.linalg.det(M)
    abs_det = abs(det_M)
    if abs_det < 1e-12:
        return 1.0 + 0j, False
    return det_M / abs_det, True


def chern_number_slice(
    n_sites: int,
    edges_pbc: list[tuple[int, int, tuple[float, float, float]]],
    t: float,
    onsite: float,
    b: np.ndarray,
    nk: int,
    axis: str,
    slice_value: float,
    n_occ: int | None,
    ef_mode: str,
    ef_fixed: float,
) -> tuple[float, np.ndarray, dict]:
    """Compute Chern number on a 2D BZ slice using Fukui link variables.

    Parameters
    ----------
    n_sites : int
        Number of orbitals.
    edges_pbc : edge list for build_tb_hamiltonian_k.
    t, onsite : TB parameters.
    b : np.ndarray, shape (3,3), reciprocal lattice vectors as rows.
    nk : int
        Grid size per in-plane dimension.
    axis : str
        Normal axis for the slice: "kx", "ky", or "kz".
    slice_value : float
        Fractional coordinate along normal axis (0..1).
    n_occ : int or None
        Number of occupied bands. None = auto from E_F.
    ef_mode, ef_fixed : Fermi energy parameters.

    Returns
    -------
    chern : float
        Computed Chern number (should be near integer).
    flux_grid : np.ndarray, shape (nk, nk)
        Berry flux per plaquette.
    info : dict
        Diagnostic information.
    """
    b = np.asarray(b, dtype=float)
    axis_map = {"kx": 0, "ky": 1, "kz": 2}
    ax_idx = axis_map[axis]
    in_plane = [i for i in range(3) if i != ax_idx]
    ax0, ax1 = in_plane

    # Build fractional coordinate grid; f[ax_idx] = slice_value, others sweep 0..1
    frac_grid = np.zeros((nk, nk, 3), dtype=float)
    f_vals = np.linspace(0, 1, nk, endpoint=False)
    for i0 in range(nk):
        for i1 in range(nk):
            frac_grid[i0, i1, ax0] = f_vals[i0]
            frac_grid[i0, i1, ax1] = f_vals[i1]
            frac_grid[i0, i1, ax_idx] = slice_value

    # Pre-compute all eigenpairs on the grid
    evals_grid = np.empty((nk, nk, n_sites), dtype=float)
    evecs_grid = np.empty((nk, nk, n_sites, n_sites), dtype=complex)
    for i0 in range(nk):
        for i1 in range(nk):
            k_cart = kfrac_to_kcart(frac_grid[i0, i1], b)
            hk = build_tb_hamiltonian_k(n_sites, edges_pbc, k_cart, t, onsite,
                                        reciprocal_b=b)
            evals_grid[i0, i1], evecs_grid[i0, i1] = _eigpairs(hk)

    # Determine Fermi energy from all sampled eigenvalues
    all_evals = evals_grid.reshape(-1, n_sites)
    ef = _determine_fermi_energy(all_evals, n_sites, ef_mode, ef_fixed)
    print(f"  Slice E_F = {ef:.6f} ({ef_mode})")

    # Determine n_occ from a representative point if not given
    if n_occ is None:
        _, n_occ_used, _ = _select_occupied_evecs(
            evals_grid[0, 0], evecs_grid[0, 0], None, ef
        )
    else:
        n_occ_used = n_occ

    # Check for n_occ consistency across the grid
    n_occ_varies = False
    for i0 in range(nk):
        for i1 in range(nk):
            n_below = int(np.sum(evals_grid[i0, i1] < ef))
            if n_below == 0:
                n_below = 1
            if n_below != n_occ_used:
                n_occ_varies = True
                break
        if n_occ_varies:
            break
    if n_occ_varies:
        print(f"  WARNING: number of bands below E_F varies across grid. "
              f"Using fixed n_occ={n_occ_used}.")

    # Check gap at each k-point
    degeneracy_warnings = 0
    for i0 in range(nk):
        for i1 in range(nk):
            ev = evals_grid[i0, i1]
            if n_occ_used < n_sites:
                gap = ev[n_occ_used] - ev[n_occ_used - 1]
                if abs(gap) < 1e-8:
                    degeneracy_warnings += 1
    if degeneracy_warnings > 0:
        print(f"  WARNING: {degeneracy_warnings}/{nk*nk} k-points have degenerate "
              f"gap at n_occ={n_occ_used}. Multi-band formulation used (gauge-invariant).")

    # Compute Berry flux on each plaquette using Fukui method
    flux_grid = np.zeros((nk, nk), dtype=float)
    n_skipped = 0

    for i0 in range(nk):
        for i1 in range(nk):
            # Four corners of plaquette (periodic wrap)
            j0 = (i0 + 1) % nk
            j1 = (i1 + 1) % nk

            U00 = evecs_grid[i0, i1, :, :n_occ_used]
            U10 = evecs_grid[j0, i1, :, :n_occ_used]
            U11 = evecs_grid[j0, j1, :, :n_occ_used]
            U01 = evecs_grid[i0, j1, :, :n_occ_used]

            # Link variables around the plaquette: 00->10->11->01->00
            L1, ok1 = _link_variable_multiband(U00, U10)
            L2, ok2 = _link_variable_multiband(U10, U11)
            L3, ok3 = _link_variable_multiband(U11, U01)
            L4, ok4 = _link_variable_multiband(U01, U00)

            if not (ok1 and ok2 and ok3 and ok4):
                n_skipped += 1
                continue

            # Wilson loop around the plaquette: 00->10->11->01->00.
            # L3 and L4 are REVERSE-direction links (11->01, 01->00) which are
            # already the inverses of the forward links.  The closed-loop product
            # is simply L1*L2*L3*L4 (no conjugation needed).
            plaq = L1 * L2 * L3 * L4
            flux_grid[i0, i1] = np.angle(plaq)

    chern = float(np.sum(flux_grid)) / (2 * np.pi)

    skip_frac = n_skipped / (nk * nk) if nk > 0 else 0.0
    trustworthy = skip_frac <= 0.05

    info = {
        "ef": ef,
        "n_occ_used": n_occ_used,
        "n_skipped": n_skipped,
        "skip_frac": skip_frac,
        "trustworthy": trustworthy,
        "degeneracy_warnings": degeneracy_warnings,
        "n_occ_varies": n_occ_varies,
        "axis": axis,
        "slice_value": slice_value,
        "nk": nk,
    }

    return chern, flux_grid, info


def _save_chern_report(
    chern: float,
    info: dict,
    out_txt: str,
    append: bool = False,
) -> None:
    """Save Chern number results to a text file."""
    os.makedirs(os.path.dirname(out_txt), exist_ok=True)
    mode = "a" if append else "w"
    with open(out_txt, mode, encoding="utf-8") as f:
        f.write(f"Chern number on {info['axis']}={info['slice_value']:.3f} slice\n")
        f.write(f"  Grid: {info['nk']}x{info['nk']}\n")
        f.write(f"  E_F: {info['ef']:.6f}\n")
        f.write(f"  n_occ: {info['n_occ_used']}\n")
        f.write(f"  Chern = {chern:.6f}\n")
        rounded = int(round(chern))
        f.write(f"  Rounded: {rounded}\n")
        f.write(f"  |C - round(C)| = {abs(chern - rounded):.6f}\n")
        near_int = abs(chern - rounded) < CHERN_TOL
        f.write(f"  Near integer (tol={CHERN_TOL}): {near_int}\n")
        f.write(f"  Skipped plaquettes: {info['n_skipped']}/{info['nk']**2}\n")
        if not info["trustworthy"]:
            f.write("  RESULT UNTRUSTWORTHY (>5% plaquettes skipped)\n")
        if info["degeneracy_warnings"] > 0:
            f.write(f"  Degeneracy warnings: {info['degeneracy_warnings']}\n")
        f.write("\n")


def _plot_berry_flux(
    flux_grid: np.ndarray,
    info: dict,
    out_png: str,
) -> None:
    """Save a heatmap of Berry flux per plaquette."""
    fig, ax = plt.subplots(figsize=(6, 5))
    vmax = max(float(np.max(np.abs(flux_grid))), 1e-10)
    im = ax.imshow(
        flux_grid.T, origin="lower", cmap="RdBu_r",
        vmin=-vmax, vmax=vmax, aspect="equal",
        extent=[0, 1, 0, 1],
    )
    ax.set_xlabel(f"f (in-plane dim 1)")
    ax.set_ylabel(f"f (in-plane dim 2)")
    ax.set_title(
        f"Berry flux ({info['axis']}={info['slice_value']:.2f}, "
        f"n_occ={info['n_occ_used']})"
    )
    plt.colorbar(im, ax=ax, label="F (rad)")
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def weyl_candidate_scan(
    n_sites: int,
    edges_pbc: list[tuple[int, int, tuple[float, float, float]]],
    t: float,
    onsite: float,
    b: np.ndarray,
    nk3: int,
    gap_bands,
    gap_ef: float,
    gap_thresh: float,
    max_candidates: int,
) -> tuple[list[dict], float]:
    """Coarse 3D scan for Weyl node candidates (small direct gap).

    Returns (candidates, min_gap_global) where candidates is a list of
    dicts sorted by ascending gap, and min_gap_global is the smallest
    direct gap observed anywhere on the grid.
    """
    b = np.asarray(b, dtype=float)
    f_vals = np.linspace(0, 1, nk3, endpoint=False)
    candidates: list[dict] = []
    min_gap_global = float("inf")

    for i0 in range(nk3):
        for i1 in range(nk3):
            for i2 in range(nk3):
                f = np.array([f_vals[i0], f_vals[i1], f_vals[i2]])
                k_cart = kfrac_to_kcart(f, b)
                hk = build_tb_hamiltonian_k(n_sites, edges_pbc, k_cart, t, onsite,
                                            reciprocal_b=b)
                evals = np.linalg.eigvalsh(hk)

                if gap_bands == "near_EF":
                    below = evals[evals <= gap_ef]
                    above = evals[evals > gap_ef]
                    if len(below) == 0 or len(above) == 0:
                        continue
                    e_below = below[-1]
                    e_above = above[0]
                    gap = float(e_above - e_below)
                    band_lo = int(np.where(evals == e_below)[0][-1])
                    band_hi = band_lo + 1
                else:
                    band_lo, band_hi = gap_bands
                    if band_hi >= n_sites:
                        continue
                    gap = float(evals[band_hi] - evals[band_lo])

                if gap < min_gap_global:
                    min_gap_global = gap

                if gap < gap_thresh:
                    candidates.append({
                        "k_frac": f.copy(),
                        "k_cart": k_cart.copy(),
                        "gap": gap,
                        "band_lo": band_lo,
                        "band_hi": band_hi,
                    })

    candidates.sort(key=lambda c: c["gap"])
    if len(candidates) > max_candidates:
        candidates = candidates[:max_candidates]

    return candidates, min_gap_global


def weyl_sphere_charge(
    n_sites: int,
    edges_pbc: list[tuple[int, int, tuple[float, float, float]]],
    t: float,
    onsite: float,
    b: np.ndarray,
    k0_cart: np.ndarray,
    band_lo: int,
    r_cart: float,
    n_theta: int,
    n_phi: int,
) -> tuple[float, dict]:
    """Compute topological charge on a small sphere around k0 using Fukui method.

    Uses a theta-phi grid on a sphere of radius r_cart centered at k0_cart.
    Computes Chern number of the lowest (band_lo+1) bands on the sphere surface.

    Returns (charge, info_dict).
    """
    k0_cart = np.asarray(k0_cart, dtype=float)
    n_occ = band_lo + 1

    # Build theta-phi grid (theta in [0,pi], phi in [0,2pi))
    theta_vals = np.linspace(0, np.pi, n_theta)
    phi_vals = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)

    # Pre-compute eigenvectors on the sphere
    evals_sphere = np.empty((n_theta, n_phi, n_sites), dtype=float)
    evecs_sphere = np.empty((n_theta, n_phi, n_sites, n_sites), dtype=complex)

    for it in range(n_theta):
        for ip in range(n_phi):
            th = theta_vals[it]
            ph = phi_vals[ip]
            dk = r_cart * np.array([
                np.sin(th) * np.cos(ph),
                np.sin(th) * np.sin(ph),
                np.cos(th),
            ])
            k_cart = k0_cart + dk
            hk = build_tb_hamiltonian_k(n_sites, edges_pbc, k_cart, t, onsite,
                                        reciprocal_b=b)
            evals_sphere[it, ip], evecs_sphere[it, ip] = _eigpairs(hk)

    # Check gap consistency
    degeneracy_count = 0
    for it in range(n_theta):
        for ip in range(n_phi):
            ev = evals_sphere[it, ip]
            if n_occ < n_sites:
                if abs(ev[n_occ] - ev[n_occ - 1]) < 1e-8:
                    degeneracy_count += 1

    # Compute Berry flux on each plaquette of the theta-phi grid
    # theta direction: index it, phi direction: index ip
    # Plaquette (it,ip) -> (it+1,ip) -> (it+1,ip+1) -> (it,ip+1) -> (it,ip)
    # Periodic in phi, NOT periodic in theta (poles)
    flux_total = 0.0
    n_skipped = 0
    n_plaq = 0

    for it in range(n_theta - 1):
        for ip in range(n_phi):
            jp = (ip + 1) % n_phi
            jt = it + 1

            U00 = evecs_sphere[it, ip, :, :n_occ]
            U10 = evecs_sphere[jt, ip, :, :n_occ]
            U11 = evecs_sphere[jt, jp, :, :n_occ]
            U01 = evecs_sphere[it, jp, :, :n_occ]

            L1, ok1 = _link_variable_multiband(U00, U10)
            L2, ok2 = _link_variable_multiband(U10, U11)
            L3, ok3 = _link_variable_multiband(U11, U01)
            L4, ok4 = _link_variable_multiband(U01, U00)

            n_plaq += 1
            if not (ok1 and ok2 and ok3 and ok4):
                n_skipped += 1
                continue

            plaq = L1 * L2 * L3 * L4
            flux_total += np.angle(plaq)

    charge = flux_total / (2 * np.pi)
    skip_frac = n_skipped / max(n_plaq, 1)

    info = {
        "n_occ": n_occ,
        "r_cart": r_cart,
        "n_theta": n_theta,
        "n_phi": n_phi,
        "n_skipped": n_skipped,
        "n_plaq": n_plaq,
        "skip_frac": skip_frac,
        "trustworthy": skip_frac <= 0.05,
        "degeneracy_count": degeneracy_count,
    }

    return charge, info


# =====================================================================
#  Robust Weyl-node detection pipeline (added alongside existing scan)
# =====================================================================

def weyl_min_gap_window_scan(
    n_sites: int,
    edges_pbc: list[tuple[int, int, tuple[float, float, float]]],
    t: float,
    onsite: float,
    b: np.ndarray,
    nk3: int,
    band_window: int,
    max_keep: int,
    include_extra_hops: bool = True,
    energy_window: float | None = None,
    ef_reference: float | None = None,
) -> tuple[list[dict], float, dict]:
    """Full-spectrum min-gap 3D scan over all adjacent band pairs.

    For each k on an nk3^3 grid, computes the minimum direct gap g_m(k)
    over ALL adjacent band pairs m in [0, n_sites-2].  The band_window
    parameter is accepted for backwards compatibility but ignored.

    If energy_window is not None, candidates are filtered to those where
    the midpoint energy 0.5*(E_m + E_{m+1}) lies within energy_window of
    ef_reference.

    Returns (topk_list, global_min_gap, argmin_info).
    """
    b = np.asarray(b, dtype=float)
    print(f"    Scanning all adjacent band pairs (0 to {n_sites - 2})")
    if energy_window is not None and ef_reference is not None:
        print(f"    Applying energy filter: |E_mid - EF| < {energy_window:.4f}"
              f"  (EF={ef_reference:.6f})")

    f_vals = np.linspace(0, 1, nk3, endpoint=False)
    all_results: list[dict] = []
    global_min_gap = float("inf")
    argmin_info: dict = {}

    for i0 in range(nk3):
        for i1 in range(nk3):
            for i2 in range(nk3):
                f = np.array([f_vals[i0], f_vals[i1], f_vals[i2]])
                k_cart = kfrac_to_kcart(f, b)
                hk = build_tb_hamiltonian_k(
                    n_sites, edges_pbc, k_cart, t, onsite,
                    include_extra_hops=include_extra_hops,
                    reciprocal_b=b,
                )
                evals = np.linalg.eigvalsh(hk)

                best_gap = float("inf")
                best_m = n_sites // 2
                for m in range(0, n_sites - 1):
                    g = float(evals[m + 1] - evals[m])
                    if energy_window is not None and ef_reference is not None:
                        mid_e = 0.5 * (evals[m] + evals[m + 1])
                        if abs(mid_e - ef_reference) > energy_window:
                            continue
                    if g < best_gap:
                        best_gap = g
                        best_m = m

                if best_gap == float("inf"):
                    continue

                entry = {
                    "k_frac": f.copy(),
                    "k_cart": k_cart.copy(),
                    "gap": best_gap,
                    "band_lo": best_m,
                    "band_hi": best_m + 1,
                }
                all_results.append(entry)

                if best_gap < global_min_gap:
                    global_min_gap = best_gap
                    argmin_info = {
                        "k_frac": f.copy(),
                        "k_cart": k_cart.copy(),
                        "gap": best_gap,
                        "band_lo": best_m,
                        "band_hi": best_m + 1,
                    }

    all_results.sort(key=lambda c: c["gap"])
    topk = all_results[:max_keep]
    return topk, global_min_gap, argmin_info


def refine_candidates_local(
    n_sites: int,
    edges_pbc: list[tuple[int, int, tuple[float, float, float]]],
    t: float,
    onsite: float,
    b: np.ndarray,
    coarse_candidates: list[dict],
    delta_frac: float,
    n_local: int,
    band_window: int,
    n_seeds: int,
    energy_window: float | None = None,
    ef_reference: float | None = None,
) -> list[dict]:
    """Adaptive local refinement around the best coarse candidates.

    For each seed, samples an n_local^3 sub-grid in fractional coordinates
    centered on the seed's k_frac, then picks the point with the smallest
    gap over ALL adjacent band pairs.  The band_window parameter is accepted
    for backwards compatibility but ignored.

    If energy_window is set, only band pairs whose midpoint energy is within
    energy_window of ef_reference are considered.

    Returns a list of refined dicts with parent info, sorted by gap.
    """
    b = np.asarray(b, dtype=float)

    refined: list[dict] = []
    seeds = coarse_candidates[:n_seeds]

    for si, seed in enumerate(seeds):
        k0 = seed["k_frac"]
        offsets = np.linspace(-delta_frac, delta_frac, n_local)

        best_gap = float("inf")
        best_entry: dict = {}

        for d0 in offsets:
            for d1 in offsets:
                for d2 in offsets:
                    f = (k0 + np.array([d0, d1, d2])) % 1.0
                    k_cart = kfrac_to_kcart(f, b)
                    hk = build_tb_hamiltonian_k(
                        n_sites, edges_pbc, k_cart, t, onsite,
                        reciprocal_b=b,
                    )
                    evals = np.linalg.eigvalsh(hk)

                    local_best = float("inf")
                    local_m = n_sites // 2
                    for m in range(0, n_sites - 1):
                        g = float(evals[m + 1] - evals[m])
                        if energy_window is not None and ef_reference is not None:
                            mid_e = 0.5 * (evals[m] + evals[m + 1])
                            if abs(mid_e - ef_reference) > energy_window:
                                continue
                        if g < local_best:
                            local_best = g
                            local_m = m

                    if local_best < best_gap:
                        best_gap = local_best
                        best_entry = {
                            "seed_idx": si,
                            "seed_k_frac": k0.copy(),
                            "seed_gap": seed["gap"],
                            "k_frac": f.copy(),
                            "k_cart": k_cart.copy(),
                            "gap": local_best,
                            "band_lo": local_m,
                            "band_hi": local_m + 1,
                        }

        if best_entry:
            refined.append(best_entry)

    refined.sort(key=lambda c: c["gap"])
    return refined


def sphere_charge_radius_sweep(
    n_sites: int,
    edges_pbc: list[tuple[int, int, tuple[float, float, float]]],
    t: float,
    onsite: float,
    b: np.ndarray,
    candidate: dict,
    radii_fracs: list[float],
    n_theta_list: list[int],
) -> list[dict]:
    """Sphere charge validation with radius and angular-grid convergence sweep.

    For each radius fraction and each angular resolution, computes the sphere
    charge.  Marks a measurement as 'stable' when the charge is near-integer
    and consistent across angular resolutions.

    Returns a list of result dicts (one per radius), each containing sub-results
    per angular resolution.
    """
    avg_b_norm = float(np.mean(np.linalg.norm(b, axis=1)))
    results: list[dict] = []

    for rf in radii_fracs:
        r_cart = rf * avg_b_norm
        sub: list[dict] = []

        for nt in n_theta_list:
            np_ = 2 * nt
            charge, info = weyl_sphere_charge(
                n_sites, edges_pbc, t, onsite, b,
                candidate["k_cart"], candidate["band_lo"],
                r_cart, nt, np_,
            )
            sub.append({
                "n_theta": nt,
                "n_phi": np_,
                "charge": charge,
                "rounded": int(round(charge)),
                "skip_frac": info["skip_frac"],
                "n_skipped": info["n_skipped"],
                "n_plaq": info["n_plaq"],
            })

        delta_ang = abs(sub[-1]["charge"] - sub[0]["charge"]) if len(sub) > 1 else 0.0
        best_charge = sub[-1]["charge"]
        stable = (
            abs(best_charge - round(best_charge)) < 0.2
            and delta_ang < 0.15
            and all(s["skip_frac"] <= 0.05 for s in sub)
        )

        results.append({
            "radius_frac": rf,
            "r_cart": r_cart,
            "sub": sub,
            "delta_ang": delta_ang,
            "stable": stable,
        })

    return results


def _toy_weyl_hamiltonian(kx: float, ky: float, kz: float, k0: float) -> np.ndarray:
    """2-band toy Weyl Hamiltonian with nodes at (0,0,+/-k0).

    H(k) = sin(kx)*sx + sin(ky)*sy + (cos(kz)-cos(k0)+cos(kx)+cos(ky)-2)*sz
    """
    dx = np.sin(kx)
    dy = np.sin(ky)
    dz = np.cos(kz) - np.cos(k0) + np.cos(kx) + np.cos(ky) - 2.0
    return np.array([
        [dz, dx - 1j * dy],
        [dx + 1j * dy, -dz],
    ], dtype=complex)


def toy_weyl_detector(out_txt: str) -> bool:
    """Self-contained positive control: detect Weyl nodes in a 2-band toy model.

    Uses a coarse grid over [-pi,pi]^3, iterative local refinement (3 rounds),
    and sphere-charge Fukui computation.  The model has exact Weyl nodes at
    (0,0,+/-pi/2) so the detector should find charge = +/-1.

    Returns True if at least one node found with |charge - round(charge)| < 0.3
    and |round(charge)| == 1.
    """
    k0 = np.pi / 2
    nk_toy = 31

    f_vals = np.linspace(-np.pi, np.pi, nk_toy, endpoint=False)

    all_pts: list[dict] = []
    for i0 in range(nk_toy):
        for i1 in range(nk_toy):
            for i2 in range(nk_toy):
                kx, ky, kz = f_vals[i0], f_vals[i1], f_vals[i2]
                hk = _toy_weyl_hamiltonian(kx, ky, kz, k0)
                evals = np.linalg.eigvalsh(hk)
                gap = float(evals[1] - evals[0])
                all_pts.append({"k": np.array([kx, ky, kz]), "gap": gap})

    all_pts.sort(key=lambda c: c["gap"])
    coarse_min = all_pts[0]["gap"]

    # Iterative refinement: 4 rounds, shrinking search cube each time.
    # Each round scans a fixed grid around the current center, then updates.
    ref_k = all_pts[0]["k"].copy()
    ref_gap = coarse_min
    for _round in range(4):
        delta = (2 * np.pi / nk_toy) / (2 ** _round)
        n_ref = 15
        offsets = np.linspace(-delta, delta, n_ref)
        center = ref_k.copy()
        for d0 in offsets:
            for d1 in offsets:
                for d2 in offsets:
                    k = center + np.array([d0, d1, d2])
                    hk = _toy_weyl_hamiltonian(k[0], k[1], k[2], k0)
                    evals = np.linalg.eigvalsh(hk)
                    g = float(evals[1] - evals[0])
                    if g < ref_gap:
                        ref_gap = g
                        ref_k = k.copy()

    # Sphere charge with small radius (node gap should be ~0 after refinement)
    r_sphere = 0.2
    n_theta_t = 25
    n_phi_t = 50
    theta_vals = np.linspace(0, np.pi, n_theta_t)
    phi_vals = np.linspace(0, 2 * np.pi, n_phi_t, endpoint=False)

    evecs_sph = np.empty((n_theta_t, n_phi_t, 2, 2), dtype=complex)
    for it in range(n_theta_t):
        for ip in range(n_phi_t):
            dk = r_sphere * np.array([
                np.sin(theta_vals[it]) * np.cos(phi_vals[ip]),
                np.sin(theta_vals[it]) * np.sin(phi_vals[ip]),
                np.cos(theta_vals[it]),
            ])
            kp = ref_k + dk
            hk = _toy_weyl_hamiltonian(kp[0], kp[1], kp[2], k0)
            _, vecs = np.linalg.eigh(hk)
            evecs_sph[it, ip] = vecs

    flux = 0.0
    n_plaq = 0
    n_skip = 0
    for it in range(n_theta_t - 1):
        for ip in range(n_phi_t):
            jp = (ip + 1) % n_phi_t
            u00 = evecs_sph[it, ip, :, :1]
            u10 = evecs_sph[it + 1, ip, :, :1]
            u11 = evecs_sph[it + 1, jp, :, :1]
            u01 = evecs_sph[it, jp, :, :1]

            L1, ok1 = _link_variable_multiband(u00, u10)
            L2, ok2 = _link_variable_multiband(u10, u11)
            L3, ok3 = _link_variable_multiband(u11, u01)
            L4, ok4 = _link_variable_multiband(u01, u00)
            n_plaq += 1
            if not (ok1 and ok2 and ok3 and ok4):
                n_skip += 1
                continue
            flux += np.angle(L1 * L2 * L3 * L4)

    charge = flux / (2 * np.pi)
    rounded = int(round(charge))
    passed = abs(charge - rounded) < 0.3 and abs(rounded) == 1

    lines: list[str] = [
        "Toy Weyl Model Detector Check",
        "=" * 40,
        f"Model: H(k)= sin(kx)*sx + sin(ky)*sy + (cos(kz)-cos(k0)+cos(kx)+cos(ky)-2)*sz",
        f"Expected nodes at (0,0,+/-k0), k0 = pi/2 = {k0:.4f}",
        f"",
        f"Coarse scan ({nk_toy}^3): global min gap = {coarse_min:.6e}",
        f"  Top seed: k = ({all_pts[0]['k'][0]:.4f}, {all_pts[0]['k'][1]:.4f}, {all_pts[0]['k'][2]:.4f})",
        f"Refined (3 rounds): min gap = {ref_gap:.6e}",
        f"  k_refined = ({ref_k[0]:.6f}, {ref_k[1]:.6f}, {ref_k[2]:.6f})",
        f"  expected  = (0, 0, +/-{k0:.6f})",
        f"",
        f"Sphere charge (r={r_sphere}, n_theta={n_theta_t}): {charge:.4f} (rounded: {rounded})",
        f"  skipped plaquettes: {n_skip}/{n_plaq}",
        f"RESULT: {'PASS' if passed else 'FAIL'} (detector {'found' if passed else 'did not find'} Weyl node with charge +/-1)",
    ]
    os.makedirs(os.path.dirname(out_txt), exist_ok=True)
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    for line in lines:
        print(f"  [toy] {line}")
    return passed


def topology_main(atoms: Atoms) -> None:
    """Run topology diagnostics: slice Chern and/or Weyl scan (legacy + robust pipeline)."""
    n_sites = len(atoms)
    edges_pbc = build_edge_list_pbc(atoms, CUT)
    cell = atoms.get_cell().array
    b = reciprocal_lattice(cell)

    print("\n" + "=" * 60)
    print("Topology diagnostics (gauge-invariant Fukui method)")
    print("=" * 60)

    run_slice = TOPO_MODE in ("slice", "both")
    run_weyl = TOPO_MODE in ("weyl", "both")

    # --- Slice Chern (unchanged) ---
    if run_slice:
        axis_map = {"kx": 0, "ky": 1, "kz": 2}
        ax_idx = axis_map.get(SLICE_AXIS, 2)
        ax_name = SLICE_AXIS

        for sv in [0.0, 0.5]:
            print(f"\n--- Chern number: {ax_name}={sv:.1f} slice, "
                  f"grid={N_K}x{N_K} ---")
            chern, flux, info = chern_number_slice(
                n_sites, edges_pbc, T_HOP, ONSITE, b,
                N_K, ax_name, sv, N_OCC, EF_MODE, EF_FIXED,
            )
            rounded = int(round(chern))
            near_int = abs(chern - rounded) < CHERN_TOL
            print(f"  Chern({ax_name}={sv:.1f}) = {chern:.6f} "
                  f"(rounded: {rounded}, near-integer: {near_int})")
            if info["n_skipped"] > 0:
                print(f"  Skipped plaquettes: {info['n_skipped']}/{N_K**2}")
            if not info["trustworthy"]:
                print("  RESULT UNTRUSTWORTHY (>5% plaquettes skipped)")

            suffix = f"_{ax_name}{sv:.1f}"
            out_png = OUT_BERRY_PNG.replace(".png", f"{suffix}.png")
            _plot_berry_flux(flux, info, out_png)
            print(f"  Saved flux plot: {out_png}")

            _save_chern_report(chern, info, OUT_CHERN_TXT, append=(sv != 0.0))

        print(f"  Saved Chern report: {OUT_CHERN_TXT}")

    # --- Weyl candidate scan (legacy near_EF kept, then robust pipeline) ---
    if run_weyl:
        # ---- Legacy near_EF scan (kept for comparison) ----
        print(f"\n--- [Legacy] Weyl candidate scan: {N_K3}^3 grid, "
              f"gap_thresh={GAP_THRESH:.1e} ---")
        if GAP_BANDS == "near_EF":
            hk_gamma = build_tb_hamiltonian_k(
                n_sites, edges_pbc, np.zeros(3), T_HOP, ONSITE,
                reciprocal_b=b,
            )
            evals_gamma = np.linalg.eigvalsh(hk_gamma)
            if EF_MODE == "half_filling":
                n_half = n_sites // 2
                ef_weyl = 0.5 * (evals_gamma[max(n_half - 1, 0)]
                                 + evals_gamma[min(n_half, n_sites - 1)])
            else:
                ef_weyl = EF_FIXED
            gap_bands_resolved = "near_EF"
            print(f"  Using near_EF mode with E_F = {ef_weyl:.6f}")
        else:
            gap_bands_resolved = GAP_BANDS
            ef_weyl = GAP_EF

        candidates_legacy, min_gap_legacy = weyl_candidate_scan(
            n_sites, edges_pbc, T_HOP, ONSITE, b,
            N_K3, gap_bands_resolved,
            ef_weyl, GAP_THRESH, MAX_CANDIDATES,
        )
        print(f"  Found {len(candidates_legacy)} candidates (gap < {GAP_THRESH:.1e})")
        if min_gap_legacy < float("inf"):
            print(f"  Global min gap on grid = {min_gap_legacy:.6e}")
        else:
            print("  Global min gap could not be computed (no valid k-points)")

        os.makedirs(os.path.dirname(OUT_WEYL_TXT), exist_ok=True)
        with open(OUT_WEYL_TXT, "w", encoding="utf-8") as f:
            f.write(f"[Legacy] Weyl candidate scan: {N_K3}^3 grid, "
                    f"gap_thresh={GAP_THRESH:.1e}\n")
            f.write(f"Found {len(candidates_legacy)} candidates\n")
            f.write(f"Global min gap = {min_gap_legacy:.6e}\n\n")
            for ci, c in enumerate(candidates_legacy):
                f.write(f"Candidate {ci}:\n")
                f.write(f"  k_frac = ({c['k_frac'][0]:.6f}, "
                        f"{c['k_frac'][1]:.6f}, {c['k_frac'][2]:.6f})\n")
                f.write(f"  gap = {c['gap']:.6e}, bands = ({c['band_lo']}, {c['band_hi']})\n\n")
        print(f"  Saved: {OUT_WEYL_TXT}")

        # ============================================================
        # Robust multi-stage Weyl pipeline
        # ============================================================
        print("\n" + "-" * 60)
        print("Robust Weyl detection pipeline")
        print("-" * 60)

        report_lines: list[str] = ["Robust Weyl Scan Summary", "=" * 50, ""]

        # Stage 1: Resolution sweep with min-gap-window scan
        best_topk: list[dict] = []
        best_nk3 = N_K3_LIST[0]

        for nk3 in N_K3_LIST:
            print(f"\n  [Stage 1] min_gap_window scan, nk3={nk3}")
            topk, g_min, g_info = weyl_min_gap_window_scan(
                n_sites, edges_pbc, T_HOP, ONSITE, b,
                nk3, WEYL_BAND_WINDOW, WEYL_TOPK,
                energy_window=WEYL_ENERGY_WINDOW,
                ef_reference=ef_weyl if WEYL_ENERGY_WINDOW is not None else None,
            )

            header = (f"nk3={nk3}: global_min_gap={g_min:.6e}")
            if g_info:
                header += (f", k_frac=({g_info['k_frac'][0]:.4f}, "
                           f"{g_info['k_frac'][1]:.4f}, {g_info['k_frac'][2]:.4f}), "
                           f"bands=({g_info['band_lo']},{g_info['band_hi']})")
            print(f"    {header}")
            report_lines.append(header)

            n_pass = sum(1 for c in topk if c["gap"] < WEYL_THRESH_INIT)
            print(f"    passes_thresh({WEYL_THRESH_INIT:.1e}): {n_pass}/{len(topk)}")
            report_lines.append(f"  passes_thresh({WEYL_THRESH_INIT:.1e}): {n_pass}/{len(topk)}")

            top10 = topk[:10]
            report_lines.append(f"  Top-10 candidates:")
            for ci, c in enumerate(top10):
                line = (f"    #{ci}: gap={c['gap']:.6e}, "
                        f"k_frac=({c['k_frac'][0]:.4f},{c['k_frac'][1]:.4f},{c['k_frac'][2]:.4f}), "
                        f"bands=({c['band_lo']},{c['band_hi']}), "
                        f"pass={c['gap'] < WEYL_THRESH_INIT}")
                if ci < 5:
                    print(f"    {line}")
                report_lines.append(line)
            report_lines.append("")

            if not best_topk or (g_min < best_topk[0]["gap"] if best_topk else True):
                best_topk = topk
                best_nk3 = nk3

        # Also run legacy near_EF at each resolution for comparison
        report_lines.append("Legacy near_EF comparison:")
        for nk3 in N_K3_LIST:
            _, mg_legacy = weyl_candidate_scan(
                n_sites, edges_pbc, T_HOP, ONSITE, b,
                nk3, gap_bands_resolved, ef_weyl, GAP_THRESH, MAX_CANDIDATES,
            )
            report_lines.append(f"  nk3={nk3}: near_EF global_min_gap={mg_legacy:.6e}")
        report_lines.append("")

        # Stage 1b: Negative control (NN-only, no extra hops)
        if WEYL_NEG_CONTROL and EXTRA_HOPS:
            print(f"\n  [Neg control] NN-only scan, nk3=21")
            topk_nn, g_min_nn, g_info_nn = weyl_min_gap_window_scan(
                n_sites, edges_pbc, T_HOP, ONSITE, b,
                21, WEYL_BAND_WINDOW, 5,
                include_extra_hops=False,
                energy_window=WEYL_ENERGY_WINDOW,
                ef_reference=ef_weyl if WEYL_ENERGY_WINDOW is not None else None,
            )
            nc_line = f"NN-only (neg control): global_min_gap={g_min_nn:.6e}"
            if g_info_nn:
                nc_line += (f", k_frac=({g_info_nn['k_frac'][0]:.4f},"
                            f"{g_info_nn['k_frac'][1]:.4f},{g_info_nn['k_frac'][2]:.4f})")
            print(f"    {nc_line}")
            report_lines.append(f"Negative control (NN-only, nk3=21):")
            report_lines.append(f"  {nc_line}")
            report_lines.append("")

        # Stage 2: Adaptive local refinement
        delta_frac = WEYL_REFINE_DELTA if WEYL_REFINE_DELTA is not None else 0.5 / best_nk3
        print(f"\n  [Stage 2] Local refinement: {WEYL_REFINE_SEEDS} seeds, "
              f"delta={delta_frac:.4f}, grid={WEYL_REFINE_LOCAL_N}^3")

        refined = refine_candidates_local(
            n_sites, edges_pbc, T_HOP, ONSITE, b,
            best_topk, delta_frac, WEYL_REFINE_LOCAL_N,
            WEYL_BAND_WINDOW, WEYL_REFINE_SEEDS,
            energy_window=WEYL_ENERGY_WINDOW,
            ef_reference=ef_weyl if WEYL_ENERGY_WINDOW is not None else None,
        )

        refined_lines: list[str] = ["Refined Weyl Candidates", "=" * 50, ""]
        for ri, r in enumerate(refined):
            imp = r["seed_gap"] - r["gap"]
            print(f"    Seed {r['seed_idx']}: gap {r['seed_gap']:.6e} -> {r['gap']:.6e} "
                  f"(improvement {imp:.2e})")
            print(f"      k_frac: ({r['seed_k_frac'][0]:.4f},{r['seed_k_frac'][1]:.4f},"
                  f"{r['seed_k_frac'][2]:.4f}) -> ({r['k_frac'][0]:.4f},{r['k_frac'][1]:.4f},"
                  f"{r['k_frac'][2]:.4f})")
            print(f"      bands=({r['band_lo']},{r['band_hi']})")
            refined_lines.append(f"Refined #{ri}:")
            refined_lines.append(f"  seed_idx={r['seed_idx']}, seed_gap={r['seed_gap']:.6e}, "
                                 f"seed_k_frac=({r['seed_k_frac'][0]:.6f},{r['seed_k_frac'][1]:.6f},"
                                 f"{r['seed_k_frac'][2]:.6f})")
            refined_lines.append(f"  refined_gap={r['gap']:.6e}, "
                                 f"k_frac=({r['k_frac'][0]:.6f},{r['k_frac'][1]:.6f},{r['k_frac'][2]:.6f})")
            refined_lines.append(f"  k_cart=({r['k_cart'][0]:.6f},{r['k_cart'][1]:.6f},{r['k_cart'][2]:.6f})")
            refined_lines.append(f"  bands=({r['band_lo']},{r['band_hi']})")
            refined_lines.append("")

        # Stage 3: Sphere charge with radius sweep on top refined candidates
        _sv_in = input(
            f"Number of refined candidates to validate? "
            f"(Press Enter for default = {WEYL_SPHERE_TOP}): "
        ).strip()
        if _sv_in == "":
            n_sphere = min(WEYL_SPHERE_TOP, len(refined))
        else:
            try:
                n_sphere = min(int(_sv_in), len(refined))
            except ValueError:
                print("Invalid input. Using default.")
                n_sphere = min(WEYL_SPHERE_TOP, len(refined))
        if n_sphere > 0:
            print(f"\n  [Stage 3] Sphere charge radius sweep for top {n_sphere} refined candidates")
            refined_lines.append("Sphere Charge Validation")
            refined_lines.append("-" * 40)

            for ri in range(n_sphere):
                cand = refined[ri]
                print(f"\n    Candidate {ri}: gap={cand['gap']:.6e}, "
                      f"k_frac=({cand['k_frac'][0]:.4f},{cand['k_frac'][1]:.4f},"
                      f"{cand['k_frac'][2]:.4f})")

                sweep = sphere_charge_radius_sweep(
                    n_sites, edges_pbc, T_HOP, ONSITE, b,
                    cand, WEYL_SPHERE_RADII, WEYL_SPHERE_NTHETA_LIST,
                )

                refined_lines.append(f"\nCandidate {ri}: gap={cand['gap']:.6e}")
                print(f"    {'r_frac':>8s}  {'r_cart':>8s}  ", end="")
                for nt in WEYL_SPHERE_NTHETA_LIST:
                    print(f"{'Q(nt='+str(nt)+')':>12s}  ", end="")
                print(f"{'delta':>8s}  {'stable':>6s}")
                refined_lines.append(
                    f"  {'r_frac':>8s}  {'r_cart':>8s}  " +
                    "  ".join(f"{'Q(nt='+str(nt)+')':>12s}" for nt in WEYL_SPHERE_NTHETA_LIST) +
                    f"  {'delta':>8s}  {'stable':>6s}"
                )

                for sr in sweep:
                    charges_str = "  ".join(
                        f"{s['charge']:>12.4f}" for s in sr["sub"]
                    )
                    line = (f"    {sr['radius_frac']:>8.4f}  {sr['r_cart']:>8.4f}  "
                            f"{charges_str}  {sr['delta_ang']:>8.4f}  "
                            f"{'Y' if sr['stable'] else 'N':>6s}")
                    print(line)
                    refined_lines.append(f"  {line.strip()}")

                any_stable = any(sr["stable"] for sr in sweep)
                tag = "STABLE at some radii" if any_stable else "NOT STABLE at any radius"
                print(f"    -> {tag}")
                refined_lines.append(f"  -> {tag}")
                refined_lines.append("")

        os.makedirs(DATA_DIR, exist_ok=True)
        with open(WEYL_REPORT, "w", encoding="utf-8") as f:
            f.write("\n".join(report_lines) + "\n")
        print(f"\n  Saved scan summary: {WEYL_REPORT}")

        with open(WEYL_REFINED, "w", encoding="utf-8") as f:
            f.write("\n".join(refined_lines) + "\n")
        print(f"  Saved refined candidates: {WEYL_REFINED}")

        # Stage 4: Toy Weyl positive control
        print(f"\n  [Positive control] Toy 2-band Weyl model")
        toy_pass = toy_weyl_detector(os.path.join(DATA_DIR, "toy_weyl_detector_check.txt"))
        report_lines.append(f"Toy Weyl detector: {'PASS' if toy_pass else 'FAIL'}")
        with open(WEYL_REPORT, "a", encoding="utf-8") as f:
            f.write(f"\nToy Weyl detector: {'PASS' if toy_pass else 'FAIL'}\n")

    print("\n" + "=" * 60)
    print("Topology diagnostics complete.")
    print("=" * 60)


def build_model(
    model_choice: str,
    atoms: Atoms,
    manual_hop_pair: tuple[int, int] | None = None,
) -> list[tuple]:
    """Set up the coupling model and return the model edge list.

    model_choice:
        "nn"            - geometric NN edges only (EXTRA_HOPS cleared)
        "nn+extra"      - geometric NN edges plus manual cross-sheet hoppings
        "nn+2nn"        - NN + 2nd-nearest-neighbour (all edges, both shells)
        "nn+2nn-bdry"   - NN (all) + 2NN boundary-only (shift_ints != (0,0,0))
        "nn+4nn-bdry"   - NN (all) + 4NN boundary-only (shift_ints != (0,0,0))
        "nn+all4"       - shells 0-3 with geometric amplitude decay T_HOP*alpha^shell
        "nn+manualpair" - NN + all pairs whose min-image distance equals seed hop

    manual_hop_pair : (i0, j0) required when model_choice == "nn+manualpair".

    Returns
    -------
    model_edges : list of 5-tuples (i, j, dr_cart, shift_ints, amp)
        Passed to build_tb_hamiltonian_k and _setup_geometry.
    """
    global EXTRA_HOPS
    EXTRA_HOPS = []

    if model_choice == "nn+2nn":
        model_edges = build_shell_model_edges(atoms, [0, 1])
    elif model_choice == "nn+2nn-bdry":
        model_edges = build_boundary_model_edges(
            atoms, target_shell=1, amp_boundary=T_HOP_2NN_BDRY)
    elif model_choice == "nn+4nn-bdry":
        model_edges = build_boundary_model_edges(
            atoms, target_shell=3, amp_boundary=T_HOP_4NN_BDRY)
    elif model_choice == "nn+all4":
        model_edges = build_decay_model_edges(atoms, n_shells=4, alpha=T_HOP_ALPHA)
    elif model_choice == "nn+manualpair":
        if manual_hop_pair is None:
            raise ValueError("nn+manualpair requires manual_hop_pair=(i0,j0).")
        model_edges = build_manual_pair_model_edges(atoms, manual_hop_pair[0], manual_hop_pair[1])
    else:
        edges_4 = build_edge_list_pbc(atoms, CUT)
        model_edges = [(i, j, dr, sh, T_HOP) for (i, j, dr, sh) in edges_4]
        if model_choice == "nn+extra":
            for (i_hop, j_hop) in EXTRA_HOPS_PAIRS:
                add_manual_hop(atoms, i_hop, j_hop, amplitude=EXTRA_HOPS_AMP, S=None)

    tag = f"{len(model_edges)} model edges"
    if EXTRA_HOPS:
        tag += f" + {len(EXTRA_HOPS)} extra hops"
    print(f"Model [{model_choice}]: {tag}")
    return model_edges


def load_refined_candidates(path: str) -> list[dict]:
    """Parse WEYL_REFINED text file and return candidate dicts.

    Each candidate dict has: k_frac, k_cart, gap, band_lo, band_hi.
    """
    candidates: list[dict] = []
    if not os.path.isfile(path):
        print(f"File not found: {path}")
        return candidates

    current: dict = {}
    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if line.startswith("Refined #"):
                if current.get("gap") is not None:
                    candidates.append(current)
                current = {}
            elif line.startswith("refined_gap="):
                parts = line.split(",")
                gap_str = parts[0].split("=")[1]
                current["gap"] = float(gap_str)
                kfrac_str = line.split("k_frac=(")[1].rstrip(")")
                kf = [float(x) for x in kfrac_str.split(",")]
                current["k_frac"] = np.array(kf)
            elif line.startswith("k_cart=("):
                kc_str = line.split("k_cart=(")[1].rstrip(")")
                kc = [float(x) for x in kc_str.split(",")]
                current["k_cart"] = np.array(kc)
            elif line.startswith("bands=("):
                inner = line.split("bands=(")[1].rstrip(")")
                lo, hi = inner.split(",")
                current["band_lo"] = int(lo)
                current["band_hi"] = int(hi)
    if current.get("gap") is not None:
        candidates.append(current)

    candidates.sort(key=lambda c: c["gap"])
    return candidates


def run_sphere_validation(atoms: Atoms) -> None:
    """Load refined candidates from file and run sphere charge validation only."""
    candidates = load_refined_candidates(WEYL_REFINED)
    if not candidates:
        print("No refined candidates to validate.")
        return

    n_sites = len(atoms)
    edges_pbc = build_edge_list_pbc(atoms, CUT)
    cell = atoms.get_cell().array
    b = reciprocal_lattice(cell)

    nv_in = input(
        f"Number of refined candidates to validate? "
        f"(Press Enter for default = {WEYL_SPHERE_TOP}): "
    ).strip()
    if nv_in == "":
        n_val = min(WEYL_SPHERE_TOP, len(candidates))
    else:
        try:
            n_val = min(int(nv_in), len(candidates))
        except ValueError:
            print("Invalid input. Using default.")
            n_val = min(WEYL_SPHERE_TOP, len(candidates))
    print(f"\nSphere charge validation for top {n_val} refined candidates")
    print(f"  (loaded from {WEYL_REFINED})")
    print("-" * 60)

    out_lines: list[str] = ["Sphere Charge Validation (standalone)", "=" * 50, ""]

    for ri in range(n_val):
        cand = candidates[ri]
        print(f"\n  Candidate {ri}: gap={cand['gap']:.6e}, "
              f"k_frac=({cand['k_frac'][0]:.4f},{cand['k_frac'][1]:.4f},"
              f"{cand['k_frac'][2]:.4f})")

        sweep = sphere_charge_radius_sweep(
            n_sites, edges_pbc, T_HOP, ONSITE, b,
            cand, WEYL_SPHERE_RADII, WEYL_SPHERE_NTHETA_LIST,
        )

        out_lines.append(f"Candidate {ri}: gap={cand['gap']:.6e}")
        print(f"  {'r_frac':>8s}  {'r_cart':>8s}  ", end="")
        for nt in WEYL_SPHERE_NTHETA_LIST:
            print(f"{'Q(nt='+str(nt)+')':>12s}  ", end="")
        print(f"{'delta':>8s}  {'stable':>6s}")
        out_lines.append(
            f"  {'r_frac':>8s}  {'r_cart':>8s}  " +
            "  ".join(f"{'Q(nt='+str(nt)+')':>12s}" for nt in WEYL_SPHERE_NTHETA_LIST) +
            f"  {'delta':>8s}  {'stable':>6s}"
        )

        for sr in sweep:
            charges_str = "  ".join(f"{s['charge']:>12.4f}" for s in sr["sub"])
            line = (f"  {sr['radius_frac']:>8.4f}  {sr['r_cart']:>8.4f}  "
                    f"{charges_str}  {sr['delta_ang']:>8.4f}  "
                    f"{'Y' if sr['stable'] else 'N':>6s}")
            print(line)
            out_lines.append(line)

        any_stable = any(sr["stable"] for sr in sweep)
        tag = "STABLE at some radii" if any_stable else "NOT STABLE at any radius"
        print(f"  -> {tag}")
        out_lines.append(f"  -> {tag}")
        out_lines.append("")

    out_path = os.path.join(DATA_DIR, "weyl_sphere_validation.txt")
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines) + "\n")
    print(f"\nSaved: {out_path}")


def run_linear_dispersion_cuts(atoms: Atoms) -> None:
    """Plot E(k) along Cartesian x/y/z cuts through a refined Weyl candidate."""
    candidates = load_refined_candidates(WEYL_REFINED)
    if not candidates:
        print(f"No refined candidates found in {WEYL_REFINED}.")
        return

    n_sites = len(atoms)
    edges_pbc = build_edge_list_pbc(atoms, CUT)
    cell = atoms.get_cell().array
    b = reciprocal_lattice(cell)
    avg_b = np.mean(np.linalg.norm(b, axis=1))

    print(f"\nAvailable refined candidates ({len(candidates)}):")
    for ci, c in enumerate(candidates):
        print(f"  {ci}) gap={c['gap']:.6e}  k_frac=({c['k_frac'][0]:.4f}, "
              f"{c['k_frac'][1]:.4f}, {c['k_frac'][2]:.4f})")

    idx_in = input("Select candidate index: ").strip()
    idx = int(idx_in)
    if idx < 0 or idx >= len(candidates):
        print(f"Invalid index {idx}.")
        return
    cand = candidates[idx]

    delta_in = input("Delta k fraction (default 0.02): ").strip()
    delta_frac = float(delta_in) if delta_in else 0.02
    delta_cart = delta_frac * avg_b

    k0_frac = cand["k_frac"]
    k0_cart = kfrac_to_kcart(k0_frac, b)
    band_lo = cand.get("band_lo", n_sites // 2 - 1)
    band_hi = cand.get("band_hi", n_sites // 2)

    n_pts = 101
    s_vals = np.linspace(-delta_cart, delta_cart, n_pts)
    dir_names = ["x", "y", "z"]
    dir_vecs = [np.array([1., 0., 0.]), np.array([0., 1., 0.]), np.array([0., 0., 1.])]

    print(f"\nLinear dispersion cuts for candidate {idx}")
    print(f"  k0_cart = ({k0_cart[0]:.6f}, {k0_cart[1]:.6f}, {k0_cart[2]:.6f})")
    print(f"  bands = ({band_lo}, {band_hi}), delta_cart = {delta_cart:.6f} 1/Ang")
    print("-" * 60)

    os.makedirs(FIG_DIR, exist_ok=True)
    for di, (dname, dvec) in enumerate(zip(dir_names, dir_vecs)):
        evals_lo = np.zeros(n_pts)
        evals_hi = np.zeros(n_pts)
        for si, s in enumerate(s_vals):
            k = k0_cart + s * dvec
            Hk = build_tb_hamiltonian_k(n_sites, edges_pbc, k, T_HOP, ONSITE,
                                        reciprocal_b=b)
            ev = np.linalg.eigvalsh(Hk)
            evals_lo[si] = ev[band_lo]
            evals_hi[si] = ev[band_hi]

        mid = n_pts // 2
        ds = s_vals[1] - s_vals[0]
        slope_lo = (evals_lo[mid + 1] - evals_lo[mid - 1]) / (2 * ds) if mid > 0 else 0.0
        slope_hi = (evals_hi[mid + 1] - evals_hi[mid - 1]) / (2 * ds) if mid > 0 else 0.0
        gap_at_center = evals_hi[mid] - evals_lo[mid]
        print(f"  k{dname}: slope_lo={slope_lo:+.4f}, slope_hi={slope_hi:+.4f}, "
              f"gap(s=0)={gap_at_center:.6e}")

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(s_vals, evals_lo, "b-", label=f"band {band_lo}")
        ax.plot(s_vals, evals_hi, "r-", label=f"band {band_hi}")
        ax.axvline(0, color="gray", ls="--", lw=0.5)
        ax.set_xlabel(f"delta k_{dname} (1/Ang)")
        ax.set_ylabel("E")
        ax.set_title(f"Dispersion cut k{dname}, candidate {idx}, gap={cand['gap']:.2e}")
        ax.legend(fontsize=8)
        fig.tight_layout()
        out_png = os.path.join(FIG_DIR, f"weyl_linear_cut_{dname}_{idx}.png")
        fig.savefig(out_png, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"    Saved: {out_png}")


# =====================================================================
#  Rigorous Weyl verification pipeline (task 5)
# =====================================================================


def _sphere_min_gap(
    n_sites: int,
    edges_pbc: list,
    t: float,
    onsite: float,
    b: np.ndarray,
    k0_cart: np.ndarray,
    r_cart: float,
    band_lo: int,
    band_hi: int,
    n_theta: int,
    n_phi: int,
) -> float:
    """Return the minimum gap between band_lo and band_hi on a sphere surface."""
    theta_vals = np.linspace(0, np.pi, n_theta)
    phi_vals = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    min_gap = float("inf")
    for it in range(n_theta):
        for ip in range(n_phi):
            th, ph = theta_vals[it], phi_vals[ip]
            dk = r_cart * np.array([
                np.sin(th) * np.cos(ph),
                np.sin(th) * np.sin(ph),
                np.cos(th),
            ])
            k = k0_cart + dk
            hk = build_tb_hamiltonian_k(n_sites, edges_pbc, k, t, onsite,
                                        reciprocal_b=b)
            ev = np.linalg.eigvalsh(hk)
            g = float(ev[band_hi] - ev[band_lo])
            if g < min_gap:
                min_gap = g
    return min_gap


def rigorous_sphere_sweep(
    n_sites: int,
    edges_pbc: list,
    t: float,
    onsite: float,
    b: np.ndarray,
    k0_cart: np.ndarray,
    band_lo: int,
    band_hi: int,
    radii_cart: list[float],
    n_theta_min: int = 15,
    n_theta_alpha: float = 0.15,
    n_theta_max: int = 200,
    q_tol: float = 0.05,
    gap_surface_tol: float = 5e-4,
) -> list[dict]:
    """Sphere charge sweep with adaptive angular resolution and surface gap guard.

    For each radius, two angular resolutions are used (nt1, nt2 = nt1+6) to
    estimate convergence.  The minimum band gap on the sphere surface is also
    computed.

    A measurement is marked 'reliable' only if:
      |Q(nt1) - Q(nt2)| < q_tol  AND  min_surface_gap > gap_surface_tol

    Returns list of dicts with raw (unrounded) charge values.
    """
    results: list[dict] = []
    for r_cart in radii_cart:
        nt1 = max(n_theta_min, min(n_theta_max, math.ceil(n_theta_alpha / r_cart)))
        nt2 = min(nt1 + 6, n_theta_max)

        q1, info1 = weyl_sphere_charge(
            n_sites, edges_pbc, t, onsite, b,
            k0_cart, band_lo, r_cart, nt1, 2 * nt1,
        )
        q2, info2 = weyl_sphere_charge(
            n_sites, edges_pbc, t, onsite, b,
            k0_cart, band_lo, r_cart, nt2, 2 * nt2,
        )

        min_surf_gap = _sphere_min_gap(
            n_sites, edges_pbc, t, onsite, b,
            k0_cart, r_cart, band_lo, band_hi, nt1, 2 * nt1,
        )

        q_diff = abs(q1 - q2)
        reliable = (q_diff < q_tol) and (min_surf_gap > gap_surface_tol)

        results.append({
            "r_cart": r_cart,
            "nt1": nt1, "np1": 2 * nt1,
            "nt2": nt2, "np2": 2 * nt2,
            "Q_raw_1": q1, "Q_raw_2": q2,
            "Q_diff": q_diff,
            "min_surface_gap": min_surf_gap,
            "skip1": info1["n_skipped"], "plaq1": info1["n_plaq"],
            "skip2": info2["n_skipped"], "plaq2": info2["n_plaq"],
            "reliable": reliable,
        })
    return results


def minimize_gap_local(
    n_sites: int,
    edges_pbc: list,
    t: float,
    onsite: float,
    b: np.ndarray,
    k0_frac: np.ndarray,
    band_lo: int,
    band_hi: int,
    box_frac: float = 0.01,
    bounds_frac: float = 0.02,
) -> dict:
    """Find the k-point (in fractional coords) that minimises the direct gap.

    Uses a coarse 7^3 grid seeded search followed by scipy L-BFGS-B.
    Fractional coordinates are wrapped to [0,1) for BZ periodicity.

    Returns dict with k0_frac, kstar_frac, dk_frac, dk_cart_mag, min_gap.
    """
    from scipy.optimize import minimize

    b = np.asarray(b, dtype=float)
    k0_frac = np.asarray(k0_frac, dtype=float)

    def gap_fn(f_raw):
        f = np.mod(f_raw, 1.0)
        k_cart = kfrac_to_kcart(f, b)
        hk = build_tb_hamiltonian_k(n_sites, edges_pbc, k_cart, t, onsite,
                                    reciprocal_b=b)
        ev = np.linalg.eigvalsh(hk)
        return float(ev[band_hi] - ev[band_lo])

    # Coarse 7^3 grid to find a good seed
    offsets = np.linspace(-box_frac, box_frac, 7)
    best_gap = float("inf")
    best_f = k0_frac.copy()
    for d0 in offsets:
        for d1 in offsets:
            for d2 in offsets:
                f_try = k0_frac + np.array([d0, d1, d2])
                g = gap_fn(f_try)
                if g < best_gap:
                    best_gap = g
                    best_f = np.mod(f_try, 1.0)

    # Scipy optimisation (gap^2 for smoothness near zero)
    def obj(x):
        g = gap_fn(x)
        return g * g

    lo = best_f - bounds_frac
    hi = best_f + bounds_frac
    res = minimize(obj, best_f, method="L-BFGS-B",
                   bounds=list(zip(lo, hi)),
                   options={"maxiter": 200, "ftol": 1e-18, "gtol": 1e-12})

    kstar_frac = np.mod(res.x, 1.0)
    kstar_cart = kfrac_to_kcart(kstar_frac, b)
    dk_frac = kstar_frac - k0_frac
    dk_cart_mag = float(np.linalg.norm(kstar_cart - kfrac_to_kcart(k0_frac, b)))
    min_gap = gap_fn(kstar_frac)

    return {
        "k0_frac": k0_frac.tolist(),
        "kstar_frac": kstar_frac.tolist(),
        "kstar_cart": kstar_cart.tolist(),
        "dk_frac": dk_frac.tolist(),
        "dk_cart_mag": dk_cart_mag,
        "min_gap": min_gap,
        "scipy_fun": float(res.fun),
        "scipy_success": bool(res.success),
    }


def _dispersion_cuts_at_center(
    n_sites: int,
    edges_pbc: list,
    t: float,
    onsite: float,
    b: np.ndarray,
    k_center_cart: np.ndarray,
    band_lo: int,
    band_hi: int,
    delta_cart: float,
    n_pts: int,
    label: str,
    idx: int,
    gap_value: float,
) -> dict:
    """Plot and return dispersion cut info for x/y/z through k_center_cart.

    Returns dict with per-direction slopes and min gaps.
    """
    dir_names = ["x", "y", "z"]
    dir_vecs = [np.array([1., 0., 0.]), np.array([0., 1., 0.]), np.array([0., 0., 1.])]
    s_vals = np.linspace(-delta_cart, delta_cart, n_pts)
    cut_info: dict = {"directions": {}}

    os.makedirs(FIG_DIR, exist_ok=True)
    for dname, dvec in zip(dir_names, dir_vecs):
        evals_lo = np.zeros(n_pts)
        evals_hi = np.zeros(n_pts)
        for si, s in enumerate(s_vals):
            k = k_center_cart + s * dvec
            hk = build_tb_hamiltonian_k(n_sites, edges_pbc, k, t, onsite,
                                        reciprocal_b=b)
            ev = np.linalg.eigvalsh(hk)
            evals_lo[si] = ev[band_lo]
            evals_hi[si] = ev[band_hi]

        gaps = evals_hi - evals_lo
        mid = n_pts // 2
        ds = s_vals[1] - s_vals[0]
        slope_lo = (evals_lo[mid + 1] - evals_lo[mid - 1]) / (2 * ds) if mid > 0 else 0.0
        slope_hi = (evals_hi[mid + 1] - evals_hi[mid - 1]) / (2 * ds) if mid > 0 else 0.0
        gap_center = float(gaps[mid])
        gap_min_cut = float(np.min(gaps))

        cut_info["directions"][dname] = {
            "slope_lo": slope_lo, "slope_hi": slope_hi,
            "gap_center": gap_center, "gap_min_on_cut": gap_min_cut,
        }
        print(f"    k{dname}: slope_lo={slope_lo:+.6f}, slope_hi={slope_hi:+.6f}, "
              f"gap(s=0)={gap_center:.6e}, gap_min_cut={gap_min_cut:.6e}")

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(s_vals, evals_lo, "b-", label=f"band {band_lo}")
        ax.plot(s_vals, evals_hi, "r-", label=f"band {band_hi}")
        ax.axvline(0, color="gray", ls="--", lw=0.5)
        ax.set_xlabel(f"delta k_{dname} (1/Ang)")
        ax.set_ylabel("E")
        ax.set_title(f"{label} k{dname}, cand {idx}, gap={gap_value:.2e}")
        ax.legend(fontsize=8)
        fig.tight_layout()
        out_png = os.path.join(FIG_DIR, f"weyl_{label}_cut_{dname}_{idx}.png")
        fig.savefig(out_png, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"      Saved: {out_png}")

    return cut_info


def run_rigorous_verification(atoms: Atoms) -> None:
    """Task 5: Rigorous Weyl-point verification pipeline.

    For each refined candidate:
      1. (Optional) Local 3D gap minimisation to find k*.
      2. Rigorous sphere-charge sweep at k* with adaptive angular resolution,
         raw (unrounded) Q reporting, and min-surface-gap reliability guard.
      3. Dispersion cuts recentred at k*.
      4. JSON summary per candidate.

    Interpretation guide:
      - Q_raw: raw Berry-flux Chern number on sphere (unrounded).
      - Q_diff: |Q(nt1) - Q(nt2)| measures angular-grid convergence.
      - min_surface_gap: smallest band gap on sphere surface.  If this drops
        below gap_surface_tol, the Berry integration crosses a near-degeneracy
        and Q is NOT trusted (band tracking breaks down).
      - reliable=Y requires BOTH small Q_diff AND large min_surface_gap.
      - A true Weyl node has Q_raw converging to +/-1 at ALL radii with
        reliable=Y.  If Q drifts or becomes unreliable at small radii the
        candidate is likely an avoided crossing, not a topological node.
    """
    candidates = load_refined_candidates(WEYL_REFINED)
    if not candidates:
        print(f"No refined candidates found in {WEYL_REFINED}.")
        return

    n_sites = len(atoms)
    edges_pbc = build_edge_list_pbc(atoms, CUT)
    cell = atoms.get_cell().array
    b = reciprocal_lattice(cell)
    avg_b = float(np.mean(np.linalg.norm(b, axis=1)))

    print(f"\nAvailable refined candidates ({len(candidates)}):")
    for ci, c in enumerate(candidates):
        print(f"  {ci}) gap={c['gap']:.6e}  k_frac=({c['k_frac'][0]:.4f}, "
              f"{c['k_frac'][1]:.4f}, {c['k_frac'][2]:.4f})  "
              f"bands=({c.get('band_lo','?')},{c.get('band_hi','?')})")

    sel_in = input("Candidate indices to verify (comma-sep, or Enter for 0): ").strip()
    if sel_in:
        sel_indices = [int(x) for x in sel_in.split(",")]
    else:
        sel_indices = [0]

    gap_tol_in = input(
        f"Gap surface tolerance? (Enter for default = {VERIFY_GAP_SURFACE_TOL}): "
    ).strip()
    gap_surface_tol = float(gap_tol_in) if gap_tol_in else VERIFY_GAP_SURFACE_TOL

    q_tol_in = input(
        f"Q convergence tolerance? (Enter for default = {VERIFY_Q_TOL}): "
    ).strip()
    q_tol = float(q_tol_in) if q_tol_in else VERIFY_Q_TOL

    do_minimize = VERIFY_MINIMIZE_GAP
    min_in = input(
        f"Run local gap minimiser? [y/n] (default {'y' if do_minimize else 'n'}): "
    ).strip().lower()
    if min_in == "y":
        do_minimize = True
    elif min_in == "n":
        do_minimize = False

    radii_cart = [r * avg_b for r in VERIFY_RADII]

    print("\n" + "=" * 70)
    print("RIGOROUS WEYL VERIFICATION")
    print(f"  radii (1/Ang): {[f'{r:.4e}' for r in radii_cart]}")
    print(f"  Q_tol={q_tol}, gap_surface_tol={gap_surface_tol}")
    print(f"  minimise_gap={do_minimize}")
    print("=" * 70)

    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    for idx in sel_indices:
        if idx < 0 or idx >= len(candidates):
            print(f"\n  Skipping invalid index {idx}")
            continue
        cand = candidates[idx]
        k0_frac = np.asarray(cand["k_frac"])
        k0_cart = kfrac_to_kcart(k0_frac, b)
        band_lo = cand.get("band_lo", n_sites // 2 - 1)
        band_hi = cand.get("band_hi", n_sites // 2)
        summary: dict = {
            "candidate_index": idx,
            "k0_frac": k0_frac.tolist(),
            "k0_cart": k0_cart.tolist(),
            "band_lo": band_lo, "band_hi": band_hi,
            "original_gap": cand["gap"],
        }

        print(f"\n{'='*60}")
        print(f"  Candidate {idx}: bands=({band_lo},{band_hi}), "
              f"gap={cand['gap']:.6e}")
        print(f"  k0_frac = ({k0_frac[0]:.6f}, {k0_frac[1]:.6f}, {k0_frac[2]:.6f})")

        # --- Step 1: Local gap minimisation ---
        if do_minimize:
            print(f"\n  [Step 1] Local gap minimisation (box={VERIFY_MINIMIZER_BOX_FRAC}, "
                  f"bounds={VERIFY_MINIMIZER_BOUNDS_FRAC})")
            opt = minimize_gap_local(
                n_sites, edges_pbc, T_HOP, ONSITE, b,
                k0_frac, band_lo, band_hi,
                box_frac=VERIFY_MINIMIZER_BOX_FRAC,
                bounds_frac=VERIFY_MINIMIZER_BOUNDS_FRAC,
            )
            kstar_frac = np.asarray(opt["kstar_frac"])
            kstar_cart = np.asarray(opt["kstar_cart"])
            print(f"    k0_frac    = ({k0_frac[0]:.6f}, {k0_frac[1]:.6f}, {k0_frac[2]:.6f})")
            print(f"    k*_frac    = ({kstar_frac[0]:.6f}, {kstar_frac[1]:.6f}, {kstar_frac[2]:.6f})")
            print(f"    |k*-k0|    = {opt['dk_cart_mag']:.6e} 1/Ang")
            print(f"    gap(k0)    = {cand['gap']:.6e}")
            print(f"    gap(k*)    = {opt['min_gap']:.6e}")
            print(f"    scipy ok   = {opt['scipy_success']}")
            summary["minimisation"] = opt
        else:
            kstar_frac = k0_frac
            kstar_cart = k0_cart
            print(f"\n  [Step 1] Skipped (minimiser disabled)")
            summary["minimisation"] = None

        # --- Step 2: Rigorous sphere charge sweep at k* ---
        print(f"\n  [Step 2] Rigorous sphere charge sweep at k*")
        print(f"    {'r_cart':>10s}  {'nt1':>4s} {'nt2':>4s}  "
              f"{'Q_raw(nt1)':>12s}  {'Q_raw(nt2)':>12s}  "
              f"{'Q_diff':>8s}  {'min_surf_gap':>12s}  {'reliable':>8s}")
        sweep = rigorous_sphere_sweep(
            n_sites, edges_pbc, T_HOP, ONSITE, b,
            kstar_cart, band_lo, band_hi,
            radii_cart,
            n_theta_min=VERIFY_NTHETA_MIN,
            n_theta_alpha=VERIFY_NTHETA_ALPHA,
            n_theta_max=VERIFY_NTHETA_MAX,
            q_tol=q_tol,
            gap_surface_tol=gap_surface_tol,
        )
        summary["sphere_sweep"] = []
        any_reliable = False
        for row in sweep:
            tag = "Y" if row["reliable"] else "N"
            if row["reliable"]:
                any_reliable = True
            print(f"    {row['r_cart']:>10.6f}  {row['nt1']:>4d} {row['nt2']:>4d}  "
                  f"{row['Q_raw_1']:>12.6f}  {row['Q_raw_2']:>12.6f}  "
                  f"{row['Q_diff']:>8.5f}  {row['min_surface_gap']:>12.6e}  "
                  f"{tag:>8s}")
            if row["min_surface_gap"] < gap_surface_tol:
                print(f"      UNRELIABLE: sphere intersects near-degeneracy; Q not trusted")
            summary["sphere_sweep"].append(row)

        if any_reliable:
            reliable_qs = [r["Q_raw_2"] for r in sweep if r["reliable"]]
            best_q = reliable_qs[-1] if reliable_qs else float("nan")
            print(f"    -> Best reliable Q = {best_q:.6f} (rounded: {round(best_q)})")
            summary["best_Q_reliable"] = best_q
        else:
            print(f"    -> NO reliable measurements at any radius")
            summary["best_Q_reliable"] = None

        # --- Step 3: Dispersion cuts at k* ---
        delta_cart = VERIFY_CUT_HALFWIDTH * avg_b
        print(f"\n  [Step 3] Dispersion cuts at k* (delta={delta_cart:.6f} 1/Ang)")
        cut_info = _dispersion_cuts_at_center(
            n_sites, edges_pbc, T_HOP, ONSITE, b,
            kstar_cart, band_lo, band_hi,
            delta_cart, 101,
            "kstar" if do_minimize else "k0",
            idx,
            summary.get("minimisation", {}).get("min_gap", cand["gap"]) if do_minimize else cand["gap"],
        )
        summary["dispersion_cuts"] = cut_info

        # --- Save JSON ---
        json_path = os.path.join(DATA_DIR, f"weyl_verify_candidate_{idx}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"\n    Saved summary: {json_path}")

    print("\n" + "=" * 70)
    print("Rigorous verification complete.")
    print("=" * 70)


def _setup_geometry(atoms: Atoms, model_edges: list[tuple] | None = None) -> dict:
    """Common geometry setup: cutoff scan, graph, real-space H, eigenvalues.

    Parameters
    ----------
    model_edges : optional list of (i, j, dr, shift, amp) 5-tuples from build_model().
        When provided the real-space Hamiltonian is built from these edges (so it
        is consistent with H(k) for all model choices including nn+2nn).
        When None, falls back to the NN-only adjacency (backward-compat).
    """
    _print_cutoff_scan(atoms, CUT_SCAN)
    print(f"Selected cutoff CUT = {CUT:.2f} Ang")
    adj = build_graph(atoms, CUT)
    stats = graph_stats(adj)

    nn_edge_list = build_edge_list_pbc(atoms, CUT)
    print(f"NN PBC edge list: {len(nn_edge_list)} unique undirected edges "
          f"(graph_stats E={stats['E']}, match={len(nn_edge_list) == stats['E']})")
    if model_edges is not None:
        print(f"Model edge list:  {len(model_edges)} edges")

    n_sites = len(atoms)
    if model_edges is not None:
        # Build H from the actual model edges so real-space H is consistent with H(k)
        h = np.zeros((n_sites, n_sites), dtype=float)
        for i_s in range(n_sites):
            h[i_s, i_s] = ONSITE
        for edge in model_edges:
            si, sj, _dr, _sh, amp = edge[0], edge[1], edge[2], edge[3], edge[4]
            h[si, sj] += -amp
            h[sj, si] += -amp
    else:
        h = build_tb_hamiltonian(adj, T_HOP, ONSITE)
    if EXTRA_HOPS:
        for hop in EXTRA_HOPS:
            hi, hj, amp = hop["i"], hop["j"], hop["amplitude"]
            h[hi, hj] += -amp
            h[hj, hi] += -amp
    assert np.allclose(h, h.T), "Hamiltonian must be symmetric"
    view_hamiltonian(h, CUT)

    if SAVE_H:
        out_h = OUT_H_TEMPLATE.format(cut=CUT)
        os.makedirs(os.path.dirname(out_h) or ".", exist_ok=True)
        np.save(out_h, h)
        print("Saved Hamiltonian:", out_h, "shape:", h.shape, "dtype:", h.dtype)

    evals, _evecs = np.linalg.eigh(h)
    spectrum_diagnostics(evals)

    if SAVE_EVALS:
        out_evals = os.path.join(DATA_DIR, f"eigs_cut{CUT:.2f}.npy")
        os.makedirs(DATA_DIR, exist_ok=True)
        np.save(out_evals, evals)
        print(f"Saved eigenvalues: {out_evals}")

    spacing_diagnostics(
        evals,
        out_png=os.path.join(FIG_DIR, f"spacing_hist_cut{CUT:.2f}.png"),
        drop_frac=0.2,
        cutoff=CUT,
    )

    return {"adj": adj, "stats": stats, "edge_list": nn_edge_list, "h": h, "evals": evals}


def main() -> None:
    atoms = load_cif(CIF_PATH)
    summarize(atoms)

    # --- Optional: geometry-only BZ figure (no TB model, no k-path) ---
    print("\n" + "=" * 50)
    print("START")
    print("  0) First Brillouin zone + reciprocal basis vectors a, b, c only")
    print("     (saves under figures/ if you exit here; no band k-path drawn)")
    print("  1) Continue to tight-binding (model + task menus)")
    print("=" * 50)
    start_input = input("Select [0/1] (default 1): ").strip()
    if start_input == "0":
        outp = os.path.join(FIG_DIR, OUT_BZ_RECIP_BASIS_PNG)
        outh = os.path.join(FIG_DIR, OUT_BZ_RECIP_BASIS_HTML) if EXPORT_INTERACTIVE else None
        render_first_bz_reciprocal_basis(atoms.get_cell().array, outp, outh)
        return

    # --- Model menu ---
    print("\n" + "=" * 50)
    print("MODEL SELECTION")
    print("  1) nn              - nearest-neighbour only")
    print("  2) nn+extra        - NN + manual cross-sheet hops")
    print("  3) nn+2nn          - NN + 2nd-nearest-neighbour (all edges)")
    print("  4) nn+2nn-bdry     - NN + 2NN boundary-only (PBC-crossing edges)")
    print("  5) nn+4nn-bdry     - NN + 4NN boundary-only (PBC-crossing edges)")
    print(f"  6) nn+2nn+3nn+4nn - shells 0-3, amplitude T_HOP*{T_HOP_ALPHA}^shell")
    print("  7) nn+manualpair   - NN + all pairs with same distance as a seed hop")
    print("=" * 50)
    model_input = input("Select model [1/2/3/4/5/6/7] (default 1): ").strip()
    if model_input == "2":
        model_choice = "nn+extra"
    elif model_input == "3":
        model_choice = "nn+2nn"
    elif model_input == "4":
        model_choice = "nn+2nn-bdry"
    elif model_input == "5":
        model_choice = "nn+4nn-bdry"
    elif model_input == "6":
        model_choice = "nn+all4"
    elif model_input == "7":
        model_choice = "nn+manualpair"
    else:
        model_choice = "nn"

    # Set mode-isolated output directories
    mode_name_map = {
        "nn":            "NN",
        "nn+extra":      "NN_plus_extra",
        "nn+2nn":        "NN_2NN",
        "nn+2nn-bdry":   "NN_2NN_BDRY",
        "nn+4nn-bdry":   "NN_4NN_BDRY",
        "nn+all4":       "NN_all4",
        "nn+manualpair": "NN_MANUALPAIR",
    }
    _set_output_dirs(mode_name_map[model_choice])

    # For nn+manualpair: prompt for seed indices BEFORE build_model
    manual_hop_pair = None
    if model_choice == "nn+manualpair":
        n_atoms = len(atoms)
        while True:
            raw = input(
                f"Manual hop selection: enter atom indices i j "
                f"(0 <= i,j < {n_atoms}, i!=j): "
            ).strip()
            parts = raw.split()
            if len(parts) == 2:
                try:
                    i0, j0 = int(parts[0]), int(parts[1])
                    if 0 <= i0 < n_atoms and 0 <= j0 < n_atoms and i0 != j0:
                        manual_hop_pair = (i0, j0)
                        break
                    print(f"  Invalid: indices must be in [0,{n_atoms-1}] and i != j.")
                except ValueError:
                    print("  Invalid: please enter two integers.")
            else:
                print("  Please enter exactly two integers separated by a space.")

    # Build the coupling model FIRST (sets EXTRA_HOPS before any H construction)
    model_edges = build_model(model_choice, atoms, manual_hop_pair=manual_hop_pair)
    print(f"Model selected: {model_choice}")

    # Geometry diagnostics (real-space H, spectrum) now reflect the selected model
    _setup_geometry(atoms, model_edges)

    # --- Task menu ---
    print("\n" + "=" * 50)
    print("TASK SELECTION")
    print("  1) Band structure")
    print("  2) Weyl scan (resolution sweep + refinement)")
    print("  3) Sphere charge validation (from saved refined file)")
    print("  4) Linear dispersion cuts (from refined file)")
    print("  5) Rigorous Weyl verification (minimise + sphere + cuts)")
    print("=" * 50)
    task_input = input("Select task [1/2/3/4/5] (default 1): ").strip()

    if task_input == "2":
        global GAP_THRESH, WEYL_THRESH_INIT
        gap_in = input(f"Custom gap threshold? (Enter for default = {GAP_THRESH}): ").strip()
        if gap_in:
            GAP_THRESH = float(gap_in)
            print(f"  GAP_THRESH = {GAP_THRESH}")
        wt_in = input(f"Custom WEYL_THRESH_INIT? (Enter for default = {WEYL_THRESH_INIT}): ").strip()
        if wt_in:
            WEYL_THRESH_INIT = float(wt_in)
            print(f"  WEYL_THRESH_INIT = {WEYL_THRESH_INIT}")

        if EXPORT_INTERACTIVE:
            export_interactive_plotly(atoms, CUT, OUT_HTML)
        topology_main(atoms)

    elif task_input == "3":
        run_sphere_validation(atoms)

    elif task_input == "4":
        run_linear_dispersion_cuts(atoms)

    elif task_input == "5":
        run_rigorous_verification(atoms)

    else:
        # Band structure (default)
        band_structure(atoms, CUT, T_HOP, ONSITE, model_edges=model_edges)

        if EXPORT_INTERACTIVE:
            export_interactive_plotly(atoms, CUT, OUT_HTML)

        render_3d(atoms, OUT_FIG, edge_cutoff=CUT, edge_sample_nodes=10)
        print(f"Saved: {OUT_FIG}")


if __name__ == "__main__":
    main()
