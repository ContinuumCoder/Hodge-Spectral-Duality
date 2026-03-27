"""
Ellipsoid External Aerodynamics Dataset
========================================
Physics: Vortex source → Poisson stream function → rotational velocity + global coupling

This mimics external flow around a bluff body:
  - Input X: vorticity (scalar 0-form) with sharp vortex pairs and smooth background
  - Output Y: tangential velocity (vector field) combining:
    1. Rotational part: v_rot = n × ∇ψ (from Poisson solve Δψ = ω)
    2. Global coupling: background flow induced by vorticity moments
    3. Vortex separation zones with stagnation points

The global coupling creates long-range correlations and rich topological
features (separation lines, stagnation points, vortex shedding patterns).
"""
import os, pickle, time, numpy as np
import pyvista as pv
from scipy.sparse import csr_matrix, eye as speye
from scipy.sparse.linalg import factorized


def create_ellipsoid(target_nodes=2500, axes=(3.0, 2.0, 1.5)):
    ico = pv.Icosahedron()
    level = 0
    while (10 * 4 ** level + 2) < target_nodes:
        level += 1
    mesh = ico.subdivide(level, subfilter='linear')
    pts = mesh.points.copy()
    r = np.linalg.norm(pts, axis=1, keepdims=True) + 1e-9
    mesh.points = (pts / r) * np.array(axes)
    mesh = mesh.triangulate().clean()
    pts = mesh.points.copy()
    fcs = mesh.faces.reshape((-1, 4))[:, 1:].astype(np.int64)
    pts -= pts.mean(0)
    pts /= (np.max(np.linalg.norm(pts, axis=1)) + 1e-9)
    return pts, fcs


def build_L(pts, fcs):
    n = len(pts)
    rows, cols, vals = [], [], []
    areas = np.zeros(n)
    for f in fcs:
        i, j, k = f
        vi, vj, vk = pts[i], pts[j], pts[k]
        eij, ejk, eki = vj - vi, vk - vj, vi - vk
        area = np.linalg.norm(np.cross(eij, -eki)) / 2
        if area < 1e-12:
            continue
        def cot(a, b):
            c = np.clip(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12), -0.999, 0.999)
            return c / (np.sqrt(1 - c ** 2) + 1e-12)
        for a, b, cv in [(j, k, cot(eij, -eki)), (k, i, cot(ejk, -eij)), (i, j, cot(eki, -ejk))]:
            w = cv / 2
            rows.extend([a, b, a, b]); cols.extend([b, a, a, b]); vals.extend([w, w, -w, -w])
        areas[[i, j, k]] += area / 3
    return csr_matrix((vals, (rows, cols)), shape=(n, n)), areas


def generate_ellipsoid_aero(output_dir, n_samples=3000, seed=42):
    os.makedirs(output_dir, exist_ok=True)

    print("Creating ellipsoid mesh...")
    pts, fcs = create_ellipsoid()
    n = len(pts)
    print(f"  Ellipsoid: {n} nodes, {len(fcs)} faces")

    # Normals
    normals = np.zeros((n, 3))
    for f in fcs:
        i, j, k = f
        nm = np.cross(pts[j] - pts[i], pts[k] - pts[i])
        normals[i] += nm; normals[j] += nm; normals[k] += nm
    normals /= (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-9)

    # Laplacian + solver
    L, areas = build_L(pts, fcs)
    solve = factorized((L + 1e-6 * speye(n)).tocsc())
    total_area = areas.sum()

    # Vectorized gradient
    fi, fj, fk = fcs[:, 0], fcs[:, 1], fcs[:, 2]
    e1 = pts[fj] - pts[fi]; e2 = pts[fk] - pts[fi]
    fn = np.cross(e1, e2)
    fa2 = np.linalg.norm(fn, axis=1, keepdims=True) + 1e-12
    fn /= fa2
    cr_kj = np.cross(fn, pts[fk] - pts[fj])
    cr_ik = np.cross(fn, pts[fi] - pts[fk])
    cr_ji = np.cross(fn, pts[fj] - pts[fi])
    nc = np.zeros(n)
    np.add.at(nc, fi, 1); np.add.at(nc, fj, 1); np.add.at(nc, fk, 1)
    nc[nc == 0] = 1

    # Background flow basis: 3 tangential fields along x, y, z
    bg_basis = np.zeros((3, n, 3))
    for d in range(3):
        uniform = np.zeros((n, 3))
        uniform[:, d] = 1.0
        tangent = uniform - np.sum(uniform * normals, axis=1, keepdims=True) * normals
        norm_t = np.linalg.norm(tangent, axis=1, keepdims=True) + 1e-9
        bg_basis[d] = tangent / norm_t

    edges = set()
    for f in fcs:
        for i in range(3):
            edges.add(tuple(sorted([f[i], f[(i + 1) % 3]])))
    edge_index = np.array(list(edges), dtype=np.int64)

    # Centered positions for moment computation
    pts_centered = pts - np.average(pts, weights=areas, axis=0)

    rng = np.random.RandomState(seed)
    X_data = np.zeros((n_samples, n), dtype=np.float32)
    Y_data = np.zeros((n_samples, n, 3), dtype=np.float32)

    GLOBAL_COUPLING_SCALE = 5.0
    GLOBAL_COUPLING_SHARPNESS = 2.0

    print(f"Generating {n_samples} samples...")
    t0 = time.time()

    for s in range(n_samples):
        omega = np.zeros(n)

        # Vortex sources: mix of single blobs and vortex pairs
        n_sources = rng.randint(3, 10)
        for _ in range(n_sources):
            center = pts[rng.randint(0, n)]
            sigma = rng.uniform(0.05, 0.25)
            strength = rng.uniform(0.5, 2.0) * rng.choice([-1, 1])

            # 70% chance of vortex pair (opposite sign nearby)
            if rng.random() < 0.7:
                separation = rng.uniform(0.03, 0.12)
                offset = rng.randn(3)
                offset = offset / (np.linalg.norm(offset) + 1e-9) * separation
                center2 = center + offset

                d1 = np.linalg.norm(pts - center, axis=1)
                d2 = np.linalg.norm(pts - center2, axis=1)
                omega += strength * np.exp(-d1 ** 2 / (2 * sigma ** 2))
                omega -= strength * np.exp(-d2 ** 2 / (2 * sigma ** 2))
            else:
                d = np.linalg.norm(pts - center, axis=1)
                omega += strength * np.exp(-d ** 2 / (2 * sigma ** 2))

        # Zero-mean
        omega -= np.sum(omega * areas) / total_area
        X_data[s] = omega

        # Solve Poisson: L @ psi = M @ omega (mass-weighted RHS)
        rhs = areas * omega
        psi = solve(rhs)
        psi -= np.sum(psi * areas) / total_area  # zero-mean

        # Rotational velocity: v_rot = n × ∇ψ
        pi, pj, pk = psi[fi], psi[fj], psi[fk]
        gt = (pi[:, None] * cr_kj + pj[:, None] * cr_ik + pk[:, None] * cr_ji) / fa2
        grad = np.zeros((n, 3))
        for d in range(3):
            np.add.at(grad[:, d], fi, gt[:, d])
            np.add.at(grad[:, d], fj, gt[:, d])
            np.add.at(grad[:, d], fk, gt[:, d])
        grad /= nc[:, None]
        v_rot = np.cross(normals, grad)

        # Global coupling: vorticity moments → background flow
        weighted_omega = omega * areas
        moments = weighted_omega @ pts_centered  # (3,) = ∫ ω * x dA
        coeff = GLOBAL_COUPLING_SCALE * np.tanh(GLOBAL_COUPLING_SHARPNESS * moments)
        v_global = sum(coeff[d] * bg_basis[d] for d in range(3))

        # Total velocity
        v = v_rot + v_global
        # Project tangentially
        v -= np.sum(v * normals, axis=1, keepdims=True) * normals
        # Normalize per sample
        v_max = np.linalg.norm(v, axis=1).max() + 1e-9
        v /= v_max

        Y_data[s] = v.astype(np.float32)

        if (s + 1) % 500 == 0:
            print(f"  {s + 1}/{n_samples} ({time.time() - t0:.1f}s)")

    print(f"Done: {time.time() - t0:.1f}s")

    dataset = {
        'points': pts, 'faces': fcs, 'edge_index': edge_index,
        'X_data': X_data, 'Y_data': Y_data,
        'n_nodes': n, 'n_samples': n_samples,
        'task': 'external_aerodynamics',
        'topology': 'ellipsoid_genus0',
        'physics': 'vortex_source → Poisson_stream → rotational_velocity + global_coupling',
    }
    path = os.path.join(output_dir, 'ellipsoid_aero_dataset.pkl')
    with open(path, 'wb') as f:
        pickle.dump(dataset, f)
    print(f"Saved: {path} ({os.path.getsize(path) / 1e6:.1f} MB)")


if __name__ == "__main__":
    generate_ellipsoid_aero('./data_ellipsoid_aero', n_samples=3000)
