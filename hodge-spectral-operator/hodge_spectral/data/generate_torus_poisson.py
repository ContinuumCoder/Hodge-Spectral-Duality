"""
Torus Poisson Flow Dataset
============================
Task: scalar source (0-form) → tangential velocity field (vector/1-form proxy)

This is the IDEAL task for Hodge decomposition because:
1. The solution naturally decomposes into exact + coexact + harmonic parts
2. The torus has b1=2 (two harmonic 1-forms), so the harmonic component matters
3. The Hodge inductive bias directly models this decomposition

Physics: Given a source function f on the torus,
  solve: Δψ = f (Poisson equation for stream function)
  then:  v = ∇⊥ψ (perpendicular gradient = curl of ψ)

The velocity field v is a 1-form that is:
  - Coexact (in the image of δ₁, the codifferential)
  - Plus a harmonic component depending on the topology

This dataset is fast to generate: pre-factorize L, then batch solve.
"""
import os, pickle, time, numpy as np
import pyvista as pv
from scipy.sparse import csr_matrix, eye as speye
from scipy.sparse.linalg import factorized


def create_torus_mesh(major_R=1.0, minor_r=0.4, u_res=60, v_res=30):
    mesh = pv.ParametricTorus(ringradius=major_R, crosssectionradius=minor_r,
                               u_res=u_res, v_res=v_res)
    mesh = pv.PolyData(mesh)
    if not mesh.is_all_triangles:
        mesh = mesh.triangulate()
    mesh = mesh.clean()
    points = mesh.points.copy()
    faces = mesh.faces.reshape((-1, 4))[:, 1:].astype(np.int64)
    centroid = points.mean(axis=0)
    points -= centroid
    scale = np.max(np.linalg.norm(points, axis=1))
    points /= (scale + 1e-9)
    return points, faces


def build_cotangent_laplacian(points, faces):
    n = len(points)
    rows, cols, vals = [], [], []
    for face in faces:
        i, j, k = face
        vi, vj, vk = points[i], points[j], points[k]
        e_ij, e_jk, e_ki = vj - vi, vk - vj, vi - vk
        area = np.linalg.norm(np.cross(e_ij, -e_ki)) / 2.0
        if area < 1e-12:
            continue
        def cot(a, b):
            c = np.clip(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12), -0.999, 0.999)
            return c / (np.sqrt(1 - c ** 2) + 1e-12)
        for (a, b, cv) in [(j, k, cot(e_ij, -e_ki)), (k, i, cot(e_jk, -e_ij)), (i, j, cot(e_ki, -e_jk))]:
            w = cv / 2.0
            rows.extend([a, b, a, b])
            cols.extend([b, a, a, b])
            vals.extend([w, w, -w, -w])
    return csr_matrix((vals, (rows, cols)), shape=(n, n))


def generate_torus_poisson(output_dir, n_samples=3000, seed=42):
    """Generate scalar source → tangential velocity on torus."""
    os.makedirs(output_dir, exist_ok=True)

    print("Creating torus mesh...")
    points, faces = create_torus_mesh()
    n = len(points)
    print(f"  Torus: {n} nodes, {len(faces)} faces")

    # Normals
    normals = np.zeros((n, 3))
    for face in faces:
        i, j, k = face
        nm = np.cross(points[j] - points[i], points[k] - points[i])
        normals[i] += nm; normals[j] += nm; normals[k] += nm
    normals /= (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-9)

    # Laplacian + factorize
    print("Building Laplacian...")
    L = build_cotangent_laplacian(points, faces)
    L_reg = L + 1e-6 * speye(n)
    solve = factorized(L_reg.tocsc())

    # Precompute vectorized gradient
    fi, fj, fk = faces[:, 0], faces[:, 1], faces[:, 2]
    e1 = points[fj] - points[fi]
    e2 = points[fk] - points[fi]
    fn = np.cross(e1, e2)
    fa2 = np.linalg.norm(fn, axis=1, keepdims=True) + 1e-12
    fn /= fa2
    cr_kj = np.cross(fn, points[fk] - points[fj])
    cr_ik = np.cross(fn, points[fi] - points[fk])
    cr_ji = np.cross(fn, points[fj] - points[fi])
    nc = np.zeros(n)
    np.add.at(nc, fi, 1); np.add.at(nc, fj, 1); np.add.at(nc, fk, 1)
    nc[nc == 0] = 1.0

    # Edges
    edges = set()
    for face in faces:
        for i in range(3):
            edges.add(tuple(sorted([face[i], face[(i + 1) % 3]])))
    edge_index = np.array(list(edges), dtype=np.int64)

    rng = np.random.RandomState(seed)
    X_data = np.zeros((n_samples, n), dtype=np.float32)
    Y_data = np.zeros((n_samples, n, 3), dtype=np.float32)

    print(f"Generating {n_samples} Poisson flow samples...")
    t0 = time.time()

    for s in range(n_samples):
        # Multi-scale source: broad Gaussians + sharp Gaussians + vortex pairs
        f = np.zeros(n)
        # Broad structures (3-6 blobs)
        nb = rng.randint(3, 7)
        for _ in range(nb):
            c = points[rng.randint(0, n)]
            sig = rng.uniform(0.15, 0.4)
            st = rng.uniform(-2.0, 2.0)
            f += st * np.exp(-np.linalg.norm(points - c, axis=1) ** 2 / (2 * sig ** 2))
        # Fine structures (5-10 sharp blobs for detail)
        nf = rng.randint(5, 11)
        for _ in range(nf):
            c = points[rng.randint(0, n)]
            sig = rng.uniform(0.02, 0.08)
            st = rng.uniform(-1.5, 1.5)
            f += st * np.exp(-np.linalg.norm(points - c, axis=1) ** 2 / (2 * sig ** 2))
        # Vortex pairs: opposite-sign close together
        nvp = rng.randint(1, 4)
        for _ in range(nvp):
            c1 = points[rng.randint(0, n)]
            offset = rng.randn(3) * 0.05
            c2 = c1 + offset
            sig = rng.uniform(0.03, 0.08)
            st = rng.uniform(1.0, 3.0)
            f += st * np.exp(-np.linalg.norm(points - c1, axis=1) ** 2 / (2 * sig ** 2))
            f -= st * np.exp(-np.linalg.norm(points - c2, axis=1) ** 2 / (2 * sig ** 2))
        f -= f.mean()
        X_data[s] = f

        # Solve Poisson: L @ psi = f
        psi = solve(f)

        # Compute gradient of psi (vectorized)
        pi, pj, pk = psi[fi], psi[fj], psi[fk]
        gt = (pi[:, None] * cr_kj + pj[:, None] * cr_ik + pk[:, None] * cr_ji) / fa2
        grad = np.zeros((n, 3))
        for d in range(3):
            np.add.at(grad[:, d], fi, gt[:, d])
            np.add.at(grad[:, d], fj, gt[:, d])
            np.add.at(grad[:, d], fk, gt[:, d])
        grad /= nc[:, None]

        # Perpendicular gradient (curl): v = n × ∇ψ
        v = np.cross(normals, grad)

        # Project tangentially (should already be tangential, but ensure)
        dot_n = np.sum(v * normals, axis=1, keepdims=True)
        v -= dot_n * normals

        Y_data[s] = v.astype(np.float32)

        if (s + 1) % 500 == 0:
            print(f"  {s + 1}/{n_samples} ({time.time() - t0:.1f}s)")

    elapsed = time.time() - t0
    print(f"Done: {elapsed:.1f}s ({elapsed / n_samples * 1000:.1f}ms/sample)")

    dataset = {
        'points': points, 'faces': faces, 'edge_index': edge_index,
        'X_data': X_data, 'Y_data': Y_data,
        'n_nodes': n, 'n_samples': n_samples,
        'task': 'poisson_flow', 'topology': 'torus_genus1',
        'physics': 'Laplacian(psi)=f, v=n×grad(psi)',
    }
    path = os.path.join(output_dir, 'torus_poisson_dataset.pkl')
    with open(path, 'wb') as f:
        pickle.dump(dataset, f)
    print(f"Saved: {path} ({os.path.getsize(path) / 1e6:.1f} MB)")
    return dataset


if __name__ == "__main__":
    generate_torus_poisson('./data_torus_poisson', n_samples=3000)
