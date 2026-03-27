"""
Generate synthetic datasets for the unified multi-input demo.

Creates a surface mesh (ellipsoid) with Darcy-flow-like scalar->vector data,
then provides the same data through three interfaces:
  1. mesh:       (points, faces, X, Y)
  2. pointcloud: (points_noisy, X, Y) — faces reconstructed by adapter
  3. graph:      (edge_index, positions, X, Y) — faces reconstructed by adapter
"""
import os
import pickle
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve


def create_ellipsoid_mesh(target_nodes=800, axes=(3.0, 2.0, 1.5)):
    """Create a triangulated ellipsoid surface mesh."""
    import pyvista as pv
    ico = pv.Icosahedron()
    level = 0
    while (10 * 4 ** level + 2) < target_nodes:
        level += 1
    mesh = ico.subdivide(level, subfilter='linear')
    pts = mesh.points.copy()
    radii = np.linalg.norm(pts, axis=1, keepdims=True) + 1e-9
    pts_sphere = pts / radii
    rx, ry, rz = axes
    mesh.points = pts_sphere * np.array([rx, ry, rz])
    mesh = mesh.triangulate().clean()

    points = mesh.points.copy()
    faces_raw = mesh.faces.reshape((-1, 4))[:, 1:]
    faces = faces_raw.astype(np.int64)

    # Normalize
    centroid = points.mean(axis=0)
    points -= centroid
    scale = np.max(np.linalg.norm(points, axis=1))
    points /= (scale + 1e-9)

    return points, faces


def build_cotangent_laplacian(points, faces):
    """Build cotangent Laplacian for surface mesh."""
    n = len(points)
    rows, cols, vals = [], [], []
    areas = np.zeros(n)

    for face in faces:
        i, j, k = face
        vi, vj, vk = points[i], points[j], points[k]
        e_ij = vj - vi
        e_jk = vk - vj
        e_ki = vi - vk

        area = np.linalg.norm(np.cross(e_ij, -e_ki)) / 2.0
        if area < 1e-12:
            continue

        def cot_angle(a, b):
            cos_val = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
            cos_val = np.clip(cos_val, -0.999, 0.999)
            sin_val = np.sqrt(1 - cos_val ** 2)
            return cos_val / (sin_val + 1e-12)

        cot_i = cot_angle(e_ij, -e_ki)
        cot_j = cot_angle(e_jk, -e_ij)
        cot_k = cot_angle(e_ki, -e_jk)

        for (a, b, c) in [(j, k, cot_i), (k, i, cot_j), (i, j, cot_k)]:
            w = c / 2.0
            rows.extend([a, b, a, b])
            cols.extend([b, a, a, b])
            vals.extend([w, w, -w, -w])

        areas[[i, j, k]] += area / 3.0

    L = csr_matrix((vals, (rows, cols)), shape=(n, n))
    return L, areas


def generate_darcy_flow_data(points, faces, n_samples=500, seed=42):
    """
    Generate scalar source -> vector velocity field data.
    Solves Poisson equation: L @ psi = f, then v = grad(psi) projected tangentially.
    Optimized: LU factorization + vectorized gradient computation.
    """
    from scipy.sparse.linalg import factorized
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing

    rng = np.random.RandomState(seed)
    n = len(points)
    L, areas = build_cotangent_laplacian(points, faces)

    # Regularize and pre-factorize (massive speedup over repeated spsolve)
    L_reg = L + 1e-6 * csr_matrix(np.eye(n))
    solve = factorized(L_reg.tocsc())

    # Compute normals (vectorized)
    i_idx, j_idx, k_idx = faces[:, 0], faces[:, 1], faces[:, 2]
    e1 = points[j_idx] - points[i_idx]
    e2 = points[k_idx] - points[i_idx]
    face_normals = np.cross(e1, e2)

    normals = np.zeros((n, 3))
    for d in range(3):
        np.add.at(normals[:, d], i_idx, face_normals[:, d])
        np.add.at(normals[:, d], j_idx, face_normals[:, d])
        np.add.at(normals[:, d], k_idx, face_normals[:, d])
    norms = np.linalg.norm(normals, axis=1, keepdims=True) + 1e-9
    normals = normals / norms

    # Precompute face geometry for vectorized gradient
    face_normals_raw = np.cross(e1, e2)
    face_area2 = np.linalg.norm(face_normals_raw, axis=1, keepdims=True) + 1e-12
    face_n = face_normals_raw / face_area2  # (F, 3)

    # Cross products for gradient formula (precompute once)
    ek_minus_ej = points[k_idx] - points[j_idx]  # vk - vj
    ei_minus_ek = points[i_idx] - points[k_idx]  # vi - vk
    ej_minus_ei = points[j_idx] - points[i_idx]  # vj - vi

    cross_n_km_j = np.cross(face_n, ek_minus_ej)  # n x (vk-vj)
    cross_n_im_k = np.cross(face_n, ei_minus_ek)  # n x (vi-vk)
    cross_n_jm_i = np.cross(face_n, ej_minus_ei)  # n x (vj-vi)

    face_area2_flat = face_area2.flatten()

    # Count per-node face contributions
    node_count = np.zeros(n)
    np.add.at(node_count, i_idx, 1)
    np.add.at(node_count, j_idx, 1)
    np.add.at(node_count, k_idx, 1)
    node_count[node_count == 0] = 1.0

    X_data = np.zeros((n_samples, n), dtype=np.float32)
    Y_data = np.zeros((n_samples, n, 3), dtype=np.float32)

    for s in range(n_samples):
        if (s + 1) % 100 == 0:
            print(f"  Sample {s+1}/{n_samples}...")

        # Random scalar source: sum of Gaussians
        n_sources = rng.randint(2, 6)
        f = np.zeros(n)
        for _ in range(n_sources):
            center_idx = rng.randint(0, n)
            center = points[center_idx]
            sigma = rng.uniform(0.1, 0.4)
            strength = rng.uniform(-2.0, 2.0)
            dists = np.linalg.norm(points - center, axis=1)
            f += strength * np.exp(-dists ** 2 / (2 * sigma ** 2))
        f -= f.mean()
        X_data[s] = f

        # Solve Poisson (fast with pre-factorized LU)
        psi = solve(f)

        # Vectorized gradient on mesh faces
        psi_i, psi_j, psi_k = psi[i_idx], psi[j_idx], psi[k_idx]
        grad_tri = (psi_i[:, None] * cross_n_km_j +
                    psi_j[:, None] * cross_n_im_k +
                    psi_k[:, None] * cross_n_jm_i) / face_area2

        # Scatter to nodes
        grad = np.zeros((n, 3))
        for d in range(3):
            np.add.at(grad[:, d], i_idx, grad_tri[:, d])
            np.add.at(grad[:, d], j_idx, grad_tri[:, d])
            np.add.at(grad[:, d], k_idx, grad_tri[:, d])
        grad /= node_count[:, None]

        # Project tangentially
        dot_n = np.sum(grad * normals, axis=1, keepdims=True)
        v = grad - dot_n * normals
        Y_data[s] = v.astype(np.float32)

    return X_data, Y_data


def generate_and_save(output_dir="./data", n_samples=500, target_nodes=800, seed=42):
    """Generate dataset and save in all three input formats."""
    os.makedirs(output_dir, exist_ok=True)

    print("Generating ellipsoid mesh...")
    points, faces = create_ellipsoid_mesh(target_nodes=target_nodes)
    print(f"  Mesh: {len(points)} nodes, {len(faces)} faces")

    print(f"Generating {n_samples} Darcy flow samples...")
    X_data, Y_data = generate_darcy_flow_data(points, faces, n_samples=n_samples, seed=seed)

    # Build graph edges from mesh
    edges = set()
    for face in faces:
        for i in range(3):
            e = tuple(sorted([face[i], face[(i + 1) % 3]]))
            edges.add(e)
    edge_index = np.array(list(edges), dtype=np.int64)

    dataset = {
        'points': points,
        'faces': faces,
        'edge_index': edge_index,
        'X_data': X_data,
        'Y_data': Y_data,
        'n_nodes': len(points),
        'n_samples': n_samples,
    }

    path = os.path.join(output_dir, "unified_dataset.pkl")
    with open(path, 'wb') as f:
        pickle.dump(dataset, f)
    print(f"Saved to {path}")

    return dataset


if __name__ == "__main__":
    generate_and_save()
