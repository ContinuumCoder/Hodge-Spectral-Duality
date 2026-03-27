"""
Fast torus advection-diffusion dataset generator.
Key optimization: pre-factorize implicit solver ONCE, vectorize gradient.
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
        e_ij, e_jk, e_ki = vj-vi, vk-vj, vi-vk
        area = np.linalg.norm(np.cross(e_ij, -e_ki)) / 2.0
        if area < 1e-12: continue
        def cot(a, b):
            c = np.clip(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12), -0.999, 0.999)
            return c / (np.sqrt(1-c**2)+1e-12)
        for (a, b, cv) in [(j,k,cot(e_ij,-e_ki)), (k,i,cot(e_jk,-e_ij)), (i,j,cot(e_ki,-e_jk))]:
            w = cv/2.0
            rows.extend([a,b,a,b]); cols.extend([b,a,a,b]); vals.extend([w,w,-w,-w])
    return csr_matrix((vals, (rows, cols)), shape=(n, n))


def generate_torus_dataset(output_dir, n_samples=2000, n_steps=15, diffusivity=0.01, dt=0.02):
    os.makedirs(output_dir, exist_ok=True)

    print("Creating torus mesh...")
    points, faces = create_torus_mesh()
    n = len(points)
    print(f"  Torus: {n} nodes, {len(faces)} faces")

    # Normals
    normals = np.zeros((n, 3))
    for face in faces:
        i, j, k = face
        nm = np.cross(points[j]-points[i], points[k]-points[i])
        normals[i] += nm; normals[j] += nm; normals[k] += nm
    normals /= (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-9)

    # Pre-factorize implicit diffusion solver (ONCE)
    print("Pre-factorizing implicit solver...")
    L = build_cotangent_laplacian(points, faces)
    solve = factorized((speye(n) - dt * diffusivity * L).tocsc())

    # Precompute vectorized gradient operator
    fi, fj, fk = faces[:,0], faces[:,1], faces[:,2]
    e1 = points[fj] - points[fi]
    e2 = points[fk] - points[fi]
    fn = np.cross(e1, e2)
    fa2 = np.linalg.norm(fn, axis=1, keepdims=True) + 1e-12
    fn /= fa2
    cr_kj = np.cross(fn, points[fk]-points[fj])
    cr_ik = np.cross(fn, points[fi]-points[fk])
    cr_ji = np.cross(fn, points[fj]-points[fi])
    nc = np.zeros(n)
    np.add.at(nc, fi, 1); np.add.at(nc, fj, 1); np.add.at(nc, fk, 1)
    nc[nc==0] = 1.0

    # Toroidal/poloidal basis
    theta = np.arctan2(points[:,1], points[:,0])
    vt = np.stack([-np.sin(theta), np.cos(theta), np.zeros(n)], axis=1)
    vt -= np.sum(vt*normals, 1, keepdims=True)*normals
    vt /= (np.linalg.norm(vt, axis=1, keepdims=True)+1e-9)
    rxy = np.sqrt(points[:,0]**2 + points[:,1]**2)
    phi = np.arctan2(rxy - 1.0, points[:,2])
    vp = np.stack([-np.sin(phi)*np.cos(theta), -np.sin(phi)*np.sin(theta), np.cos(phi)], axis=1)
    vp -= np.sum(vp*normals, 1, keepdims=True)*normals
    vp /= (np.linalg.norm(vp, axis=1, keepdims=True)+1e-9)

    edges = set()
    for face in faces:
        for i in range(3):
            edges.add(tuple(sorted([face[i], face[(i+1)%3]])))
    edge_index = np.array(list(edges), dtype=np.int64)

    rng = np.random.RandomState(42)
    X_data = np.zeros((n_samples, n), dtype=np.float32)
    Y_data = np.zeros((n_samples, n), dtype=np.float32)
    V_data = np.zeros((n_samples, n, 3), dtype=np.float32)

    print(f"Generating {n_samples} samples...")
    t0 = time.time()

    for s in range(n_samples):
        # Random IC
        nb = rng.randint(2, 6)
        u0 = np.zeros(n)
        for _ in range(nb):
            c = points[rng.randint(0,n)]
            sig = rng.uniform(0.1, 0.3)
            st = rng.uniform(-2.0, 2.0)
            u0 += st * np.exp(-np.linalg.norm(points-c, axis=1)**2/(2*sig**2))
        u0 -= u0.mean()
        X_data[s] = u0

        alpha, beta = rng.uniform(0.3, 1.5), rng.uniform(0.0, 0.8)
        vf = alpha*vt + beta*vp
        V_data[s] = vf

        # Time-step
        u = u0.astype(np.float64)
        for _ in range(n_steps):
            ui, uj, uk = u[fi], u[fj], u[fk]
            gt = (ui[:,None]*cr_kj + uj[:,None]*cr_ik + uk[:,None]*cr_ji) / fa2
            g = np.zeros((n, 3))
            for d in range(3):
                np.add.at(g[:,d], fi, gt[:,d])
                np.add.at(g[:,d], fj, gt[:,d])
                np.add.at(g[:,d], fk, gt[:,d])
            g /= nc[:,None]
            u = solve(u - dt * np.sum(vf*g, axis=1))

        Y_data[s] = u.astype(np.float32)
        if (s+1) % 200 == 0:
            print(f"  {s+1}/{n_samples} ({time.time()-t0:.1f}s)")

    elapsed = time.time() - t0
    print(f"Done: {elapsed:.1f}s total ({elapsed/n_samples*1000:.1f}ms/sample)")

    dataset = {
        'points': points, 'faces': faces, 'edge_index': edge_index,
        'X_data': X_data, 'Y_data': Y_data, 'V_data': V_data,
        'n_nodes': n, 'n_samples': n_samples,
        'task': 'scalar_advection_diffusion', 'topology': 'torus_genus1',
        'physics': {'diffusivity': diffusivity, 'dt': dt, 'n_steps': n_steps},
    }
    path = os.path.join(output_dir, 'torus_advdiff_dataset.pkl')
    with open(path, 'wb') as f:
        pickle.dump(dataset, f)
    print(f"Saved: {path} ({os.path.getsize(path)/1e6:.1f} MB)")
    return dataset


if __name__ == "__main__":
    generate_torus_dataset('./data_torus', n_samples=2000, n_steps=15)
