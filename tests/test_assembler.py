import numpy as np
from fe.assembler import assembler_master_stiffness 

def build_mesh():
    """Example based off an exercise in Felippa's IFEM textbook"""
    s = [1, 0.70, 0.48, 0.30, 0.16, 0.07, 0.0]
    corners = {
        1:(0,6), 7:(0,1), 8:(2.5,6),
        14:(np.cos(3*np.pi/8), np.sin(3*np.pi/8)),
        21:(np.cos(np.pi/4), np.sin(np.pi/4)),
        15:(5,6), 22:(5,2),
        28:(np.cos(np.pi/8), np.sin(np.pi/8)),
        29:(5,0), 35:(1,0),
    }
    xy = {k: np.array(v) for k, v in corners.items()}
    coords = np.zeros((35, 2))
    strips = [(1,7,8),(8,14,15),(15,21,22),(22,28,29),(29,35,36)]  # (start,inner,end)
    for start, inner, _ in strips:
        for k in range(7):
            coords[start-1+k] = s[k]*xy[start] + (1-s[k])*xy[inner]

    conn = np.zeros((24, 4), dtype=int)
    conn[0] = [1, 2, 9, 8]
    for e in range(2, 7):  conn[e-1] = conn[e-2] + 1
    conn[6] = conn[5] + 2
    for e in range(8, 13): conn[e-1] = conn[e-2] + 1
    conn[12] = conn[11] + 2
    for e in range(14,19): conn[e-1] = conn[e-2] + 1
    conn[18] = conn[17] + 2
    for e in range(20,25): conn[e-1] = conn[e-2] + 1
    conn -= 1 # making connectivity list 0 based.
    return coords, conn.tolist()

COORDS, CONN = build_mesh()
K = assembler_master_stiffness(COORDS, CONN, thickness=1.0, E_modulus=1000, nu=0.25)
XY = COORDS


def test_symmetry():
    np.testing.assert_allclose(K, K.T, atol=1e-9)

def test_three_rigid_body_modes():
    """Free-free 2D mesh has exactly 3 zero energy modes (2 translations and 1 rotation)."""
    eig = np.linalg.eigvalsh(K)
    assert np.sum(np.abs(eig) < 1e-8) == 3

def test_rigid_body_nullspace():
    """Each rigid body displacement produces zero nodal force."""
    n = len(XY)
    tx = np.zeros(2*n); tx[0::2] = 1.0
    ty = np.zeros(2*n); ty[1::2] = 1.0
    rot = np.zeros(2*n); rot[0::2] = -XY[:,1]; rot[1::2] = XY[:,0]
    for v in (tx, ty, rot):
        np.testing.assert_allclose(K @ v, 0.0, atol=1e-8)

def test_reference_values():
    """compared against an independent second assembler"""
    np.testing.assert_allclose(np.trace(K), 109158.11016571027, rtol=1e-9)
    np.testing.assert_allclose(np.linalg.norm(K), 17052.263588200225, rtol=1e-9)
    np.testing.assert_allclose(K[0,0], 454.17044718268676, rtol=1e-9)
    np.testing.assert_allclose(K[0,1], -194.99654774618568, rtol=1e-9)

