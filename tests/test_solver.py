import numpy as np
from fe.solver import solver
from tests.test_boundary import partition_solve

K = np.array([
    [10, -1,  2,  0],
    [-1, 11, -1,  3],
    [ 2, -1, 10, -1],
    [ 0,  3, -1,  8]], dtype=float)

F = np.array([0, 50, 20, 0], dtype=float)
PRESCRIBED = [0, 3]
FREE = [1, 2]

def test_matches_partition():
    u_mod = solver(K, F, PRESCRIBED)
    np.testing.assert_allclose(u_mod, partition_solve(K, F, FREE))
    np.testing.assert_allclose(u_mod[PRESCRIBED], 0.0, atol=1e-12)

def test_prescribed_dofs_are_zero_in_displacements():
    u = solver(K,F, PRESCRIBED)
    np.testing.assert_allclose(u[PRESCRIBED], 0.0, atol=1e-12)





