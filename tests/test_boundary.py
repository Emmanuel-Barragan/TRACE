import numpy as np
from fe.boundary import modified_stiffness_matrix, modified_force_vector

K = np.array([
    [10, -1,  2,  0],
    [-1, 11, -1,  3],
    [ 2, -1, 10, -1],
    [ 0,  3, -1,  8]], dtype=float)

F = np.array([0, 50, 20, 0], dtype=float)
PRESCRIBED = [0, 3]
FREE = [1, 2]


def partition_solve(K, f, free):
    """Solves via the partition method, used as independent check"""
    u = np.zeros(len(f))
    u[free] = np.linalg.solve(K[np.ix_(free, free)], f[free])
    return u


def test_prescribed_rows_are_unit_vectors():
    Km = modified_stiffness_matrix(K, PRESCRIBED)
    np.testing.assert_allclose(Km[PRESCRIBED], np.eye(len(F))[PRESCRIBED])

def test_symmetry():
    Km = modified_stiffness_matrix(K, PRESCRIBED)
    np.testing.assert_allclose(Km, Km.T, atol=1e-12)

def test_free_block_unchanged():
    Km = modified_stiffness_matrix(K, PRESCRIBED)
    np.testing.assert_allclose(Km[np.ix_(FREE, FREE)], K[np.ix_(FREE, FREE)])

def test_force_vector_modified():
    fm = modified_force_vector(F, PRESCRIBED)
    np.testing.assert_allclose(fm[PRESCRIBED], 0.0, atol=1e-12)   # prescribed zeroed
    np.testing.assert_allclose(fm[FREE], F[FREE])                 # free untouched

def test_solution_matches_partition():
    u_mod = np.linalg.solve(modified_stiffness_matrix(K, PRESCRIBED),
                            modified_force_vector(F, PRESCRIBED))
    np.testing.assert_allclose(u_mod, partition_solve(K, F, FREE))
    np.testing.assert_allclose(u_mod[PRESCRIBED], 0.0, atol=1e-12)
