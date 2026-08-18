import numpy as np
from fe.quad4 import stiffness_matrix_quad4

# Example based off an exercise in Felippa's IFEM textbook
nu = 1/3
E = 96
E_matrix = np.array([
    [E/(1-nu**2),    nu*E/(1-nu**2), 0],
    [nu*E/(1-nu**2), E/(1-nu**2),    0],
    [0, 0, E*(1-nu)/(2*(1-nu**2))],
])
nat_coords = [(0, 0), (8, 0), (8, 4), (0, 4)]  

def test_symmetry():
    Ke = stiffness_matrix_quad4(nat_coords, 1.0, E_matrix)
    np.testing.assert_allclose(Ke, Ke.T, atol=1e-12)

def test_rigid_body_translation():
    """Translating all nodes by (1,0) is a rigid motion, thus zero nodal forces are expected."""
    Ke = stiffness_matrix_quad4(nat_coords, 1.0, E_matrix)
    u_translate_x = np.array([1, 0, 1, 0, 1, 0, 1, 0], dtype=float)
    np.testing.assert_allclose(Ke @ u_translate_x, 0.0, atol=1e-10)

def test_matches_reference():
    """Full matrix vs. example's real results"""
    Ke = stiffness_matrix_quad4(nat_coords, 1.0, E_matrix)
    assert np.isclose(Ke[0, 0], 42)
    np.testing.assert_allclose(Ke[0], [42, 18, -6, 0, -21, -18, -15, 0], atol=1e-10)
    np.testing.assert_allclose(Ke[1], [18, 78, 0, 30, -18, -39, 0, -69], atol=1e-10)
    np.testing.assert_allclose(Ke[2], [-6, 0, 42, -18, -15, 0, -21, 18], atol=1e-10)
    np.testing.assert_allclose(Ke[3], [0, 30, -18, 78, 0, -69, 18, -39], atol=1e-10)
    np.testing.assert_allclose(Ke[4], [-21, -18, -15, 0, 42, 18, -6, 0], atol=1e-10)
    np.testing.assert_allclose(Ke[5], [-18, -39, 0, -69, 18, 78, 0, 30], atol=1e-10)
    np.testing.assert_allclose(Ke[6], [-15, 0, -21, 18, -6, 0, 42, -18], atol=1e-10)
    np.testing.assert_allclose(Ke[7], [0, -69, 18, -39, 0, 30, -18, 78], atol=1e-10)
