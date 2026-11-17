import numpy as np 
from fe.quad4 import E_matrix_generation
from fe.recovery import nodal_stress_quad4, total_nodal_stresses



E_MODULUS = 1000
NU = 0.25
E_MATRIX = E_matrix_generation(E_MODULUS, NU)

COORDS = np.array([[0.0, 0.0],
                   [2.0, 0.0],
                   [2.3, 1.1],
                   [0.2, 0.9]])

def linear_field(coords, a, b):
    x, y = coords[:, 0], coords[:, 1]
    u = a[0] + a[1]*x + a[2]*y
    v = b[0] + b[1]*x + b[2]*y
    return np.column_stack([u, v]).flatten()

def test_constant_stress_from_linear_fiell():
    a = (0.0, 1e-3, -4e-4)      
    b = (0.0, 2e-4,  7e-4)     

    u_element = linear_field(COORDS, a, b)

    eps_expected = np.array([a[1], b[2], a[2] + b[1]]) 
    sigma_expected = E_MATRIX @ eps_expected

    stress = nodal_stress_quad4(COORDS, E_MATRIX, u_element)

    assert stress.shape == (4, 3)
    for node in range(4):
        np.testing.assert_allclose(stress[node], sigma_expected, rtol=1e-9)

def test_rigid_body_zero_stress():
    u_element = [1,0,1,0,1,0,1,0]
    stress = nodal_stress_quad4(COORDS, E_MATRIX, u_element)
    for node in range(4):
        np.testing.assert_allclose(stress[node], [0, 0, 0], atol=1e-9)

def test_averaging_at_shared_node():

    coordinates = np.array([[0,0],
                   [1,0], # boundary 
                   [2,0],
                   [2,1],
                   [1,1], # boundary
                   [0,1]])
    
    connectivity = [[0,1, 4, 5],
                    [1, 2, 3, 4]]

    a = (0.0, 1e-3, -4e-4)      
    b = (0.0, 2e-4,  7e-4)     

    u = linear_field(coordinates, a , b)
    
    eps_expected = np.array([a[1], b[2], a[2] + b[1]])

    sigma_expected = E_MATRIX @ eps_expected

    stress = total_nodal_stresses(coordinates, connectivity, E_MODULUS, NU, u)

    assert stress.shape == (6,3)

    for node in range(6):
        np.testing.assert_allclose(stress[node], sigma_expected, rtol=1e-9)



