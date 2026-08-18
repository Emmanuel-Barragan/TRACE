# This module handles 4QUAD element generation
import numpy as np 
import scipy.special

def stiffness_matrix_quad4(nat_coords, thickness, E_matrix):
    """ Generates the element stiffness matrix for a 4 node bilinear quad.

    Input: nat_coords (list format)
           thickness (float)
           E_matrix, plains stress elastic modulii (3x3 matrix)
    Output: Ke
    """

    Ke = np.zeros((8,8))

    gauss_rule = 2 # a Gaussian quadrature rule of 2 is used. Given mesh generation makes rectangles only, this is an exact result.

    gauss_points, weights = scipy.special.roots_legendre(gauss_rule)
    for i in range(gauss_rule):
        for j in range(gauss_rule):

            xi = gauss_points[i]
            eta = gauss_points[j]

            weight_product = weights[i] * weights[j]

            N, dN_dx, dN_dy, detJ = isoparametric_calc_quad4(nat_coords, [xi,eta])

            B = np.zeros((3, 8))
            
            # ::2 means every other element.
            B[0, ::2]  = dN_dx  
            B[1, 1::2] = dN_dy
            B[2, ::2]  = dN_dy; B[2, 1::2] = dN_dx
            
            Ke += (B.T @ E_matrix @ B ) * thickness * weight_product * detJ 

    return Ke
        
def isoparametric_calc_quad4(nat_coords, iso_coords):
    """ Calculates shape functions, x and y derivatives of shape functions, and Jacobian determinant.

    Input: nat_coords (list format)
           iso_coords (as obtained from roots_legendre)

    Output: N, dN_dx, dN_dy, detJ        
    """
    xi, eta = iso_coords
    
    N = 0.25 * np.array([
        (1-xi)*(1-eta),
        (1+xi)*(1-eta),
        (1+xi)*(1+eta),
        (1-xi)*(1+eta)])

    dN_dxi = 0.25 * np.array([
        -(1-eta), (1-eta),
         (1+eta), -(1+eta)])

    dN_deta = 0.25 * np.array([
        -(1-xi), -(1+xi),
         (1+xi),  (1-xi)])

    x, y = np.asarray(nat_coords).T

    J = np.array([
        [dN_dxi @ x,  dN_deta @ x],
        [dN_dxi @ y,  dN_deta @ y]])

    detJ = np.linalg.det(J)

    dN_dx = (J[1,1]*dN_dxi - J[1,0]*dN_deta) / detJ
    dN_dy = (-J[0,1]*dN_dxi + J[0,0]*dN_deta) / detJ

    return N, dN_dx, dN_dy, detJ


