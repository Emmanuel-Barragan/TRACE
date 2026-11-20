# This module handles both stiffness matrix and force vector boundary condition application

def modified_stiffness_matrix(K_global, prescribed_DoF):
    """ Modifies the global master stiffness to take into account BC DoFs

    Input: K_global (ndarray)
           prescribed_DoF (list of DoF, 0-based)

    Output: K_modified
    """
    K_modified = K_global.copy()

    # numpy lets you modify all prescribed_DoF entries at once 
    K_modified[prescribed_DoF, :] = 0
    K_modified[:, prescribed_DoF] = 0
    K_modified[prescribed_DoF, prescribed_DoF] = 1

    return K_modified


def modified_force_vector(f_vector, K_global, prescribed_DoF, prescribed_DoF_values):
    """ Modifies the force vector to take into account BC DoFs

    Input: f_vector (ndarray)
           prescribed_DoF (list of DoF, 0-based)
           prescribed_DoF_values (list of DoF prescribed values)

    Output: f_modified
    """
    f_modified = f_vector.copy().astype(float)

    for index_prescribed, prescribed_val in zip(prescribed_DoF, prescribed_DoF_values):
        f_modified = f_modified - K_global[:, index_prescribed]*prescribed_val

    f_modified[prescribed_DoF] = prescribed_DoF_values 

    return f_modified


