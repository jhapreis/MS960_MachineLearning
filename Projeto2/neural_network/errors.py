# =============================================================================
def dimensional_error(shape_1, shape_2):
    '''
    '''

    if shape_1 != shape_2:
        raise ValueError(f'\n   Error on the shapes: {shape_1} and {shape_2}. They were supposed to be equal.\n')



# =============================================================================
def matrix_multiplication_error(shape_1, shape_2):
    '''
    '''
    
    if shape_1[1] != shape_2[0]:
        raise ValueError(f'\n   Error on the shapes: {shape_1} and {shape_2}. They were supposed to be allined, such as (x,y) and (y,z).\n')



# =============================================================================
def single_dimension_error(shape_1, shape_2, axis=0):
    '''
    '''

    if shape_1[axis] != shape_2[axis]:
        raise ValueError(f'\n   Error on the shapes: {shape_1} and {shape_2}. Along axis={axis}, they were supposed to be equal, such as (x,y) and (z,y).\n')



