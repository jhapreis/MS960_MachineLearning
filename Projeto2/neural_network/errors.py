import sys



# =============================================================================
def dimensional_error(shape_1, shape_2):
    '''
    '''

    if shape_1 != shape_2:
        err_msg = f'\n   Error on the shapes: {shape_1} and {shape_2}. They were supposed to be equal.\n'
        raise Exception(err_msg)



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



# =============================================================================
def check_equal_values(df_previous, df_now, exit=1):
    '''
    '''

    equality = df_previous.equals(df_now)

    if equality:
        msg = f"\n\n      The DataFrame is not being updated.\n\n"
        if exit == 1:
            raise ValueError(msg)
        else:
            print(msg)
        


# =============================================================================
def Check_ValuesFrac_Lambdas(values_frac, lambdas):
    '''
    '''

    if ( len(values_frac) > 1) and ( len(lambdas) > 1):

        print("   \nCAUTION: you are going to execute the backprop multiple times for both, percentual_frac and lambda_values.\nDo you want to continue? [y/n] ")
        userInput = input()

        if userInput == 'y':
            raise NotImplementedError()
        elif userInput == 'n':
            print("\nok... quitting\n")
            sys.exit()
        else:
            print("\nShould be \"y\" or \"n\"")
            sys.exit()


