import numpy as np
 
 
 
def Phase_1(A, b, z):
    m, n = A.shape
    I = np.identity(m)
    # Adds artificial variables to all the constraints i.e "checks" for the identity part
    A_new = np.hstack((A, I))
    cb = np.array([1 for _ in range(m)])
    z_new = cb@I@A          # C_N is zero in 2 phase objective
    padding = np.array([0 for _ in range(m)])
    z_new = np.hstack((z_new, padding))
    xb1 = np.array([(n+i+1) for i in range(m)])
    z_o = cb@I@b
    # building the tableau for 2 phase
    tb, m_2, n_2 = build_tableau(A_new, b, -z_new, xb1, z_o)
    # Phase 1 of Simplex
    tb, z_star, sol, xb = Simplex(tb, m_2, n_2, opt_flag= False)
    if z_star != 0:
        print("Infeasible")
        return (0, 0, 0, 0) 
    # Removing the artificial variables 
    artf_idx = list(range(n+m+2))
    del artf_idx[-(m+1):-1]
    tb = tb[:, artf_idx]
    # Redundent Constraints
    l = [np.where(xb==i)[0] for i in xb1]
    for i in l:
        tb = np.delete(tb, i+2, axis = 0)
        xb = np.delete(xb, i, axis=0)
        sol = np.delete(sol, i, axis = 0)
    m = xb.shape[0]    
    # Convert the  2 phase z row to regular z row
    tb[0, :] = np.array([i for i in range(n+2)])
    nbv_idx = np.array(list(range(1, n+1)))
    nbv_idx = np.setdiff1d(nbv_idx, xb)
    B_invN = tb[2:, nbv_idx]
    cb = z[(xb-1).astype(int)]
    cn = z[(nbv_idx-1).astype(int)] 
    tb[1, nbv_idx.astype(int)] = cb@B_invN - cn
    tb[1, xb.astype(int)] = 0
    tb[1, -1] = cb@sol
    f_tb, f_z_star, f_sol, f_xb = Simplex(tb, m, n)
    return f_tb, f_z_star, f_sol, f_xb
 
 
 
def Optimality_check(z, xb):
    #Bland's rule: picks the smallest index for a tie
    idx = np.argmax(z)
    if z[idx] == 0:
        # Have to add the multiple solutions case, use xb
        return True, idx+1
    return False, idx+1
    
def ratio_test(b, pivot_col, xb):
    ix = 0
    flag2 = False
    if pivot_col[np.argmax(pivot_col)] <= 0:
        flag2 = True
        return flag2, ix+2
    pivot_col[pivot_col<0] = 0
    np.seterr(divide='ignore')
    t = b/pivot_col
    t[np.isnan(t)] = np.Inf
    # t[t<0] = np.Inf
    ix = np.argmin(t)
    #Bland's rule: picks the smallest index for a tie
    bld_rule = (t==t[ix])
    if sum(bld_rule) > 1:
        ties = xb[bld_rule]
        min_idx = np.argmin(ties)
        ix = np.where(xb==ties[min_idx])[0][0]
    return flag2, ix+2
 
 
 
def build_tableau(A, b, z, xb, z_o = 0):
    m, n = A.shape
    tableau = np.zeros((m+2, n+2))
    tableau[0,:] = np.array(range(n+2))
    tableau[1, 1:n+1] = -z
    tableau[1, -1] = z_o
    tableau[2:(m+2), 1:n+1] = A
    tableau[2:(m+2), -1] = b
    tableau[2:, 0] = xb
    return tableau, m, n
 
def Simplex(tableau, m, n, opt_flag = True):
    i = 0
    while 1:
        eps = 1e-6
        # eps = 1e-15
        tableau[np.abs(tableau) < eps] = 0
        print("dtypes:",tableau.dtype)
        z = tableau[1, 1:n+1]
        xb = tableau[2:, 0]
        b = tableau[2:(m+2), -1]
        flag1, idx = Optimality_check(z, xb)
        if flag1 == True:
            if opt_flag == True:
                print("Optimal Solution.")
            break
        pivot_col = np.copy(tableau[2:, idx])
        flag2, ix = ratio_test(b, pivot_col, xb)
        if flag2 == True:
            print("Unbounded.")
            break
        pivot = tableau[ix, idx]
        tableau[ix, 1:] = tableau[ix, 1:]/pivot
        npvt = list(range(1, m+2))
        npvt.remove(ix)
        pivot_col_ex_pivot = tableau[npvt, idx]
        pivot_col_ex_pivot.shape = (m,1)
        pivot_row = tableau[ix, 1:]
        tableau[npvt, 1:] = tableau[npvt, 1:] - (pivot_col_ex_pivot * pivot_row) #broadcasting works
        tableau[ix, 0] = idx
        i = i+1
        print("tableau for iteration {0}: \n".format(i),tableau)

 
    z_star =  tableau[1, -1]
    sol = b
    xb = tableau[2:, 0]
    
    return tableau, z_star, sol, xb
 


if __name__ == '__main__':

    # TEST CASES

    # Regular problem for testing
    # P = [[1, 1, -4, 0, 0, 0],
    #     [1, 1, 2, 1, 0, 0],
    #     [1, 1, -1, 0, 1, 0],
    #     [-1, 1, 1, 0, 0, 1]]
    
    # b = [9, 2, 4]
    
    # xb = [4, 5, 6]
    
    # Two phase problem for testing
    # P = [[6, 3, 0, 0, 0],
    #     [1, 1, -1, 0, 0],
    #     [2, -1, 0, -1, 0],
    #     [0, 3, 0, 0, 1]]
    
    # b = [1, 1, 2]
    
    # xb = [4, 5, 6]
    
    # P = [[-3, -2, -1, 0, 0],
    #     [3, -3, 2, 1, 0],
    #     [-1, 2, 1, 0, 1]]
    
    # b = [3, 6]
    
    # xb = [4, 5]
    
    # Edge case with 0/-ve in ratio test
    # P = [[-1, -3, 0, 0, 0],
    #     [1, -2, 1, 0, 0],
    #     [-2, 1, 0, 1, 0],
    #     [5, 3, 0, 0, 1]]
    
    # b = [0, 4, 15]
    
    # xb = [3, 4, 5]

    # Redundent constraint problem

    # P = [[19, 17, 23, 21, 25],
    #     [60, 25, 45, 20, 50],
    #     [10, 15, 45, 50, 40],
    #     [30, 60, 10, 30, 10],
    #     [1, 1, 1, 1, 1]]
    
    # b = [40, 35, 25, 1]

    # degenerate LP

    # P = [[-1, -1, -1, 0, 0],
    #      [1, 1, 0, 1, 0],
    #      [0, -1, 1, 0, 1]]
    
    # b = [8, 0]
    
    # PROJECT MODELS

    # Model 5
    #['x11', 'x21', 'x31', 'x41', 'x12', 'x22', 'x32', 'x42', 's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8']
    #[38.95, 53.95, 60.95, 29.95, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #[0, 0, 0, 0, 38.95, 53.95, 60.95, 29.95, 0, 0, 0, 0, 0, 0, 0, 0],

    # P = [[-0.82, -1.23, -1.19, -1.05, -0.32, -0.73, -0.69, -0.55, 0, 0, 0, 0, 0, 0, 0, 0],
    #      [43.43, 6.57, 11.42, 25.97, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0],
    #      [11.09, 35.8, 23.45, 37.7, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    #      [3.58, 0.88, 1.78, 2.68, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0],
    #      [0, 0, 0, 0, 11.09, 35.8, 24.7, 39.7, 0, 0, 0, -1, 0, 0, 0, 0],
    #      [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    #      [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    #      [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    #      [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1]]
    
    # b = [0, 0, 0, 0, 1500, 500, 1000, 2000]

    #Edited

    # P = [[-0.895, -1.305, -1.265, -1.125, -0.37, -0.78, -0.74, -0.6, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [43.419, 6.56, 11.41, 25.96, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0],
    #     [11.075, 35.785, 23.435, 37.685, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    #     [3.579, 0.879, 1.779, 2.679, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 11.075, 35.785, 23.435, 37.685, 0, 0, 0, -1, 0, 0, 0, 0],
    #     [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    #     [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    #     [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    #     [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1]]
    
    # b = [0, 0, 0, 0, 1500, 500, 1000, 2000]

    # EXTRA CREDIT: MODEL 22
    #[x111, x121, x131, x141, x151, x211, x221, x231, x241, x251, x311, x321, x331, x341, x351, x112, x122, x132, x142, x152, x212, x222, x232, x242, x252, x312, x322, x332, x342, x352, s1, s2, s3, s4]
    P = [[61, 72, 45, 55, 66, 69, 78, 60, 49, 56, 59, 66, 63, 61, 47, 58.5, 68.3, 47.8, 0, 63.5, 65.3, 74.8, 55.0, 49.0, 57.5, 0, 61.3, 63.5, 58.8, 50.0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0],
        [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 27.5, 30.3, 23.8, 0, 28.5, 29.3, 31.8, 27.0, 25.0, 26.5, 0, 28.3, 27.5, 26.8, 24.0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]]
    
    b = [10, 20, 15, 7, 11, 9, 10, 8, 675000, 0, 0]

    
    P = np.array(P)
    b = np.array(b)
    z = P[0, :]
    A = P[1:, :]
    # xb = np.array(xb)
    
    print("Input matrix A:{0}\nb:{1}".format(A,  b.T))

    f_tb, f_z_star, f_sol, f_xb = Phase_1(A, b, z)

    print("Final Optimal Tableau:", f_tb)

    print("Optimal Objective function value:", f_z_star)
    for i, j in zip(f_xb, f_sol):
        print("Var:{0} = {1}".format(i, j))
    print("Rest of the variables are non basic variables, therfore = 0")
