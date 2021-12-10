import numpy as np
 
 
 
def Phase_1(A, b, z):
    m, n = A.shape
    I = np.identity(m)
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
        return 0
    # Removing the artificial variables 
    artf_idx = list(range(n+m+2))
    del artf_idx[-(m+1):-1]
    tb = tb[:, artf_idx]
    # Redundent Constraints
    l = [np.where(xb==i)[0] for i in xb1]
    for i in l:
        np.delete(tb, i, axis = 0)    
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
 
 
 
def Optimality_check(z, xb):
    idx = np.argmax(z)
    if z[idx] == 0:
        # Have to add the multiple solutions case, use xb
        return True, idx+1
    return False, idx+1
    
def ratio_test(b, pivot_col):
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
        z = tableau[1, 1:n+1]
        xb = tableau[2:, 0]
        b = tableau[2:(m+2), -1]
        flag1, idx = Optimality_check(z, xb)
        if flag1 == True:
            if opt_flag == True:
                print("Optimal Solution.")
            break
        pivot_col = np.copy(tableau[2:, idx])
        flag2, ix = ratio_test(b, pivot_col)
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
P = [[-1, -3, 0, 0, 0],
    [1, -2, 1, 0, 0],
    [-2, 1, 0, 1, 0],
    [5, 3, 0, 0, 1]]
 
b = [0, 4, 15]
 
xb = [3, 4, 5]
 
 
P = np.array(P)
b = np.array(b)
z = P[0, :]
A = P[1:, :]
xb = np.array(xb)
 
print(A, "\n", b.T)