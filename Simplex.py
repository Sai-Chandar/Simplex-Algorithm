import numpy as np


def Phase_1():
    pass

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
    np.seterr(divide='ignore')
    t = b/pivot_col
    t[t<0] = np.Inf
    ix = np.argmin(t)
    return flag2, ix+2



def build_tableau(A, b, z, xb):
    m, n = A.shape
    tableau = np.zeros((m+2, n+2))
    tableau[0,:] = np.array(range(n+2))
    tableau[1, 1:n+1] = -z
    tableau[2:(m+2), 1:n+1] = A
    tableau[2:(m+2), -1] = b
    tableau[2:, 0] = xb
    return tableau, m, n

def Simplex(tableau, m, n):
    while 1:
        z = tableau[1, 1:n+1]
        xb = tableau[2:, 0]
        b = tableau[2:(m+2), -1]
        flag1, idx = Optimality_check(z, xb)
        if flag1 == True:
            print("Optimal Solution.")
            break
        pivot_col = tableau[2:, idx]
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

    z_star =  tableau[1, -1]
    sol = b
    xb = tableau[2:, 0] 

    return tableau, z_star, sol, xb




P = [[1, 1, -4, 0, 0, 0],
    [1, 1, 2, 1, 0, 0],
    [1, 1, -1, 0, 1, 0],
    [-1, 1, 1, 0, 0, 1]]

b = [9, 2, 4]

xb = [4, 5, 6]

P = np.array(P)
b = np.array(b)
z = P[0, :]
A = P[1:, :]
xb = np.array(xb)

print(A, "\n", b.T)