import numpy as np
from itertools import product
N = 4
hfield = 2
basis = list(product([-1, 1], repeat=N))

print('Generated %d basis functions' % (len(basis)))
# print(len(basis_functions))

# list(permutations([0,1,0,0]))
H = np.zeros((2 ** N, 2 ** N))
for H_i in range(2 ** N):
    for H_j in range(2 ** N):
        H_sum = 0
        for i in range(N):
            if H_i == H_j:
                if i == N - 1:
                    H_sum -= basis[H_j][i] * basis[H_j][0]
                else:
                    H_sum -= basis[H_j][i] * basis[H_j][i + 1]

            sj = list(basis[H_j])
            sj[i] *= -1
            if H_i == basis.index(tuple(sj)):
                H_sum -= hfield

        H[H_i, H_j] = H_sum

print('Ground state energy:', np.min(np.linalg.eigvals(H)) / N)