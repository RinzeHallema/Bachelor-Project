import numpy as np
from scipy.spatial.distance import euclidean
import math

def product(numbers):
    result = 1
    for num in numbers:
        result *= num
    return result


data = np.array([
    [0.72, 0.06, 0, 0, 0, 0],
    [0.86, 0.48, 0.61, 0, 0.02, 0],
    [0.63, 0, 0.40, 0, 0, 0.02],
    [0.26, 0.56, 0.57, 0, 0, 0],
    [0.3, 0.82, 0.29, 0, 0, 0],
    [0, 0.51, 0.16, 0.08, 0.07, 0]
])

known_indices = np.argwhere(data > 0)
zero_indices = np.argwhere(data == 0)
m = len(known_indices)
n = m + len(zero_indices)
grid_spacing = 0.2

x = 2.5
y = 2.5
loc = np.array([x,y]) 
K_tu = np.zeros(m)
for i, known_idx in enumerate(known_indices):
    distance = euclidean(loc, known_idx) * grid_spacing
    cov_ij = np.exp(-distance)
    K_tu[i] = cov_ij
    
K_tv = np.zeros(n-m)
for i, zero_idx in enumerate(zero_indices):
    distance = euclidean(loc, zero_idx) * grid_spacing
    cov_ij = np.exp(-distance)
    K_tv[i] = cov_ij
    
K_uu = np.zeros((m,m))
for i, known_idx in enumerate(known_indices):
    for j, known_idx2 in enumerate(known_indices):
        distance = euclidean(known_idx, known_idx2) * grid_spacing
        cov_ij = np.exp(-distance)
        K_uu[i,j] = cov_ij

K_uv = np.zeros((m, n-m))
for i, known_idx in enumerate(known_indices):
    for j, zero_idx in enumerate(zero_indices):
        distance = euclidean(known_idx, zero_idx) * grid_spacing
        cov_ij = np.exp(-distance)
        K_uv[i,j] = cov_ij

K_vu = np.zeros((n-m, m))
for i, known_idx in enumerate(known_indices):
    for j, zero_idx in enumerate(zero_indices):
        distance = euclidean(known_idx, zero_idx) * grid_spacing
        cov_ij = np.exp(-distance)
        K_vu[j,i] = cov_ij
        
K_vv = np.zeros((n-m, n-m))
for i, zero_idx in enumerate(zero_indices):
    for j, zero_idx2 in enumerate(zero_indices):
        distance = euclidean(zero_idx, zero_idx2) * grid_spacing
        cov_ij = np.exp(-distance)
        K_vv[i,j] = cov_ij
        
K_aa = np.array([1])
K_ab_list = []

for i, zero_idx in enumerate(zero_indices):
    K_ab = np.zeros((1, m + i))                                                # Making the covariances matrices K_ab

    for j, known_idx in enumerate(known_indices):                              # Fills the covariances between the zero point and the known points
        distance = euclidean(zero_idx, known_idx) * grid_spacing
        cov_ij = np.exp(-distance)
        K_ab[0, j] = cov_ij

    for j in range(i):                                                         # Fills the covariances between the zero point and the earlies zero points
        distance = euclidean(zero_idx, zero_indices[j]) * grid_spacing
        cov_ij = np.exp(-distance)
        K_ab[0, m + j] = cov_ij

    K_ab_list.append(K_ab)

K_ba_list = [np.transpose(i) for i in K_ab_list]                               # Creates all K_ba's
    
K_bb_list = []

for i, zero_idx in enumerate(zero_indices):                                    # Making the sizes of the K_bb arrays
    K_bb_list.append(np.zeros((m+i,m+i)))

for i, zero_idxextra in enumerate(zero_indices):                               # Filling the the matrices with the positive data points covariances
    for j, known_idx in enumerate(known_indices):    
        for k, known2_idx in enumerate(known_indices):
            distance = euclidean(known_idx, known2_idx) * grid_spacing
            cov_ij = np.exp(-distance)
            K_bb_list[i][j,k] = cov_ij
        for l in range(i):
            distance = euclidean(zero_indices[l],known_idx) * grid_spacing
            K_bb_list[i][m+l, j] = np.exp(-distance)
            K_bb_list[i][j, m+l] = np.exp(-distance)
    for j in range(i):
        for k in range(i):
            distance = euclidean(zero_indices[j], zero_indices[k]) * grid_spacing
            K_bb_list[i][m+j,m+k] = np.exp(-distance)

#### Method 1
            
N = 100
mu_list = np.zeros((N,n-m))
sigma_list = np.zeros((N,n-m))
v_list = np.zeros((N,n-m))
known_data_points = data[known_indices[:, 0], known_indices[:, 1]]
for i in range(N):
    for j, zero_idx in enumerate(zero_indices):
        K_tu = np.zeros(m)
        for k, known_idx in enumerate(known_indices):
            distance = euclidean(zero_idx, known_idx) * grid_spacing
            cov_ij = np.exp(-distance)
            K_tu[k] = cov_ij
        mu_list[i][j] = np.matmul(np.matmul(K_tu, np.linalg.inv(K_uu)), known_data_points)
        sigma_list[i][j] = K_aa - np.matmul(np.matmul(K_tu, np.linalg.inv(K_uu)),np.transpose(K_tu))
        v_list[i][j] = np.random.normal(mu_list[i][j],math.sqrt(sigma_list[i][j]))

### Check whether there are vectors v smaller or equal than 0. 

posv = 0
for i in range(N):
    if v_list[i].max() < 0:
        posv += 1

q = 0
for i in range(N):
    for j in range(n-m):
        if v_list[i][j] <= 0:
            q += 1
            

