import numpy as np
from scipy.spatial.distance import euclidean
from scipy.stats import norm
import random
import math
import pandas as pd
import matplotlib.pyplot as plt

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


## Calculating a and b
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
        
block1 = K_uu - np.matmul(np.matmul(K_uv,np.linalg.inv(K_vv)),K_vu)
block2 = K_vv - np.matmul(np.matmul(K_vu,np.linalg.inv(K_uu)),K_uv)
a = np.matmul(np.matmul(K_tu,np.linalg.inv(block1)) + np.matmul(K_tv,np.matmul(np.matmul(-np.linalg.inv(block2),K_vu),np.linalg.inv(K_uu))),data[known_indices[:, 0], known_indices[:, 1]])
b = np.matmul(K_tu,np.matmul(np.matmul(-np.linalg.inv(block1),K_uv),np.linalg.inv(K_vv))) + np.matmul(K_tv,np.linalg.inv(block2))

N = 10000

mu_list = np.zeros((N,n-m))
sigma_list = np.zeros((N,n-m))
v_list = np.zeros((N,n-m))
d = np.zeros(N)
prod_list = np.zeros(n-m)

for j in range(N):
    known_data_points = data[known_indices[:, 0], known_indices[:, 1]]
    for i in range(n-m):
        mu_list[j][i] = np.matmul(np.matmul(K_ab_list[i], np.linalg.inv(K_bb_list[i])),known_data_points)
        sigma_list[j][i] = K_aa - np.matmul(np.matmul(K_ab_list[i], np.linalg.inv(K_bb_list[i])), K_ba_list[i])      #Overkill
        U = random.uniform(0,1)
        v_list[j][i] = mu_list[j][i] + math.sqrt(sigma_list[j][i])*norm.ppf(U*norm.cdf(-mu_list[j][i]/math.sqrt(sigma_list[j][i])))
        known_data_points = np.append(known_data_points, v_list[j][i])
        prod_list[i] = norm.cdf(-mu_list[j][i]/math.sqrt(sigma_list[j][i]))
    d[j] = product(prod_list)

mu_new_point = np.zeros(N)


for j in range(N):
    mu_new_point[j] = a + np.matmul(b, v_list[j])
    
def Method2_estimator(t):
    teller = 0
    noemer = 0
    for j in range(N):
        teller += d[j]*norm.cdf((t-mu_new_point[j])/math.sqrt(sigma_list[j][-1]))
        noemer += d[j]
    return teller/noemer
    
T = np.round(np.arange(-1.4,1.5,0.2), decimals = 1)
estimates = np.zeros(len(T))
for i,j in enumerate(T):
    estimates[i] = Method2_estimator(j)
estimates = np.round(estimates, decimals = 5)
Stein = np.array([0.00002, 0.00022, 0.00156, 0.00825, 0.03306, 0.10109, 0.23868, 0.44315, 0.66542, 0.8414, 0.94245, 0.98436, 0.99686, 0.99954, 0.99995])
Stein_formatted = [f"{val:.5f}" for val in Stein]

### Creating the first table

data = {'T': T, 'Stein': Stein, 'Estimates': estimates}
df = pd.DataFrame(data)

fig, ax = plt.subplots()
ax.axis('off')

table = ax.table(cellText=df.values,
                 colLabels=df.columns,
                 cellLoc='center',
                 loc='center',
                 bbox=[0, 0, 1, 1])

table.auto_set_font_size(False)
table.set_fontsize(12)

plt.savefig('table.jpg', format='jpg', dpi=300)












## Code for creating the second table for point B

x = 1.5
y = 2.1
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
        
block1 = K_uu - np.matmul(np.matmul(K_uv,np.linalg.inv(K_vv)),K_vu)
block2 = K_vv - np.matmul(np.matmul(K_vu,np.linalg.inv(K_uu)),K_uv)
a = np.matmul(np.matmul(K_tu,np.linalg.inv(block1)) + np.matmul(K_tv,np.matmul(np.matmul(-np.linalg.inv(block2),K_vu),np.linalg.inv(K_uu))),data[known_indices[:, 0], known_indices[:, 1]])
b = np.matmul(K_tu,np.matmul(np.matmul(-np.linalg.inv(block1),K_uv),np.linalg.inv(K_vv))) + np.matmul(K_tv,np.linalg.inv(block2))

N = 10000

mu_list = np.zeros((N,n-m))
sigma_list = np.zeros((N,n-m))
v_list = np.zeros((N,n-m))
d = np.zeros(N)
prod_list = np.zeros(n-m)

for j in range(N):
    known_data_points = data[known_indices[:, 0], known_indices[:, 1]]
    for i in range(n-m):
        mu_list[j][i] = np.matmul(np.matmul(K_ab_list[i], np.linalg.inv(K_bb_list[i])),known_data_points)
        sigma_list[j][i] = K_aa - np.matmul(np.matmul(K_ab_list[i], np.linalg.inv(K_bb_list[i])), K_ba_list[i])      #Overkill
        U = random.uniform(0,1)
        v_list[j][i] = mu_list[j][i] + math.sqrt(sigma_list[j][i])*norm.ppf(U*norm.cdf(-mu_list[j][i]/math.sqrt(sigma_list[j][i])))
        known_data_points = np.append(known_data_points, v_list[j][i])
        prod_list[i] = norm.cdf(-mu_list[j][i]/math.sqrt(sigma_list[j][i]))
    d[j] = product(prod_list)

mu_new_point = np.zeros(N)


for j in range(N):
    mu_new_point[j] = a + np.matmul(b, v_list[j])
    
def Method2_estimator(t):
    teller = 0
    noemer = 0
    for j in range(N):
        teller += d[j]*norm.cdf((t-mu_new_point[j])/math.sqrt(sigma_list[j][-1]))
        noemer += d[j]
    return teller/noemer
    
T = np.round(np.arange(-1.4,1.5,0.2), decimals = 1)
estimates = np.zeros(len(T))
for i,j in enumerate(T):
    estimates[i] = Method2_estimator(j)
estimates = np.round(estimates, decimals = 5)
Stein = np.array([0.00000, 0.00000, 0.0000, 0.00001, 0.00012, 0.00105, 0.00671, 0.03079, 0.10278, 0.25381, 0.47640, 0.70683, 0.87441, 0.96001, 0.99072])
Stein_formatted = [f"{val:.5f}" for val in Stein]

data = {'T': T, 'Stein': Stein, 'Estimates': estimates}
df = pd.DataFrame(data)

fig, ax = plt.subplots()
ax.axis('off')

table = ax.table(cellText=df.values,
                 colLabels=df.columns,
                 cellLoc='center',
                 loc='center',
                 bbox=[0, 0, 1, 1])

table.auto_set_font_size(False)
table.set_fontsize(12)

plt.savefig('table2.jpg', format='jpg', dpi=300)



    
    
    
    
    
 
   