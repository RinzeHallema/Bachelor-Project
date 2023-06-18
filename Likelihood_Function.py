import numpy as np
from scipy.spatial.distance import euclidean
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import math
from scipy.stats import norm
import random


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
m_0 = -0.07
c_0 = 0.631
m_max_likelihood = -0.07
c_max_likelihood = 0.54
    
grid_size = 21
m_values = np.linspace(-2, 2, grid_size)
logc_values = np.linspace(-2, 1, grid_size)
m_grid, logc_grid = np.meshgrid(m_values, logc_values)

#Firstly we compute the likelihood for the reference model

K_aa = np.array([1])
K_ab_list = []
            
for i, zero_idx in enumerate(zero_indices):
    K_ab = np.zeros((1, m + i))                                                # Making the covariances matrices K_ab
            
    for e, known_idx in enumerate(known_indices):                              # Fills the covariances between the zero point and the known points
        distance = euclidean(zero_idx, known_idx) * grid_spacing
        cov_ij = c_0*np.exp(-distance)
        K_ab[0, e] = cov_ij
                
    for e in range(i):                                                         # Fills the covariances between the zero point and the earlies zero points
        distance = euclidean(zero_idx, zero_indices[e]) * grid_spacing
        cov_ij = c_0*np.exp(-distance)
        K_ab[0, m + e] = cov_ij
    K_ab_list.append(K_ab)
            
K_ba_list = [np.transpose(i) for i in K_ab_list]                               # Creates all K_ba's
K_bb_list = []
            
for i, zero_idx in enumerate(zero_indices):                                    # Making the sizes of the K_bb arrays
    K_bb_list.append(np.zeros((m+i,m+i)))
            
for i, zero_idxextra in enumerate(zero_indices):                               # Filling the the matrices with the positive data points covariances
    for e, known_idx in enumerate(known_indices):    
        for k, known2_idx in enumerate(known_indices):
            distance = euclidean(known_idx, known2_idx) * grid_spacing
            cov_ij = c_0*np.exp(-distance)
            K_bb_list[i][e,k] = cov_ij
        for l in range(i):
            distance = euclidean(zero_indices[l],known_idx) * grid_spacing
            K_bb_list[i][m+l, e] = c_0*np.exp(-distance)
            K_bb_list[i][e, m+l] = c_0*np.exp(-distance)
    for j in range(i):
        for k in range(i):
            distance = euclidean(zero_indices[j], zero_indices[k]) * grid_spacing
            K_bb_list[i][m+j,m+k] = c_0*np.exp(-distance)

N = 10
mu = np.zeros((N,n-m))
sigma = np.zeros((N,n-m))
v = np.zeros((N,n-m))
prod_list = np.zeros(n-m)
U = np.zeros((N,n-m))
d_0 = np.zeros(N)

for j in range(N):
    known_data_points = data[known_indices[:, 0], known_indices[:, 1]]
    for q in range(n-m):
        vectorm_0 = m_0*np.ones(known_data_points.size)
        mu[j][q] = m_0 + np.matmul(K_ab_list[q],np.matmul(np.linalg.inv(K_bb_list[q]),(known_data_points-vectorm_0)))
        sigma[j][q] =  K_aa - np.matmul(np.matmul(K_ab_list[q], np.linalg.inv(K_bb_list[q])), K_ba_list[q])
        U[j][q] = random.uniform(0,1)
        v[j][q] = mu[j][q] + math.sqrt(np.abs(sigma[j][q]))*norm.ppf(U[j][q]*norm.cdf(-mu[j][q]/math.sqrt(np.abs(sigma[j][q]))))
        known_data_points = np.append(known_data_points, v[j][q])
        prod_list[q] = norm.cdf(-mu[j][q]/math.sqrt(np.abs(sigma[j][q])))
    d_0[j] = product(prod_list)
mean = m_0 * np.ones(m)
p_theta_0 = multivariate_normal(mean, K_bb_list[0])
p_theta_0_u = p_theta_0.pdf(data[known_indices[:, 0], known_indices[:, 1]])
sumdjref = sum(d_0)




### thetas for the contour plot


N = 10
d = np.zeros([N, 21,21])
p_theta = np.zeros((21,21))
p_theta_u = np.zeros((21,21))
for a,b in enumerate(m_values):
    for c,z in enumerate(logc_values):
        for j in range(N):
        
            K_aa = np.array([1])
            K_ab_list = []
            
            for i, zero_idx in enumerate(zero_indices):
                K_ab = np.zeros((1, m + i))                                                # Making the covariances matrices K_ab
            
                for e, known_idx in enumerate(known_indices):                              # Fills the covariances between the zero point and the known points
                    distance = euclidean(zero_idx, known_idx) * grid_spacing
                    cov_ij = np.exp(z)*np.exp(-distance)
                    K_ab[0, e] = cov_ij
            
                for e in range(i):                                                         # Fills the covariances between the zero point and the earlies zero points
                    distance = euclidean(zero_idx, zero_indices[e]) * grid_spacing
                    cov_ij = np.exp(z)*np.exp(-distance)
                    K_ab[0, m + e] = cov_ij
            
                K_ab_list.append(K_ab)
            
            K_ba_list = [np.transpose(i) for i in K_ab_list]                               # Creates all K_ba's
                
            K_bb_list = []
            
            for i, zero_idx in enumerate(zero_indices):                                    # Making the sizes of the K_bb arrays
                K_bb_list.append(np.zeros((m+i,m+i)))
            
            for i, zero_idxextra in enumerate(zero_indices):                               # Filling the the matrices with the positive data points covariances
                for e, known_idx in enumerate(known_indices):    
                    for k, known2_idx in enumerate(known_indices):
                        distance = euclidean(known_idx, known2_idx) * grid_spacing
                        cov_ij = np.exp(z)*np.exp(-distance)
                        K_bb_list[i][e,k] = cov_ij
                    for l in range(i):
                        distance = euclidean(zero_indices[l],known_idx) * grid_spacing
                        K_bb_list[i][m+l, e] = np.exp(z)*np.exp(-distance)
                        K_bb_list[i][e, m+l] = np.exp(z)*np.exp(-distance)
                for l in range(i):
                    for k in range(i):
                        distance = euclidean(zero_indices[l], zero_indices[k]) * grid_spacing
                        K_bb_list[i][m+l,m+k] = np.exp(z)*np.exp(-distance)
            mu = np.zeros((N,n-m))
            sigma = np.zeros((N,n-m))
            v = np.zeros((N,n-m))
            prod_list = np.zeros(n-m)
            known_data_points = data[known_indices[:, 0], known_indices[:, 1]]
            for q in range(n-m):
                vectorb = b*np.ones(known_data_points.size)
                mu[j][q] = b + np.matmul(K_ab_list[q],np.matmul(np.linalg.inv(K_bb_list[q]),(known_data_points-vectorb)))
                sigma[j][q] =  K_aa - np.matmul(np.matmul(K_ab_list[q], np.linalg.inv(K_bb_list[q])), K_ba_list[q])
                v[j][q] = mu[j][q] + math.sqrt(np.abs(sigma[j][q]))*norm.ppf(U[j][q]*norm.cdf(-mu[j][q]/math.sqrt(np.abs(sigma[j][q]))))
                prod_list[q] = norm.cdf(-mu[j][q]/math.sqrt(np.abs(sigma[j][q])))
                known_data_points = np.append(known_data_points, v[j][q])
            d[j][a][c] = product(prod_list)
            est_likelihood_theta = np.zeros((21,21))
            mean = b * np.ones(m)
            p_theta = multivariate_normal(mean, K_bb_list[0])
            p_theta_u[a][c] = p_theta.pdf(data[known_indices[:, 0], known_indices[:, 1]])

maxlikelihood = p_theta_0_u*sum(d_0)
Bayes_Factor = np.zeros((21,21))
dnew = np.zeros((21,21))
likelihood = np.zeros((21,21))
for a in range(21):
    for b in range(21):
        for j in range(N):
            dnew[a][b] += d[j][a][b]
        likelihood[a][b] = p_theta_u[a][b]*dnew[a][b]
        Bayes_Factor[a][b] = likelihood[a][b] / maxlikelihood


loc_best_model_from_grid = np.argwhere(Bayes_Factor == np.max(Bayes_Factor))
np.argwhere(Bayes_Factor > 1)
#Scaled likelihood
likelihood = likelihood/np.max(likelihood)
