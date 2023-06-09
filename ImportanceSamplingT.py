import numpy as np
import math

n_0 = 1000           #Number of draws from p_0
n_1 = 1000              #Number of draws from p_1
t = 15                 #Number of steps iterative method
m = 300                 #Number of estimates
s_0 = n_0/(n_0+n_1)
s_1 = n_1/(n_0+n_1)
nu_0 = 2               #Degrees of freedom dist 0
nu_1 = 3                #Degrees of freedom dist 1
r = np.zeros((m,t))
r[:,0] = 0.96                #Initial guess ratio normalizing constants
for a in range(m):
    
    w_0 = np.zeros(n_0)
    for j in range(n_0):
        w_0[j] = np.random.standard_t(nu_0)
        j += 1
        
    w_1 = np.zeros(n_1)
    for j in range(n_1):
        w_1[j] = np.random.standard_t(nu_1)
        j += 1
        
    for i in range(1,t):
        numerator = 0
        for j in range(n_0):
            numerator += (((1+(w_0[j]**2)/nu_1)**(-(nu_1+1)/2))/((1+(w_0[j])**2/nu_0)**(-(nu_0+1)/2)))/(r[a,i-1]*s_0+s_1*(((1+(w_0[j])**2/nu_1)**(-(nu_1+1)/2))/((1+(w_0[j])**2/nu_0)**(-(nu_0+1)/2))))
        numerator = (1/n_0)*numerator
        denumerator = 0
        for k in range(n_1):
            denumerator += 1/(r[a,i-1]*s_0+s_1*(((1+(w_1[j])**2/nu_1)**(-(nu_1+1)/2))/((1+(w_1[j])**2/nu_0)**(-(nu_0+1)/2))))
        denumerator = (1/n_1)*denumerator
        r[a,i] = numerator/denumerator
        i += 1

averaged_r = np.zeros(t)
for i in range(t):
    averaged_r[i] = sum(r[:,i])/m
    i+=1
    
lambdaa = math.log(averaged_r[-1])

