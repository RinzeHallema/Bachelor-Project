import numpy as np
import matplotlib.pyplot as plt
import math

nu0 = 2
nu1 = 3

def nu(t):
    return t*nu0 + (1-t)*nu1

n = 100000
lambdaa = np.zeros(n)
for i in range(n):
    t = np.random.uniform()
    w = np.random.standard_t(nu(t))
    lambdaa[i] = ((w**2*(nu(t)+1))/(2*nu(t)**2*(w**2/nu(t)+1)) - math.log(w**2/nu(t)+1)/2)*(nu1-nu0)
    i += 1
lambdahat = sum(lambdaa)/n
print(lambdahat)

## Theoretical lambda

theolambda = math.log((math.gamma((nu0+1)/2)/(math.sqrt(nu0*math.pi)*math.gamma(nu0/2)))/(math.gamma((nu1+1)/2)/(math.sqrt(nu1*math.pi)*math.gamma(nu1/2))))
print(theolambda)

## Plot simulated lambdas
plt.hist(lambdaa, density = True, bins = 100)
plt.xlim(-1,0.5)
plt.axvline(x = theolambda, color = 'r')
plt.title('Histogram of the estimated path sampling estimators')
plt.xlabel('Estimate')
plt.ylabel('Probability')

