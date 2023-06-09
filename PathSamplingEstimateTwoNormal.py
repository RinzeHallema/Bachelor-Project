import math
import numpy as np
import matplotlib.pyplot as plt

mu0 = 0
mu1 = 5
sigma0 = 1
sigma1 = 1
R = math.sqrt(((mu0-mu1)/2)**2+3/2*(sigma1**2+sigma0**2)+9/4*((sigma1**2-sigma0**2)/(mu1-mu0))**2)
C = (mu0+mu1)/2 + 3/2*(sigma1**2-sigma0**2)/(mu1-mu0)
phi0 = math.atanh((mu0-C)/R)
phi1 = math.atanh((mu1-C)/R)

def mu(x):
    return R*math.tanh(phi0*(1-x)+phi1*x) + C
def sigma(x):
    return (R/math.sqrt(3)) * (1/math.cosh(phi0*(1-x)+phi1*x))
def dmu(x):
    return R*(phi1-phi0)*(1/(math.cosh(phi0*(1-x)+phi1*x)**2))
def dsigma(x):
    return (R/math.sqrt(3)) * (phi0-phi1) * (1/math.cosh(phi0*(1-x)+phi1*x)) * math.tanh(phi0*(1-x)+phi1*x)

n = 100000
lambdaa = np.zeros(n)
for i in range(n):
    t = np.random.uniform()
    w = np.random.normal(mu(t),sigma(t))
    lambdaa[i] = dmu(t) * (w - mu(t))/(sigma(t)**2) + dsigma(t) * ((w-mu(t))**2)/(sigma(t)**3)
    i += 1
lambdahat = sum(lambdaa)/n
print(lambdahat)

plt.hist(lambdaa, density = True, bins = 2000)
plt.axvline(x = 0, color = 'b')
plt.title('Histogram of the estimated path sampling estimators')
plt.xlabel('Estimate')
plt.ylabel('Probability')
print(math.sqrt(np.var(lambdaa)))

D = mu1 - mu0
error = math.sqrt(12)* (math.log(D/math.sqrt(12)+math.sqrt(1+(D**2)/12)))
print(error)