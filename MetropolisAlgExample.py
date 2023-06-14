import numpy as np
import math
import random
import matplotlib.pyplot as plt
import scipy.stats as stats



def p(a):
    return (1/math.sqrt(2*math.pi*3)) * math.exp(-0.5*(((a-2)**2)/3))

N = 1000000
omega = np.zeros(N)
omega[0] = 2
for i in range(1,N):
    proposed_omega = omega[i-1] + np.random.normal(0,1)
    A = min(1,p(proposed_omega)/p(omega[i-1]))
    U = random.uniform(0,1)
    if A>U:
        omega[i] = proposed_omega
    else:
        omega[i] = omega[i-1]

plt.hist(omega, density = True, bins = 30)

mu = 2
variance = 3
sigma = math.sqrt(variance)
x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
plt.plot(x, stats.norm.pdf(x, mu, sigma))
plt.title('N(2,3) density and draws using the Metropolis algorithm')
plt.xlabel('omega')
plt.ylabel('density')
plt.savefig('MHalg.png')
plt.show()


