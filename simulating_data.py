# -*- coding: utf-8 -*-

import random, csv, time, numpy as np, pandas as pd

# how do surrogate models work
# simulate 10,000 data points
# store the training data
# 80% used for training, 20% for testing
# learn more about surrogate models in practice, start from the first three articles
# implement the fully connected neural network

# phi_0 ranges from -200 to 200
# chi_0 ranges from 0 to pi
# P_0 ranges from 0.1 to 1
# benchmark the runtime

# random, time, csv
# must truly understand what a surrogate model is

t1 = time.time()

# Ramdomly generating 10,000 phi_0, chi_0 and P_0.

def generating(n):
    gen_phi_0 = np.random.uniform(-200,200,n)
    gen_chi_0 = np.random.uniform(0,np.pi,n)
    gen_P_0 = np.random.uniform(0.1,1,n)
    return gen_phi_0, gen_chi_0, gen_P_0

# Computing QU for each randomly generated data point.
def simulate_QU(phi_0, chi_0, P_0, lambda2):
    
    Q = P_0*np.cos(2*(phi_0*lambda2+chi_0))
    U = P_0*np.sin(2*(phi_0*lambda2+chi_0))
    
    return Q.tolist(), U.tolist()

def main():
    # Defining constants.
    const_c = 3e8
    N = 512 
    n = 10000 # Number of random samples generated.
    
    # Make data regularly spaced in frequency:
    nu_min = 0.58e9
    nu_max = 2.50e9
    nu = np.linspace(nu_min, nu_max, N)
    lambda2 = (const_c/nu)**2
    
    # Generating data.
    gen_phi_0, gen_chi_0, gen_P_0 = generating(n)

    with open('test_train_data.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["phi_0", "chi_0", "P_0", "Q", "U"])
        for i in range(n):
            Q, U = simulate_QU(gen_phi_0[i], gen_chi_0[i], gen_P_0[i],lambda2)
            writer.writerow([gen_phi_0[i], gen_chi_0[i], gen_P_0[i], Q, U])
    
    
    results = pd.read_csv('test_train_data.csv')
    print("Number of lines present: ", len(results))
    
    t2 = time.time()
    print("Runtime (seconds): ", t2-t1)


if __name__ == "__main__":
    main()





