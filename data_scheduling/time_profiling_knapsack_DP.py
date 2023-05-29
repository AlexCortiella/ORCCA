import numpy as np
import matplotlib.pyplot as plt
import time


from knapsack_DP import knapSack_DP1, knapSack_DP2, knapSack_DP3, knapSack_DP4, knapSack_DP5

### TIME PROFILING ###
num_cases = 4
item_case = [10**i for i in range(1,num_cases+1)]
max_val = 100
max_weight = 100

time_case_v1 = []
time_case_v2 = []
time_case_v3 = []
time_case_v4 = []
time_case_v5 = []

for i, n_items in enumerate(item_case):
    print(f'Running case {i+1}: {n_items} items')
    vals = list(np.random.randint(0, max_val, n_items))
    wt = list(np.random.randint(1, max_weight, n_items))
    W = 5*max_weight
    
    start = time.time()
    opt_val, isel = knapSack_DP1(W, wt, vals)
    time_case_v1.append(time.time() - start)
    
    start = time.time()
    opt_val, isel = knapSack_DP2(W, wt, vals)
    time_case_v2.append(time.time() - start)
    
    start = time.time()
    opt_val, isel = knapSack_DP3(W, wt, vals)
    time_case_v3.append(time.time() - start)    
    
    #start = time.time()
    #opt_val, isel = knapSack_DP4(W, wt, vals)
    #time_case_v4.append(time.time() - start)

    start = time.time()
    opt_val, isel = knapSack_DP5(W, wt, vals)
    time_case_v5.append(time.time() - start) 

plt.loglog(item_case, time_case_v1, 'ro-', label = 'DP1')
plt.loglog(item_case, time_case_v2, 'go-', label = 'DP2')
plt.loglog(item_case, time_case_v3, 'bo-', label = 'DP3')
#plt.loglog(item_case, time_case_v4, 'mo-', label = 'DP4')
plt.loglog(item_case, time_case_v5, 'co-', label = 'DP5')

plt.xlabel('# items')
plt.ylabel('Time [s]')
plt.legend()     

plt.savefig('time_profiling_knapsack_DP.png')

