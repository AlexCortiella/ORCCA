import numpy as np
import matplotlib.pyplot as plt
from memory_profiler import memory_usage
import time

from knapsack_DP import knapSack_DP1, knapSack_DP2, knapSack_DP3, knapSack_DP4, knapSack_DP5

def main():
    num_cases = 4
    item_case = [10**i for i in range(1,num_cases+1)]
    max_val = 100
    max_weight = 100

    mem_case_v1 = []
    mem_case_v2 = []
    mem_case_v3 = []
    mem_case_v4 = []
    mem_case_v5 = []

    ### MEMORY USAGE ####

    for i, n_items in enumerate(item_case):
        print(f'Running case {i+1}: {n_items} items')
        vals = list(np.random.randint(0, max_val, n_items))
        wt = list(np.random.randint(1, max_weight, n_items))
        W = 5*max_weight
        
        mu1 = max(memory_usage((knapSack_DP1, (W, wt, vals), {})))
        mem_case_v1.append(mu1)
            
        mu2 = max(memory_usage((knapSack_DP2, (W, wt, vals), {})))
        mem_case_v2.append(mu2)
        
        mu3 = max(memory_usage((knapSack_DP3, (W, wt, vals), {})))
        mem_case_v3.append(mu3)
    
        #mu4 = max(memory_usage((knapSack_DP4, (W, wt, vals),{})))
        #mem_case_v4.append(mu4)

        mu5 = max(memory_usage((knapSack_DP5, (W, wt, vals),{})))
        mem_case_v5.append(mu5)

    plt.plot(item_case, mem_case_v1, 'ro-', label = 'DP1')
    plt.plot(item_case, mem_case_v2, 'go-', label = 'DP2')
    plt.plot(item_case, mem_case_v3, 'bo-', label = 'DP3')
    #plt.plot(item_case, mem_case_v4, 'mo-', label = 'DP4')
    plt.plot(item_case, mem_case_v5, 'co-', label = 'DP5')

    plt.xlabel('# items')
    plt.ylabel('Peak memory [MiB]')
    plt.legend()     

    plt.savefig('memory_profiling_knapsack_DP.png')

if __name__ == "__main__":
    main()
