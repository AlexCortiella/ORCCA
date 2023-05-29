import numpy as np


##### ALGORITHM 1 (Full C + backward pass v1) #####
def knapSack_DP1(W, wt, vals):
    
    n = len(vals)
    Z = [[0 for x in range(W + 1)] for x in range(n + 1)]
    Wid = []
    # Build table Z[][] in bottom up manner
    for i in range(n + 1):
        for w in range(W + 1):
            if i == 0 or w == 0:#initialize table (zero items)
                Z[i][w] = 0
            elif wt[i-1] <= w:
                    Z[i][w] = max(vals[i-1] + Z[i-1][w-wt[i-1]], Z[i-1][w])
            else:
                Z[i][w] = Z[i-1][w]
                
    # stores the result of Knapsack
    res = Z[n][W]     
    w = W
    for i in range(n, 0, -1):
        if res <= 0:
            break
        # either the result comes from the
        # top (K[i-1][w]) or from (val[i-1]
        # + K[i-1] [w-wt[i-1]]) as in Knapsack
        # table. If it comes from the latter
        # one/ it means the item is included.
        if res == Z[i - 1][w]:
            continue
        else:
 
            # This item is included.
            Wid.append(wt[i - 1])
             
            # Since this weight is included
            # its value is deducted
            res = res - vals[i - 1]
            w = w - wt[i - 1]
        
    ids = [i for i in range(n) if wt[i] in Wid]
    return Z[n][W], ids

##### ALGORITHM 2 (DP3 book) #####
def KS_DP_base(W, wt, vals, n):
    
    z = [0 for d in range(W+1)]
    r = [0 for d in range(W+1)]
    for j in range(n):
        wj = wt[j]
        pj = vals[j]
        for d in reversed(range(wj, W+1)):
            if z[d - wj] + pj > z[d]:
                z[d] = z[d - wj] + pj
                r[d] = j
    
    return z[W], r[W]

def knapSack_DP2(W, wt, vals):

    n = len(vals)
    Xs = []
    Z = []
    Wb = W
    nb = n
    while Wb > 0 and nb > 0:
        z, r = KS_DP_base(Wb, wt[0:nb], vals[0:nb], nb)
        Xs.append(r)
        Z.append(z)
        nb = r
        Wb = Wb - wt[r]
    return Z[0], Xs
    
    
##### ALGORITHM 3 (Full C + backward pass v2) #####

def dynamic_downlink_scheduling(W, wt, vals, n):

    C = [[0 for x in range(W + 1)] for x in range(n + 1)]
    Wid = []
    # Build table C[][] in bottom up manner
    for i in range(n + 1):
        for w in range(W + 1):
            if i == 0 or w == 0:#initialize table (zero items)
                C[i][w] = 0
            elif wt[i-1] <= w:
                    C[i][w] = max(vals[i-1] + C[i-1][w-wt[i-1]], C[i-1][w])
            else:
                C[i][w] = C[i-1][w]
    return C
    
def downlink_data_sequence(C, wt, n, W):
    ids = []
    while n > 0 and W > 0:
        
        if C[n][W] > C[n-1][W]:
            ids.append(n-1)
            W = W - wt[n-1]
        n = n - 1
    return ids

def knapSack_DP3(W, wt, vals):

    n = len(vals)
    ## Build value matrix
    C = dynamic_downlink_scheduling(W, wt, vals, n)
    ## Compute items selected
    ids = downlink_data_sequence(C, wt, n, W)

    return C[n][W], ids

#### ALGORITHM 4 (Tara's implementation) ####

def knapSack_DP4(BW, wt, val):

    n=len(val)
    saved_vals = [None]*(n+1)
    allroots = [None]*(n+1)

    for i in range(n+1):
        if i == 0:
            allroots[n] = [BW]
            prev_root_array = [BW]
        else:
            num_root = len(prev_root_array)*2
            roots = []  #Check this!
            for itr in range (num_root):
                if itr%2 == 0:
                    root = prev_root_array[int(itr/2)] - wt[n-i]
                else:
                    root = prev_root_array[int(itr/2)]

                if root > 0:
                    roots.append(root)
            #roots.sort()
            allroots[n-i] = roots
            prev_root_array = roots
            
    K = [[0 for x in range(BW+1)] for y in range(2)]
 
    # We know we are always using the  current row or
    # the previous row of the array/vector . Thereby we can
    # improve it further by using a 2D array but with only
    # 2 rows i%2 will be giving the index inside the bounds
    # of 2d array K
    for i in range(n + 1): 
        for w in range(BW + 1): 
            if i == 0 or w == 0: 
                K[1 - i%2][w] = 0
            elif wt[i-1] <= w: 
                K[i%2 - 1][w] = max(val[i-1] + K[1- (i-1)%2][w-wt[i-1]],  K[1-(i-1)%2][w]) 
            else: 
                K[i%2 - 1][w] = K[1- (i-1)%2][w]
                        
        for num in range(len(allroots[i])):
            W = int(allroots[i][num])
            if num == 0:
                saved_vals[i] = [K[1 - i%2][W]]  
            else:
                saved_vals[i] += [K[1 - i%2][W]]
    Opt_Val = K[1 - n%2][BW]

    Opt_Seq = sequence_of_data(wt, BW, n, allroots, saved_vals)
    
    return Opt_Val, Opt_Seq

def sequence_of_data(wt, W, n, allroots, saved_vals):
    Opt_Seq = []
    while n>0 and W>0:
        idx = [i for i in range(len(allroots[n])) if allroots[n][i] == W][0]
        idx_previous = [i for i in range(len(allroots[n-1])) if allroots[n-1][i] == W][0]
        if saved_vals[n][idx] > saved_vals[n-1][idx_previous]:
            Opt_Seq.append(n-1)
            W = W - wt[n-1]
        n = n-1
    return Opt_Seq

#### ALGORITHM 5 (knapsack_hirschberg) ####
def optimal_cost(val, wt, W):
     
    # array to store final result
    # dp[i] stores the profit with KnapSack capacity "i"
    n = len(val)
    dp = [0]*(W+1);
    dp_idx = [-1]*(W+1);
 
    # iterate through all items
    for i in range(n):
         
        #traverse dp array from right to left
        for j in range(W,0,-1):
            if wt[i] <= j and dp[j] < val[i] + dp[j-wt[i]]:
                dp[j] = val[i] + dp[j-wt[i]]
                dp_idx[j] = i
             
    '''above line finds out maximum of dp[j](excluding ith element value)
    and val[i] + dp[j-wt[i]] (including ith element value and the
    profit with "KnapSack capacity - ith element weight") *'''
    return dp, dp_idx
    
def knapSack_DP5(W, wt, val, offset = 0):
    
    n = len(val)

    #Divide
    mid = int(n/2)
    
    dp1, idx1 = optimal_cost(val[0:mid], wt[0:mid], W)
    dp2, idx2 = optimal_cost(val[mid:], wt[mid:], W)
    
    b1 = -1
    b2 = -1
    for i in range(W+1):
        if dp1[i] + dp2[W-i] > b1:
            b1 = dp1[i] + dp2[W-i]
            b2 = max(b2, i)

    max_val = b1

    #Conquer
    solution = []
    if idx1[b2] != -1:
        iChosen = idx1[b2]
        _, subsolution1 = knapSack_DP5(b2 - wt[iChosen], wt[0:iChosen], val[0:iChosen], offset)
        solution += subsolution1
        solution.append(idx1[b2] + offset)
    if idx2[W - b2] != -1:
        iChosen = mid + idx2[W - b2]
        _, subsolution2 = knapSack_DP5(W - b2 - wt[iChosen], wt[mid:iChosen], val[mid:iChosen], offset + mid)
        solution += subsolution2
        solution.append(iChosen + offset)
        
    return max_val, solution