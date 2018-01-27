# -*-coding:utf-8 -*-
from __future__ import print_function
import numpy as np
import itertools
import math
import pandas as pd
import multiprocessing


def robust_holds(A, S1, S2, r, s):
    """
     :param A: the adjMatrix
     :param S1: the subset of vertex set
     :param S2: the subset of verex set
     :param r:  (r, s)-robust
     :param s:  (r, s)-robust
     :return: 
         boolean
     """

    def calc_reachable(S):
        ss = 0
        # iterate the node(>0) in S
        for i in S:
            t = 0  # count for the node that owns the neighbors outside S
            for j in range(A.shape[0]):
                if A[j][i - 1] != 0 and (j + 1) not in S:
                    t += 1
            if t >= r:
                ss += 1
        return ss

    isRSRobust = False
    if calc_reachable(S1) == len(S1) \
            or calc_reachable(S2) == len(S2) \
            or calc_reachable(S1) + calc_reachable(S2) >= s:
        isRSRobust = True

    return isRSRobust


def nonempty_subset(l):
    ll = len(l) // 2
    for i in range(1, ll + 1):
        for S1 in itertools.combinations(l, i):
            yield S1, tuple([val for val in l if val not in S1])


def check_robustness(A, r, s):
    isRSRobust = True
    n = A.shape[0]
    for k in range(2, n + 1):
        for K in itertools.combinations(range(1, n + 1), k):
            for S1, S2 in nonempty_subset(K):
                if not robust_holds(A, S1, S2, r, s):
                    isRSRobust = False
                    return isRSRobust, S1, S2

    return isRSRobust


def calc_in_degree(local_A):
    for i in range(local_A.shape[0]):
        count = 0
        for j in range(local_A.shape[0]):
            if local_A[j][i] != 0:
                count += 1
        yield count


def determine_robustness_multi_process(A):
    r = min(calc_in_degree(A))
    r = min(r, int(math.ceil(A.shape[0] * 1.0 / 2.0)))
    s = A.shape[0]
    n = A.shape[0]  # the number of vertex
    for k in range(2, n + 1):
        for K in itertools.combinations(range(1, n + 1), k):
            for S1, S2 in nonempty_subset(K):
                isRSRoubst = robust_holds(A, S1, S2, r, s)
                if not isRSRoubst and s > 0:
                    s -= 1
                while (not isRSRoubst) and r > 0:
                    while (not isRSRoubst) and s > 0:
                        isRSRoubst = robust_holds(A, S1, S2, r, s)
                        if not isRSRoubst:
                            s -= 1
                    if not isRSRoubst:
                        r -= 1
                        s = n
                if r == 0:
                    return pd.DataFrame(A.reshape(-1, n * n), columns=[str(i) for i in range(n * n)]), (r, s)
    return pd.DataFrame(A.reshape(-1, n * n), columns=[str(i) for i in range(n * n)]), (r, s)


def determine_robustness(A):
    """the func is the same as determine_robustness
    :param A: adjmatrix
    :return: 
    """
    r = min(calc_in_degree(A))
    r = min(r, int(math.ceil(A.shape[0] * 1.0 / 2.0)))
    s = A.shape[0]
    n = A.shape[0]  # the number of vertex
    for k in range(2, n + 1):
        for K in itertools.combinations(range(1, n + 1), k):
            for S1, S2 in nonempty_subset(K):
                isRSRoubst = robust_holds(A, S1, S2, r, s)
                if not isRSRoubst and s > 0:
                    s -= 1
                while (not isRSRoubst) and r > 0:
                    while (not isRSRoubst) and s > 0:
                        isRSRoubst = robust_holds(A, S1, S2, r, s)
                        if not isRSRoubst:
                            s -= 1
                    if not isRSRoubst:
                        r -= 1
                        s = n
                if r == 0:
                    return r, s
    return r, s


def determine_robustness2(A):
    return determine_partial_robust2(A, 0)


def determine_partial_robust2(A, i):
    """ this func totally delete one node from the original matrix 
        then calc the r, s
    :param A: adjmatrix
    :param i: i is more than 0, and i is the excepted node
    :return: 
        r, s
    """
    flag = 1  # set the upper bound of the K set
    if i:
        A = np.delete(np.delete(A, i - 1, 0), i - 1, 1)  # !!!this part is different from the paper
        flag = 0
    # print A
    r = min(calc_in_degree(A))
    r = min(r, int(math.ceil(A.shape[0] * 1.0 / 2.0)))
    s = A.shape[0]
    n = A.shape[0]  # the number of vertex

    # partition the set with k nodes, k at least 2 because of for 1-robust
    for k in range(2, n + flag):
        # combinition set of which the number is k
        for K in itertools.combinations([t for t in range(1, n + 1) if t != i], k):
            # subsets of K both of which are nonempty
            for S1, S2 in nonempty_subset(K):
                isRSRoubst = robust_holds(A, S1, S2, r, s)
                if not isRSRoubst and s > 0:
                    s -= 1
                while (not isRSRoubst) and r > 0:
                    while (not isRSRoubst) and s > 0:
                        isRSRoubst = robust_holds(A, S1, S2, r, s)
                        if not isRSRoubst:
                            s -= 1
                    if not isRSRoubst:
                        r -= 1
                        s = n
                if r == 0:
                    return r, s
    return r, s


def determine_partial_robust(A, i):
    """ this func means partition i as the independent subset, 
        not delete the i from the matrix 
    :param A: adjmatrix
    :param i: i is more than 0, and i is the excepted node
    :return: 
        r, s
    """
    r = min(calc_in_degree(A))
    r = min(r, int(math.ceil(A.shape[0] * 1.0 / 2.0)))
    s = A.shape[0]
    n = A.shape[0]  # the number of vertex

    # partition the set with k nodes, k at least 2 because of for 1-robust
    for k in range(2, n):
        # combinition set of which the number is k
        for K in itertools.combinations([t for t in range(1, n + 1) if t != i], k):
            # subsets of K both of which are nonempty
            for S1, S2 in nonempty_subset(K):
                isRSRoubst = robust_holds(A, S1, S2, r, s)
                if not isRSRoubst and s > 0:
                    s -= 1
                while (not isRSRoubst) and r > 0:
                    while (not isRSRoubst) and s > 0:
                        isRSRoubst = robust_holds(A, S1, S2, r, s)
                        if not isRSRoubst:
                            s -= 1
                    if not isRSRoubst:
                        r -= 1
                        s = n
                if r == 0:
                    return r, s
    return r, s


def print_dict(local_dict):
    for k in local_dict.keys():
        print(k, '=>', '|' * local_dict[k], local_dict[k])


local_dict = {}
def recursive_cal(adjmatrix, p, q, count):

    for i in range(p, adjmatrix.shape[0]):
        if i == p:
            k = q + 1
        else:
            k = 0
        for j in range(k, adjmatrix.shape[1]):
            if i != j:
                adjmatrix[i][j] = 0
                if count == 100:
                    # print(adjmatrix)
                    kk = determine_robustness(adjmatrix)
                    # print(kk)
                    if kk not in local_dict:
                        local_dict[kk] = 1
                    else:
                        local_dict[kk] += 1
                    print_dict(local_dict)
                    print('*' * 75)
                else:
                    recursive_cal(adjmatrix, i, j, count+1)
                adjmatrix[i][j] = 1

if __name__ == '__main__':
    import consensus_algo
    import time
    # local_dict = {}

    # while True:
    #     start = time.time()
    #     mm = consensus_algo.NetworkAlgo(vertex_num=15, p=0.92)
    #     kk = determine_robustness2(mm.adjMatrix)
    #     end = time.time()-start
    #     print(end)
    #     if kk not in local_dict:
    #         local_dict[kk] = 1
    #     else:
    #         local_dict[kk] += 1
    #     print_dict(local_dict)
    #     print('*' * 75)
    #     del mm
    def swap_col(aa, col1, col2):
        if col1 == col2:
            return
        tmp = np.copy(aa[:, col1])
        aa[:, col1] = aa[:, col2]
        aa[:, col2] = tmp
        del tmp

    def swap_row(aa, row1, row2):
        if row1 == row2:
            return
        tmp = np.copy(aa[row1, :])
        aa[row1, :] = aa[row2, :]
        aa[row2, :] = tmp
        del tmp

    is_used = None

    def permutation(adj, index):
        if index == adj.shape[0]:
            global is_used
            if is_used is not None and any((adj.reshape((-1, adj.shape[0] ** 2)) == x).all() for x in is_used):
                # print(adj.reshape((-1, adj.shape[0]**2)))
                return
            else:
                if is_used is None:
                    is_used = np.copy(adj.reshape(-1, adj.shape[0]**2))
                else:
                    is_used = np.vstack((is_used, adj.reshape(-1, adj.shape[0]**2)))
            print(determine_robustness(adj))
            print('-'*20, is_used.shape)

        else:
            for i in range(index, adj.shape[0]):
                swap_col(adj, index, i)
                swap_row(adj, index, i)
                #print('swap',index, i)
                # print(adj)
                permutation(adj, index+1)
                swap_row(adj, index, i)
                swap_col(adj, index, i)

    local_dict = {}
    while True:
        mm = consensus_algo.NetworkAlgo(vertex_num=5, p=0.85)
        adjmatrix = mm.adjMatrix
        kk = determine_robustness(adjmatrix)
        # print(kk)
        # if kk not in local_dict:
        #     local_dict[kk] = 1
        # else:
        #     local_dict[kk] += 1
        # print_dict(local_dict)
        # print('*'*75)
        if kk == (3, 1):
            permutation(adjmatrix, 0)