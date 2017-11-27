# -*-coding:utf-8 -*-
# data can be viewed in assets/1.png:
# 0 0 0 0 1 1 1 0
# 0 0 0 1 1 1 0 1
# 0 0 0 1 1 0 1 1
# 0 1 1 0 0 0 0 1
# 1 1 1 0 0 1 1 1
# 1 1 0 0 1 0 1 1
# 1 0 1 0 1 1 0 1
# 0 1 1 1 1 1 1 0

# result:
# no remove, (3, 1)-robust
# if remove node1, (3, 1)-robust
# if remove node2, (2, 3)-robust
# if remove node3, (2, 7)-robust
# if remove node4, (3, 1)-robust
# if remove node5, (2, 2)-robust
# if remove node6, (2, 2)-robust
# if remove node7, (2, 3)-robust
# if remove node8, (2, 1)-robust
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
                if A[j][i-1] != 0 and (j+1) not in S:
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
    ll = len(l)//2
    for i in range(1, ll+1):
        for S1 in itertools.combinations(l, i):
            yield S1, tuple([val for val in l if val not in S1])


def check_robustness(A, r, s):
    isRSRobust = True
    n = A.shape[0]
    for k in range(2, n+1):
        for K in itertools.combinations(range(1, n+1), k):
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
    r = min(r, int(math.ceil(A.shape[0]*1.0/2.0)))
    s = A.shape[0]
    n = A.shape[0] # the number of vertex
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
                    return pd.DataFrame(A.reshape(-1, n*n), columns=[str(i) for i in range(n*n)]), (r, s)
    return pd.DataFrame(A.reshape(-1, n*n), columns=[str(i) for i in range(n*n)]), (r, s)

def determine_robustness(A):
    """the func is the same as determine_robustness
    :param A: adjmatrix
    :return: 
    """
    r = min(calc_in_degree(A))
    r = min(r, int(math.ceil(A.shape[0]*1.0/2.0)))
    s = A.shape[0]
    n = A.shape[0] # the number of vertex
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
    flag = 1    # set the upper bound of the K set
    if i:
        A = np.delete(np.delete(A, i-1, 0), i-1, 1)  # !!!this part is different from the paper
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


if __name__ == '__main__':
    import consensus_algo
    #
    # pool = multiprocessing.Pool(processes=5)
    #
    # ll = [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1,
    #     0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0,
    #     1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1,
    #     1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0,
    #     1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1,
    #     1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0,
    #     1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1,
    #     1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1,
    #     1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0,
    #     1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1,
    #     0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1,
    #     1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1,
    #     1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0]
    #
    #
    # adjmatrix = np.array(ll)
    # adjmatrix= adjmatrix.reshape((13, 13))
    # print(adjmatrix)
    # mm = consensus_algo.NetworkAlgo(adjMatrix=adjmatrix)
    # result = []
    # print(mm.adjMatrix)
    # mm.show_network()
    # # print(determine_partial_robust(mm.adjMatrix, 13))
    # # mm.show_network()
    # # print(determine_robustness2(mm.adjMatrix))
    # import time
    # start = time.time()
    # for i in range(1, 14):
    #     print("hello %d" %(i))
    #     result.append(pool.apply_async(determine_partial_robust, (mm.adjMatrix, i)))
    # pool.close()
    # pool.join()
    # final_result = []
    # for r in result:
    #     final_result.append(r.get())
    # print(final_result)
    # r, s = min(final_result)
    # print(r, s)
    # print(check_robustness(mm.adjMatrix, r, s))
    # print('multi-process time:', time.time()-start)
    # start = time.time()
    # r, s = determine_robustness(mm.adjMatrix)
    # print(r, s)
    # print(check_robustness(mm.adjMatrix, r, s))
    # print('one process time', time.time()-start)
    #

    mm = consensus_algo.NetworkAlgo(vertex_num=10, p=0.95)
    kk = determine_robustness2(mm.adjMatrix)
    del mm
    while kk != (5, 5):
        mm = consensus_algo.NetworkAlgo(vertex_num=10, p=0.92)
        kk = determine_robustness2(mm.adjMatrix)
        print('2->', kk)
        kk = determine_robustness(mm.adjMatrix)
        print('1->', kk)
        if kk == (5, 4):
            print(check_robustness(mm.adjMatrix, 5, 5))


        del mm


