# -*-coding:utf-8 -*-


import numpy as np
import itertools


def robust_holds(A, S1, S2, r, s):
    isRSRobust = False

    def calc_reachable(r, S):
        ss = 0
        for i in S:
            t = 0
            for j in range(A.shape[0]):
                if A[j][i-1] != 0 and (j+1) not in S:
                    t += 1
            if t >= r:
                ss += 1
        return ss

    if calc_reachable(r, S1) == len(S1) \
            or calc_reachable(r, S2) == len(S2) \
            or calc_reachable(r, S1) + calc_reachable(r, S2) >= s:
            isRSRobust = True

    return isRSRobust


def nonempty_subset(l):
    ll = len(l)/2
    for i in range(1, len(l)/2+1):
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

    return isRSRobust, None, None


if __name__ == '__main__':
    with open('data3.in', 'r') as f:
        A = []
        for line in f.readlines():
            A.append(line.split(' '))
        print check_robustness(np.array(A, dtype=np.int), 3, 3)





