# -*-coding:utf-8-*-
# reference
# [1] https://code.tutsplus.com/articles/introduction-to-parallel-and-concurrent-programming-in-python--cms-28612
# [2] https://www.oreilly.com/learning/python-cookbook-concurrency
# [3] http://python.jobbole.com/86181/
# [4] https://www.cnblogs.com/liuxiaowei/p/7462453.html
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from multiprocessing import Pool, Queue, Manager
from consensus_algo import NetworkAlgo
import os
import sys
import numpy as np
import networkx as nx
import pandas as pd
import traceback
import json
import time
import signal
import CheckRobustness


def print_dict(local_dict):
    for k in local_dict:
        print(k, '=>', local_dict[k], )


def init_directory(node_num, directed):
    if directed:
        write_file = "./data/non-isomorphism/directed/node_{0}/"
    else:
        write_file = "./data/non-isomorphism/undirected/node_{0}/"

    # make sure there exists the directory
    if not os.path.exists(write_file.format(node_num)):
        os.mkdir(write_file.format(node_num))

    write_file += '{1}.{2}'

    return write_file


def init_file_variables(label_features, write_dir, node_num, file_name, threshold):
    # make sure there is the json file
    if os.path.exists(write_dir.format(node_num, file_name, 'json')):
        with open(write_dir.format(node_num, file_name, 'json'), 'r') as f:
            # the dict will save  (r, s) => count
            local_dict = json.loads(f.read())

        print('=> local dict restore finished')
        r_s_is_count = {}
        count = 0
        for rr_s in local_dict.keys():
            # if local_dict[rr_s] >= threshold:
            #     if rr_s not in r_s_is_count:
            #         r_s_is_count[rr_s] = 1
            #         count += 1
            count += local_dict[rr_s]
        print('=> r_s_is_count restore finished')
        print('=> count restore finished')

    else:
        # the dict will save  (r, s) => count,
        # if not exiting, init it
        local_dict = {}
        r_s_is_count = {}
        count = 0
        print('=> local dict init finished')
        print('=> dict init finished')

    # make sure there is the csv file
    if os.path.exists(write_dir.format(node_num, file_name, 'csv')):
        # the df save all adjmatrix which will reshape to a vector
        df = pd.read_csv(write_dir.format(node_num, file_name, 'csv'), index_col=0)
        # is_used is a numpy array for deduplicating
        label_size = len(label_features)
        is_used = df.iloc[:, :-label_size].values
        print('=> matrix data frame restore finished')
        print('=> is_used restored finished')
    else:
        # features as columns to format the csv file
        features = [str(q) for q in range(node_num ** 2)]
        features.extend(label_features)
        df = pd.DataFrame(columns=features)
        is_used = None
        print('=> matrix data frame init finished')
        print('=> is_used init finished')

    print('=> init finished')

    return r_s_is_count, df, is_used, local_dict, count


def fine_tune(p, node_num, count):
    base = 0.3

    for i in range(1, node_num):
        p = base + count / node_num

    if p >= 1:
        p = 0.99

    return p


def process_produce(node_num, directed, is_used, p, que):
    print('-' * 75, 'produce working')
    try:
        for nx_obj in nx.read_graph6('./data/non-isomorphism/graph7c.g6'):
            # fine_tune(p, node_num)
            # mm = NetworkAlgo(node_num, p, directed=directed)
            mm = NetworkAlgo(nx_obj=nx_obj, directed=directed)
            adjmatrix = mm.adjMatrix
            if is_used is not None and any((adjmatrix.reshape((-1, node_num ** 2)) == x).all() for x in is_used):
                # print('continue...')
                continue
            else:
                # if is_used is None:
                #     is_used = adjmatrix.reshape((-1, node_num ** 2))
                # else:
                #     is_used = np.vstack((is_used, adjmatrix.reshape((-1, node_num ** 2))))
                is_used = None

                que.put(adjmatrix)
    except KeyboardInterrupt:
        print('=> producer %d in exception' % os.getpid())
        sys.exit(0)
    else:
        print('=> producer %d read finished and wait last 20 seconds' % os.getpid())
        time.sleep(20)
        os.kill(os.getppid(), signal.SIGINT)
        sys.exit(0)


def process_write(data_features, label_features,
                  node_num, local_dict, r_s_is_count, count, threshold,
                  df, write_file, file_name, que):
    print('-' * 75, 'writer working')
    print('=> init count', count)
    try:
        # if node_num % 2 == 1:
        #     if count == ((node_num + 1) // 2 - 1) * (node_num - 1) + node_num // 2 - 1:
        #         1 / 0
        # elif count == (node_num + 1) // 2 * (node_num - 1):  # if count == 4:
        #     1 / 0

        while True:
            r_s, adjmatrix = que.get()
            tt = pd.DataFrame(adjmatrix.reshape(-1, node_num ** 2), columns=data_features)
            rr_s = str(r_s)

            if rr_s in local_dict:
                # if local_dict[rr_s] >= threshold:
                #     if rr_s not in r_s_is_count:
                #         r_s_is_count[rr_s] = 1
                #         count += 1
                #     continue
                local_dict[rr_s] += 1
            else:
                local_dict[rr_s] = 1
            count += 1

            print_dict(local_dict)

            df = df.append(tt.join(
                pd.DataFrame(np.array(r_s, dtype=np.int).reshape(-1, len(label_features)), columns=label_features)))

            print(count)
            print('*' * 75)

    except ZeroDivisionError:
        df.to_csv(write_file.format(node_num, file_name, 'csv'))
        with open(write_file.format(node_num, file_name, 'json'), 'wr') as f:
            f.write(json.dumps(local_dict))
        print('=> writer %d write finished' % os.getpid())
        os.kill(os.getppid(), signal.SIGINT)
        print('=> writer %d send signal finished' % os.getpid())
        sys.exit(0)
    except KeyboardInterrupt:
        df.to_csv(write_file.format(node_num, file_name, 'csv'))
        with open(write_file.format(node_num, file_name, 'json'), 'wr') as f:
            f.write(json.dumps(local_dict))
        print('=>writer %d write finished' % os.getpid())

        sys.exit(0)


def process_compute(que1, que2):
    try:
        adjmatrix = que1.get()
        # k = nx.node_connectivity(local_G)
        k = CheckRobustness.determine_robustness(adjmatrix)
        # print(k)
        # if k[0] == 0:
        #     return

        que2.put([k, adjmatrix])
    except KeyboardInterrupt:
        print('=> computing node %d finished' % os.getpid())
        sys.exit(0)


def init_worker():
    """https://stackoverflow.com/questions/1408356/
        keyboard-interrupts-with-pythons-multiprocessing-pool#comment12678760_6191991

    :return: 
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def main(node_num, file_name, dircted):
    manager = Manager()
    que_adj = manager.Queue()
    que_df = manager.Queue()

    # initialize
    node_num = node_num
    file_name = file_name
    dircted = dircted

    data_features = [str(q) for q in range(node_num ** 2)]
    label_features = ['r', 's']
    threshold = 30
    init_p = 0.5

    write_file = init_directory(node_num, True)
    r_s_is_count, df, is_used, local_dict, count = init_file_variables(label_features, write_file, node_num, file_name,
                                                                       threshold)

    pool = Pool(4, init_worker)
    pool.apply_async(process_produce, args=(node_num, dircted, is_used, init_p, que_adj))
    pool.apply_async(process_write, args=(data_features, label_features,
                                          node_num, local_dict, r_s_is_count, count, threshold,
                                          df, write_file, file_name, que_df))

    try:
        while True:
            pool.apply_async(process_compute, (que_adj, que_df))

    except KeyboardInterrupt:
        # pool.close()
        print('=> in main exception stop')
        pool.terminate()
        pool.join()
        del pool


if __name__ == '__main__':
    main(7, 'r_{}'.format(7), True)
