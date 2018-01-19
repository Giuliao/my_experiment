from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import consensus_algo
import CheckRobustness
import numpy as np
import pandas as pd
import time
import multiprocessing
import os
import json
import traceback
import signal
import networkx as nx


def print_dict(local_dict):
    for k in local_dict:
        print(k, '=>', local_dict[k],)


def set_labels():
    r_1_train_data = pd.read_csv("./data/r_5_train_data.csv", header=0, index_col=0)
    labels = r_1_train_data.iloc[:, -1]
    r_1_train_data = r_1_train_data.iloc[:, : -1]
    df = pd.DataFrame(np.zeros((labels.shape[0], 50)), dtype=np.int)

    for row in range(labels.shape[0]):
        df.iloc[row][labels[row]-1] = 1

    r_1_train_data.join(df).to_csv("./data/r_5_train_data.csv")


def set_s_labels(file_name, local_index, n_class, new_n_class):
    train_data = pd.read_csv(file_name, header=0, index_col=0, dtype=np.int)  # remember setting the same dtype!!!
    train_data = train_data.iloc[:, :-n_class].reset_index(drop=True)
    old_labels = (train_data.iloc[:, -n_class:]).reset_index(drop=True)
    old_labels.columns = range(n_class)
    new_labels = pd.DataFrame(np.zeros((old_labels.shape[0], new_n_class), dtype=np.int))

    for row in range(old_labels.shape[0]):
        index = np.argmax(old_labels.iloc[row, :])
        if index >= 12:
            index -= 12
        elif index >= 6:
            index -= 6
        # print(index)
        new_labels.iloc[row][index] = 1

    # print(new_labels)
    train_data.join(new_labels).to_csv("./data/r_"+str(local_index+1)+"_train_data_with_s.csv")
    # pd.concat([train_data, new_labels], axis=1)
    return


def set_new_label(file_name, local_index, r, s, n_class, new_n_class):
    local_dict = {}
    for i in range(r):
        for j in range(s):
            local_dict[i*s+j] = i*(s+1) + j

    print_dict(local_dict)

    train_data = pd.read_csv(file_name, header=0, index_col=0, dtype=np.int)  # remember setting the same dtype!!!
    old_labels = (train_data.iloc[:, -n_class:]).reset_index(drop=True)
    train_data = train_data.iloc[:, :-n_class].reset_index(drop=True)
    old_labels.columns = range(n_class)
    new_labels = pd.DataFrame(np.zeros((old_labels.shape[0], new_n_class), dtype=np.int))

    # print(old_labels)
    for row in range(old_labels.shape[0]):
        index = np.argmax(old_labels.iloc[row, :].values)
        index = local_dict[index]
        print(index)
        new_labels.iloc[row][index] = 1

    # print(new_labels)
    train_data.join(new_labels).to_csv("./data/node_6/r_"+str(local_index)+"_train_data_with_dis_5.csv")
    # pd.concat([train_data, new_labels], axis=1)
    return


def set_r_labels(file_name, local_index, n_class, new_n_class):
    """
    :param file_name: 
    :param local_index: start from 0, the index of r
    :param n_class: 
    :param new_n_class: 
    :return: 
    """
    r_train_data = pd.read_csv(file_name, header=0, index_col=0)
    r_train_data = r_train_data.iloc[:, :-n_class]
    labels = pd.DataFrame(np.zeros((r_train_data.shape[0], new_n_class)), dtype=np.int)
    for row in range(labels.shape[0]):
        labels.iloc[row][local_index] = 1

    r_train_data.join(labels).to_csv("./data/r_"+str(local_index+1)+"_train_data_new.csv")
    return


def set_origin_labels(file_name, local_index, n, n_class):
    df = pd.read_csv(file_name, index_col=0, header=0, dtype=np.int)
    train_data = df.iloc[:, :-n_class].reset_index(drop=True)
    labels = df.iloc[:, -n_class:].reset_index(drop=True)
    print(labels.head())
    new_label = pd.DataFrame(np.zeros((labels.shape[0], 2), dtype=np.int))
    import math
    r = int(math.ceil(n/2.0))
    s = n
    local_dict = {}
    for i in range(r):
        for j in range(s):
            local_dict[i*n+j] = (i+1, j+1)

    for row in range(labels.shape[0]):
        # print(labels.loc[row])
        index = labels.loc[row].values.argmax()
        # print(index)
        new_label.iloc[row][0], new_label.iloc[row][1] = local_dict[index]

    final_df = train_data.join(new_label)
    print(final_df.head())
    print(final_df.tail())
    final_df.to_csv('./data/r_%d_train_origin_data.csv' % local_index)

# mlist=[]
# index = [0 for i in range(24)]
# tt = pd.DataFrame()
# ddict = {}
# tt_dict = {}
# dd_count = 1
# for i in range(1, 3):
#     for j in range(1, 5):
#         tt_dict[(i, j)] = dd_count
#         dd_count += 1
#
#
# def generate(deep):
#     if deep >= 16:
#         adj1 = np.array(index[:-8])
#         # print(adj1)
#         local_tuple = CheckRobustness.determine_robustness(adj1.reshape([4, 4]))
#         if local_tuple in tt_dict:
#             index[-tt_dict[local_tuple]] = 1
#             global tt
#             tt = pd.concat([tt, pd.DataFrame(np.array(index)).transpose()])
#             index[-tt_dict[local_tuple]] = 0
#
#         if local_tuple in ddict:
#             ddict[local_tuple] += 1
#         else:
#             ddict[local_tuple] = 1
#
#         mlist.append(index)
#         return
#
#     if deep == 0 or deep == 5 or deep == 10 or deep == 15:
#         index[deep] = 0
#         generate(deep + 1)
#     else:
#         index[deep] = 0
#         generate(deep + 1)
#         index[deep] = 1
#         generate(deep + 1)
#
#
# def permutate_4_node():
#     generate(0)
#     print(tt.shape)
#     tt.to_csv("./data/node_4/all_data_with_label.csv")
#     # print_dict(ddict)
#     # print(len(mlist))


def init_worker():
    """https://stackoverflow.com/questions/1408356/
        keyboard-interrupts-with-pythons-multiprocessing-pool#comment12678760_6191991
    
    :return: 
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def init_file_variables(write_file, node_num, file_name, threshold):
    """ initialize some parameters
    :param write_file: 
    :param node_num: 
    :param file_name: 
    :param threshold: 
        the threshold of number of different graph
    :return: 
        r_s_is_count: record the number of graph is more than threshold 
        df: data frame read from a file
        is_used: whether the matrix used 
        local_dict: count the number of every r_s
        count: the total number of r_s_is_count 
    """

    r_s_is_count = {}
    count = 0
    labels_size = 1 # attention here need to change
    labels_feature = ['k'] # attention here need to change

    # make sure there is the json file
    if os.path.exists(write_file.format(node_num, file_name, 'json')):
        with open(write_file.format(node_num, file_name, 'json'), 'r') as f:
            # the dict will save  (r, s) => count
            local_dict = json.loads(f.read())
            print('local dict restored finished')

        for rr_s in local_dict.keys():
            if local_dict[rr_s] >= threshold:
                if rr_s not in r_s_is_count:
                    r_s_is_count[rr_s] = 1
                    count += 1
    else:
        # the dict will save  (r, s) => count,
        # if not exiting, init it
        local_dict = {}

    # make sure there is the csv file
    if os.path.exists(write_file.format(node_num, file_name, 'csv')):
        # the df save all adjmatrixs which will reshape to a vector
        df = pd.read_csv(write_file.format(node_num, file_name, 'csv'), index_col=0)
        # is_used is a numpy array for deduplicating
        is_used = df.iloc[:, :-labels_size].values
        print('is_used restored finished')

    else:
        # features as columns to format the csv file
        features = [str(q) for q in range(node_num**2)]
        features.extend(labels_feature)
        df = pd.DataFrame(columns=features)
        is_used = None

    return r_s_is_count, df, is_used, local_dict, count


def init_directory(node_num, directed):
    if directed:
        write_file = "./data/k_connectivity/directed/node_{0}/"
    else:
        write_file = "./data/k_connectivity/undirected/node_{0}/"

    # make sure there exists the directory
    if not os.path.exists(write_file.format(node_num)):
        os.mkdir(write_file.format(node_num))

    write_file += '{1}.{2}'

    return write_file


def fine_tune(p, node_num, count):

    try:
        for i in range(1, node_num):
            if count == node_num-1:
                1/0
            elif count == i:
                p += 1/node_num

        if p > 1:
            p = 0.99

    except ZeroDivisionError as e:
        raise e

    return p


def data_generate(node_num, file_name, directed=False):
    """ data generate by multiprocess
    :param node_num: 
        the number of node
    :param file_name: 
        file name must be 'r_1' like
    :param directed: 
        if True it's a directed graph
    :return: 
        see the output data in ./data/undirected or ./data/directed/
    """
    start_time = time.time()

    # initialize
    threshold = 200
    write_file = init_directory(node_num, directed)
    r_s_is_count, df, is_used, local_dict, count = init_file_variables(write_file, node_num, file_name, threshold)
    data_features = [str(i) for i in range(node_num ** 2)]
    p = 0.3  # the probability of the binominal graph
    pool = None
    # node_connectivity = local_wrapper(nx.node_connectivity)
    print('init finish..')

    try:
        while 1:
            pool = multiprocessing.Pool(4, init_worker)
            result = []
            network_objects = []
            for k in range(16):
                mm = consensus_algo.NetworkAlgo(node_num, p, directed=directed)
                adjmatrix = mm.adjMatrix
                if is_used is not None and any((adjmatrix.reshape((-1, node_num**2)) == x).all() for x in is_used):
                    # print('continue...')
                    continue
                else:
                    if is_used is None:
                        is_used = adjmatrix.reshape((-1, node_num**2))
                    else:
                        is_used = np.vstack((is_used, adjmatrix.reshape((-1, node_num**2))))

                    network_objects.append(pd.DataFrame(adjmatrix.reshape(-1, node_num ** 2),
                                                        columns=data_features))
                # print(k)
                # result.append(pool.apply_async(CheckRobustness.determine_robustness_multi_process, (adjmatrix,)))
                result.append(pool.apply_async(nx.node_connectivity, (mm.G,)))

            pool.close()
            pool.join()

            for r, tt in zip(result, network_objects):
                r_s = r.get()

                # if r_s[0] != int(file_name.split('_')[1]):
                #     continue
                if r_s == 0:
                    continue

                rr_s = 'k_'+str(r_s)
                if rr_s in local_dict:
                    if local_dict[rr_s] >= threshold:
                        if rr_s not in r_s_is_count:
                            r_s_is_count[rr_s] = 1
                            count += 1
                        continue
                    local_dict[rr_s] += 1       
                else:
                    local_dict[rr_s] = 1      

                # df = df.append(tt.join(pd.DataFrame(np.array(r_s, dtype=np.int).reshape(-1, 2), columns=['r', 's'])))
                df = df.append(tt.join(pd.DataFrame(np.array(r_s, dtype=np.int).reshape(-1, 1), columns=['k'])))
            print_dict(local_dict)
            print(count)
            print('*'*75)
            
            # del pool  # del will delete the variable completely, no None here
            del network_objects
            del result

            # modify the probability according to the count
            p = fine_tune(p, node_num, count)

    except Exception as e:
        traceback.print_exc()
        # for the case keyboard interrupt
        if pool is not None:
            pool.terminate()
            pool.join()
    finally:
        # for the case no keys
        if len(local_dict.keys()) == 0:
            pass
        else:
            df.to_csv(write_file.format(node_num, file_name, 'csv'))
            with open(write_file.format(node_num, file_name, 'json'), 'wr') as f:
                f.write(json.dumps(local_dict))
        print('finished, time:', time.time()-start_time)


def data_generate_test():
    from CheckRobustness import determine_robustness2
    my_dict = {}
    is_used = None
    while 1:
        mm = consensus_algo.NetworkAlgo(vertex_num=5, p=0.38, directed=True)
        adjmatrix = mm.adjMatrix
        node_num = mm.vertex_num
        if is_used is not None and any((adjmatrix.reshape((-1, node_num ** 2)) == x).all() for x in is_used):
            # print('continue...')
            continue
        else:
            if is_used is None:
                is_used = adjmatrix.reshape((-1, node_num ** 2))
            else:
                is_used = np.vstack((is_used, adjmatrix.reshape((-1, node_num ** 2))))

        kk = determine_robustness2(mm.adjMatrix)
        if kk not in my_dict:
            my_dict[kk] = 1
        else:
            my_dict[kk] += 1
        print_dict(my_dict)
        print('*' * 75)
        del mm


def local_wrapper(func):
    def inner(**kwargs):
        adjmatrix = kwargs.get('adjmatrix', None)
        network_G = kwargs.get('G', None)
        print("hell0")
        func(network_G)
        return adjmatrix, func
    return inner


def k_connectivity_data_generate(directed=False):

    pool = multiprocessing.Pool(4)
    result = []
    objs = []
    node_connectivity = local_wrapper(nx.node_connectivity)
    for k in range(16):
        mm = consensus_algo.NetworkAlgo(vertex_num=5, p=0.5, directed=directed)
        objs.append(mm)
        result.append(pool.apply_async(node_connectivity, {'G': mm.G, 'adjmatrix': mm.adjMatrix}))

    pool.close()
    pool.join()

    # for r in result:
    #     print(r.get())
    print('*'*75)

    # for k in objs:
    #     print(nx.node_connectivity(k.G))

if __name__ == '__main__':
    for i in range(10, 21):
        data_generate(i, 'k_{0}'.format(i), True)