from __future__ import print_function
from __future__ import division
import consensus_algo
import CheckRobustness
import numpy as np
import pandas as pd
import time
import math
import traceback
import signal


def print_dict(local_dict):
    for k in local_dict:
        print(k, '=>', local_dict[k],)


def directed_data_generate(node_number):

    const_node = node_number  # the number of vertex set
    l2 = []  # record the matrix generated and in order to check duplicate matrix
    local_dict = {}  # map the (1, 1) -> 1 and the like
    count_s_dict = {}  # record the different number of s
    count_r_dict = {}  # record the different number of r
    ll_r = {}  # record every the number of every s part
    r_scale = int(math.ceil(const_node/2.0))
    const_combine = const_node * r_scale
    s_part_number = 300

    # record every local count
    record_local_count = {}
    control_factorize = const_combine - r_scale - 1

    # in every r part, the total number of data, briefly, s_number* s_part_number
    total_number = const_node*s_part_number

    # calc the map relation between (1,1) -> 1 and the like
    tmp_count = 1
    for i in range(1, r_scale+1):
        count_r_dict[i] = total_number
        ll_r[i] = pd.DataFrame()
        for j in range(1, const_node+1):
            local_dict[(i, j)] = tmp_count
            count_s_dict[tmp_count] = s_part_number
            tmp_count += 1

    start = time.time()
    try:
        while 1:
            adjmatrix = consensus_algo.NetworkAlgo(vertex_num=const_node).adjMatrix

            if adjmatrix.tolist() in l2:
                continue
            else:
                l2.append(adjmatrix.tolist())

            y = CheckRobustness.determine_robustness(adjmatrix)
            tmp_adjmatrix = (adjmatrix.reshape([1, adjmatrix.shape[0]**2]).tolist())[0]
            local_label = [0 for i in range(const_combine)]

            for p in range(2, r_scale):
                if y[0] == p and count_r_dict[p] > 0 and count_s_dict[local_dict[y]] > 0:
                    count_s_dict[local_dict[y]] -= 1
                    local_label[local_dict[y]-1] = 1
                    tmp_adjmatrix.extend(local_label)
                    ll_r[p] = pd.concat([ll_r[p], pd.DataFrame(tmp_adjmatrix).transpose()])
                    count_r_dict[p] -= 1
                    print("count_r_%d = %d" % (p, count_r_dict[p]))
                    print_dict(count_s_dict)
                    break

            # print(adjmatrix)
            # print(y)
            local_count = 0

            for p in range(1, const_combine):
                if count_s_dict[p] == 0:
                    local_count += 1

            if local_count == const_combine:
                break
            elif local_count > control_factorize:
                if local_count not in record_local_count:
                    record_local_count[local_count] = 1
                else:
                    record_local_count[local_count] += 1

                if record_local_count[local_count] > 300:
                    1/0

    except Exception as e:
        print(traceback.print_exc())
    finally:
        # for p in range(2, r_scale):
        #     ll_r[p].to_csv(("./data/node_{}/r_{}_train_data.csv".format(const_node, p)))
        print("epoch %f" % (time.time()-start))


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


def undirected_data_generate(node_num, file_name):
    import multiprocessing
    import os
    import json

    start_time = time.time()
    r_s_is_count = {}
    count = 0
    if os.path.exists('./data/undirected/node_{0}/{1}.json'.format(node_num, file_name)):
        with open('./data/undirected/node_{0}/{1}.json'.format(node_num, file_name), 'r') as f:
            local_dict = json.loads(f.read())

        for rr_s in local_dict.keys():
            if local_dict[rr_s] > 500:
                if rr_s not in r_s_is_count:
                    r_s_is_count[rr_s] = 1
                    count += 1
    else:
        local_dict = {}

    if os.path.exists('./data/undirected/node_{0}/{1}.csv'.format(node_num, file_name)):
        df = pd.read_csv('./data/undirected/node_{0}/{1}.csv'.format(node_num, file_name), index_col=0)
        is_used = df.iloc[:, :-2].values

    else:

        features = [str(q) for q in range(node_num**2)]
        features = features.extend(['r', 's'])
        df = pd.DataFrame(columns=features)
        is_used = None

    p = 0.93
    pool = None

    try:
        while 1:
            pool = multiprocessing.Pool(4, init_worker)
            result = []
            for k in range(16):
                adjmatrix = consensus_algo.NetworkAlgo(node_num, p).adjMatrix
                # https://stackoverflow.com/questions/14766194/testing-whether-a-numpy-array-contains-a-given-row
                if is_used is not None and any((adjmatrix.reshape((-1, node_num**2)) == x).all() for x in is_used):
                    # print('continue...')
                    continue
                else:
                    if is_used is None:
                        is_used = adjmatrix.reshape((-1, node_num**2))
                    else:
                        is_used = np.vstack((is_used, adjmatrix.reshape((-1, node_num**2))))
                # print(k)
                result.append(pool.apply_async(CheckRobustness.determine_robustness_multi_process, (adjmatrix,)))

            pool.close()
            pool.join()

            for r in result:
                tt, r_s = r.get()

                if r_s[0] != int(file_name.split('_')[1]):
                    continue
                rr_s = str(r_s)
                if rr_s in local_dict:
                    if local_dict[rr_s] > 500:
                        if rr_s not in r_s_is_count:
                            r_s_is_count[rr_s] = 1
                            count += 1
                        continue
                    local_dict[rr_s] += 1
                else:
                    local_dict[rr_s] = 1
                df = df.append(tt.join(pd.DataFrame(np.array(r_s, dtype=np.int).reshape(-1, 2), columns=['r', 's'])))
            print_dict(local_dict)
            print(count)
            print('*'*100)
            del pool
            del result
            if 3 <= count < 5:
                p = 0.95
            elif 5 <= count < 7:
                p = 0.99
            elif count == 7:
                1/0

    except Exception as e:
        print(traceback.print_exc())
        if pool is not None:
            pool.terminate()
            pool.join()
    finally:
        df.to_csv("./data/undirected/node_{0}/{1}.csv".format(node_num, file_name))
        with open("./data/undirected/node_{0}/{1}.json".format(node_num, file_name), 'wr') as f:
            f.write(json.dumps(local_dict))
        print('finished, time:', time.time()-start_time)

if __name__ == '__main__':
    undirected_data_generate(10, 'r_5')