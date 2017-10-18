from __future__ import print_function
from __future__ import division
import consensus_algo
import CheckRobustness
import numpy as np
import pandas as pd
import time
import math


def main():
    const_node = 6
    const_combine = int(math.ceil(const_node/2))*const_node
    ll_r3 = pd.DataFrame()
    ll_r2 = pd.DataFrame()
    ll_r1 = pd.DataFrame()

    l2 = []

    local_dict = {}
    count_s = {}

    count = 1
    for i in range(1, int(math.ceil(const_node/2))+1):
        for j in range(1, const_node+1):
            local_dict[(i, j)] = count
            count_s[count] = 300
            count += 1

    count_r1 = 1800
    count_r2 = 1800
    count_r3 = 1800

    start = time.time()
    try:
        while count_r1 or count_r2 or count_r3:
            adjmatrix = consensus_algo.NetworkAlgo.init_network(vertex_num=const_node)

            if adjmatrix.tolist() in l2:
                continue
            else:
                l2.append(adjmatrix.tolist())

            y = CheckRobustness.determine_robustness(adjmatrix)
            tmp_adjmatrix = (adjmatrix.reshape([1, adjmatrix.shape[0]**2]).tolist())[0]
            local_label = [0 for i in range(const_combine)]
            if y[0] == 3 and count_r3 != 0:
                if count_s[local_dict[y]] > 0:
                    count_s[local_dict[y]] -= 1
                    local_label[local_dict[y]-1] = 1
                    tmp_adjmatrix.extend(local_label)
                    ll_r3 = pd.concat([ll_r3, pd.DataFrame(tmp_adjmatrix).transpose()])
                    count_r3 -= 1

                    print("count_r3 = %d" % count_r3)
            elif y[0] == 2 and count_r2 != 0:
                if count_s[local_dict[y]] > 0:
                    count_s[local_dict[y]] -= 1
                    local_label[local_dict[y]-1] = 1
                    tmp_adjmatrix.extend(local_label)
                    ll_r2 = pd.concat([ll_r2, pd.DataFrame(tmp_adjmatrix).transpose()])
                    count_r2 -= 1
                    print("count_r2 = %d" % count_r2)
            elif y[0] == 1 and count_r1 != 0:
                if count_s[local_dict[y]] > 0:
                    count_s[local_dict[y]] -= 1
                    local_label[local_dict[y] - 1] = 1
                    tmp_adjmatrix.extend(local_label)
                    ll_r1 = pd.concat([ll_r1, pd.DataFrame(tmp_adjmatrix).transpose()])
                    count_r1 -= 1
                    print("count_r1 = %d" % count_r1)
            else:
                continue

            # print(adjmatrix)
            # print(y)

    except Exception as e:
        print(str(e))
    finally:
        ll_r3.to_csv('./data/node_6/r_3_train_data.csv')
        ll_r2.to_csv('./data/node_6/r_2_train_data.csv')
        ll_r1.to_csv('./data/node_6/r_1_train_data.csv')
        print("epoch %f" % (time.time()-start))


def test_data():
    a = np.load("./data/train_data.npy")
    dict = {}
    print(a[:, 1])
    for row in range(a.shape[0]):
        if not dict.has_key(a[row, 1]):
            dict[a[row, 1]] = 0
        else:
            dict[a[row, 1]] += 1

    for k, v in dict.iteritems():
        print(k, '=>', v)


def test_hvstack():
    r_1_train_data = np.load("./data/r_5_train_data.npy")
    r_1_ll = r_1_train_data[0, :-1][0].reshape(1, 100)
    r_1_ll = np.column_stack((r_1_ll, r_1_train_data[0, -1]))
    for i in range(1, r_1_train_data.shape[0]):
        tt = np.column_stack((r_1_train_data[i, :-1][0].reshape(1, 100), r_1_train_data[i, -1]))
        r_1_ll = np.row_stack((r_1_ll, tt))

    pd.DataFrame(r_1_ll).to_csv("./data/r_5_train_data.csv")

    print(r_1_ll.shape)


def set_labels():
    r_1_train_data = pd.read_csv("./data/r_5_train_data.csv", header=0, index_col=0)
    labels = r_1_train_data.iloc[:, -1]
    r_1_train_data = r_1_train_data.iloc[:, : -1]
    df = pd.DataFrame(np.zeros((labels.shape[0], 50)), dtype=np.int)

    for row in range(labels.shape[0]):
        df.iloc[row][labels[row]-1] = 1

    r_1_train_data.join(df).to_csv("./data/r_5_train_data.csv")


def set_r_labels(file_name, local_index):
    r_train_data = pd.read_csv(file_name, header=0, index_col=0)
    r_train_data = r_train_data.iloc[:, :-50]
    labels = pd.DataFrame(np.zeros((r_train_data.shape[0], 5)), dtype=np.int)
    for row in range(labels.shape[0]):
        labels.iloc[row][local_index] = 1

    r_train_data.join(labels).to_csv("./data/r_"+str(local_index+1)+"_train_data_new.csv")


if __name__ == '__main__':
    main()