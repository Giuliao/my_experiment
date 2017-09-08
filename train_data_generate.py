import consensus_algo
import CheckRobstness
import numpy as np
import pandas as pd
import time

def main():
    ll_r3 = np.array(range(100)).reshape((1, 101))
    ll_r2 = np.array(range(100)).reshape((1, 101))
    ll_r1 = np.array(range(100)).reshape((1, 101))
    l2 = []
    local_dict = {}
    count = 1
    for i in range(1, 6):
        for j in range(1, 11):
            local_dict[(i, j)] = count
            count += 1

    count_r1 = 10000
    count_r2 = 10000
    count_r3 = 10000
    start = time.time()
    try:
        while count_r1 or count_r2 or count_r3:
            adjmatrix = consensus_algo.NetworkAlgo.init_network(vertex_num=10)

            if adjmatrix.tolist() in l2:
                continue
            else:
                l2.append(adjmatrix.tolist())


            y = CheckRobstness.determine_robustness(adjmatrix)

            if y[0] == 3 and count_r3 != 0:
                x = np.column_stack((adjmatrix.reshape((1, adjmatrix.shape[0]**2)), [local_dict[y]]))
                np.concatenate()
                count_r3 -= 1
                print "count_r3 = %d" % count_r3
            elif y[0] == 2 and count_r2 != 0:
                ll_r2.append((x, local_dict[y]))
                count_r2 -= 1
                print "count_r2 = %d" % count_r2
            elif y[0] == 1 and count_r1 != 0:
                ll_r1.append((x, local_dict[y]))
                count_r1 -= 1
                print "count_r1 = %d" % count_r1
            else:
                continue

            print adjmatrix
            print y

    except Exception as e:
        print str(e)
    finally:
        np.save("./data/r_3_train_data", np.array(ll_r3))
        np.save("./data/r_2_train_data", np.array(ll_r2))
        np.save("./data/r_1_train_data", np.array(ll_r1))

        print "epoch %f" % (time.time()-start)


def test_data():
    a = np.load("./data/train_data.npy")
    dict = {}
    print a[:, 1]
    for row in range(a.shape[0]):
        if not dict.has_key(a[row, 1]):
            dict[a[row, 1]] = 0
        else:
            dict[a[row, 1]] += 1

    for k, v in dict.iteritems():
        print k, '=>', v


def test_hvstack():
    r_1_train_data = np.load("./data/r_5_train_data.npy")
    r_1_ll = r_1_train_data[0, :-1][0].reshape(1, 100)
    r_1_ll = np.column_stack((r_1_ll, r_1_train_data[0, -1]))
    for i in range(1, r_1_train_data.shape[0]):
        tt = np.column_stack((r_1_train_data[i, :-1][0].reshape(1, 100), r_1_train_data[i, -1]))
        r_1_ll = np.row_stack((r_1_ll, tt))

    pd.DataFrame(r_1_ll).to_csv("./data/r_5_train_data.csv")

    print r_1_ll.shape



if __name__ == '__main__':
    test_hvstack()