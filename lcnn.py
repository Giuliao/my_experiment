# -*-coding: utf-8-*-
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import pandas as pd
import numpy as np
import math


def select_node_sequence():
    pass


def neighbor_assemble():
    pass


def receptive_field():
    pass


def normalize_graph():
    pass


def dfs_traverse(adj_matrix, field_size):

    def dfs(vertex, ll, is_used):
        if len(ll) > field_size:
            return

        for k in range(adj_matrix.shape[0]):
            if int(adj_matrix[k][vertex]) != 0 and k not in is_used:
                ll.append(k)
                is_used[k] = True
                dfs(k, ll, is_used)
        return

    total_ll = []
    for i in range(adj_matrix.shape[0]):
        ll = []
        dfs(i, ll, {})
        while len(ll) < field_size:
            ll.append(0)
        # print(ll)
        total_ll.extend(ll[:field_size])


def bfs_traverse(adj_matrix, field_size):

    def bfs(vertex):

        if vertex >= adj_matrix.shape[0]:
            return [0]*field_size

        ll = []
        is_used = {}
        count = 1
        que = [vertex]
        is_used[vertex] = True
        ll.append(count)
        while len(que) > 0 and len(ll) < field_size:
            vv = que.pop(0)
            count += 1
            for k in range(adj_matrix.shape[0]):
                if k not in is_used and int(adj_matrix[k][vv]) != 0:
                    ll.append(count)
                    que.append(k)
                    is_used[k] = True

        while len(ll) < field_size:
            ll.append(0)

        return ll

    total_ll = [None]*3
    # 9 means only get 9 nodes from the graph
    select_node = 9
    for i in range(select_node):
        tt = bfs(i)[:field_size]
        # print(tt) # debug
        field = np.array(tt).reshape((int(math.sqrt(field_size)), int(math.sqrt(field_size))))
        if total_ll[i//3] is None:
            total_ll[i//3] = field
        else:
            total_ll[i//3] = np.hstack((total_ll[i//3], field))

        # print(field)
    final_field = None
    for i in range(len(total_ll)):
        if final_field is None:
            final_field = total_ll[i]
        else:
            final_field = np.vstack((final_field, total_ll[i]))

    # print(final_field)

    return final_field.reshape((-1, final_field.shape[0]*final_field.shape[1]))


def bfs_data_generator(dir_path, file_name, node_num, field_size):

    df = pd.read_csv(dir_path+file_name, header=0, index_col=0)
    local_label = df.loc[:, ['r', 's']]
    new_df = None
    class_num = 2
    for i in range(df.shape[0]):
        data = df.iloc[i][:-class_num].values
        data = data.reshape(node_num, node_num)
        data = bfs_traverse(data, field_size)
        if new_df is None:
            new_df = pd.DataFrame(data)
        else:
            new_df = new_df.append(pd.DataFrame(data))
    new_df = pd.concat([new_df, local_label], axis=1)
    print(new_df.head())
    print(new_df.shape)

    new_df.reset_index(drop=True).to_csv(dir_path+'bfs_in_'+str(field_size)+'_'+file_name)


def test():
    import consensus_algo
    gg = consensus_algo.NetworkAlgo(vertex_num=6)
    print(gg.adjMatrix)
    bfs_traverse(gg.adjMatrix, 16)


if __name__ == '__main__':
    for i in range(6, 7):
          bfs_data_generator("./data/directed/node_6/", "r_{}.csv".format(i), 6, 16)


    # test()