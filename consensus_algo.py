# -*-coding:utf-8-*-
# reference:
# [1]py2 compatible with py3, http://blog.csdn.net/u012151283/article/details/58049151
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random
import pandas as pd
import threading
import time


class NetworkAlgo(object):
    def __init__(self, vertex_num=5, p=0.5, directed=False, network_file=None, vertex_value_file=None, adjMatrix=None, nx_obj=None):
        """ the init constructor
        :param vertex_num: 
        :param network_file: 
        :param vertex_value_file: 
        
        :member variable
            self.adjMatrix
            self.G
            self.malicious_node
        """
        self.vertex_num = vertex_num
        self.G = self.init_network(vertex_num, p, directed, network_file, adjMatrix, nx_obj)
        self.adjMatrix = nx.adjacency_matrix(self.G).todense().view(np.ndarray).astype(np.int8).reshape(self.vertex_num,
                                                                                        self.vertex_num)

        self.init_vertex_value(self.G, vertex_value_file)
        self.malicious_node = []

    def init_network(self, vertex_num=5, p=0.9, directed=False, file_name=None, adjMatrix=None, nx_obj=None):
        """ init the network by reading a file
        
        :param file_name: 
                the first line is the number of vertex
                the next lines of which the first number is the vertex as 
                the start point then the next are the end respectively
        :param vertex_num:
        :param p:
                
        :return:
            
        """
        local_adjMatrix = adjMatrix
        if not file_name:
            # init by random
            # local_list = np.random.permutation(vertex_num)
            # local_adjMatrix = np.zeros([vertex_num, vertex_num], dtype=np.int)
            #
            # for index, var in enumerate(local_list):
            #     if index == vertex_num - 1:
            #         break
            #
            #     kk = np.random.randint(0, 2)  # control the direction of a matrix
            #     if kk == 0:
            #         local_adjMatrix[local_list[index]][local_list[index+1]] = 1
            #     else:
            #         local_adjMatrix[local_list[index+1]][local_list[index]] = 1
            #
            # m = np.random.randint(1, vertex_num*vertex_num*vertex_num)  # control the density of the matrix
            # for i in range(m):
            #     p1 = np.random.randint(0, vertex_num)
            #     p2 = np.random.randint(0, vertex_num)
            #     while p1 == p2:
            #         p2 = np.random.randint(0, vertex_num)
            #     local_adjMatrix[p1][p2] = 1
            if local_adjMatrix is not None:
                local_G = nx.from_numpy_matrix(local_adjMatrix)
                if directed:
                    local_G = local_G.to_directed()
            elif nx_obj is not None:
                local_G = nx_obj.copy()
                if directed:
                    local_G = local_G.to_directed()
            else:
                local_G = nx.binomial_graph(vertex_num, p, directed=directed)

        else:
            # init by file
            with open(file_name, 'r') as fd:
                for line in fd.readlines():
                    tt = line.split(' ')

                    if len(tt) == 1:
                        vv = int(tt[0])
                        local_adjMatrix = np.zeros([vv, vv], dtype=np.int)
                        self.vertex_num = vv
                        continue

                    for i in range(1, len(tt)):
                        local_adjMatrix[int(tt[0]) - 1][int(tt[i]) - 1] = 1

            local_G = nx.from_numpy_matrix(local_adjMatrix)
        self.vertex_num = len(local_G)

        return local_G

    @staticmethod
    def init_vertex_value(local_G, file_name=None):
        """ init the value on every vertex

        :param file_name: 
                every line is like such:
                number of node  and the vlaue
                1 2
        :param local_G:
                
        :return: 
            boolean
        """
        # if not isinstance(local_G, nx.DiGraph):
        #     return False
        if not file_name:
            for v in local_G.nodes():
                local_G.node[v]['value'] = [random.uniform(0.0, 1.0)]

        else:
            with open(file_name, 'r') as fd:
                for line in fd.readlines():
                    tt = line.split(' ')
                    local_G.node[int(tt[0])]['value'] = [float(tt[1])]
        return True

    @staticmethod
    def write_adjmatrix_to_file(local_adjmatrix, out_file="./data/data.out"):
        """
        :param local_adjmatrix: 
        :param out_file: 
        :return: 
        """
        np.savetxt(out_file, local_adjmatrix)

    def set_malicious_node(self, kwargs):
        for k, v in kwargs.iteritems():
            self.G.node[k]['value'][0] = v
            if k not in self.malicious_node:
                self.malicious_node.append(k)

    def get_indegree_node_value(self, node, iter_time):
        neighbors_value = []
        # for k, v in self.G.in_edges(node):
        for k, v in self.G.edges(node):
            # print(k, v)
            neighbors_value.append(self.G.node[v]['value'][iter_time])
        return neighbors_value

    def show_network(self):
        import time
        plt.figure(time.time())
        for v in self.G.nodes():
            self.G.node[v]['state'] = str(v+1)

        node_labels = nx.get_node_attributes(self.G, 'state')
        pos = nx.circular_layout(self.G)
        nx.draw_networkx_labels(self.G, pos, labels=node_labels)
        nx.draw(self.G, pos)
        plt.savefig('./assets/result2.png')
        # plt.show(block=False)
        plt.close()


class ArcpAlgo(NetworkAlgo):
    def __init__(self, vertex_num=5, p=0.9, network_file=None, vertex_value_file=None, adjmatrix=None):
        """
        :param vertex_num: 
        :param network_file: 
        :param vertex_value_file: 
        """
        super(ArcpAlgo, self).__init__(vertex_num, p, network_file, vertex_value_file, adjmatrix)

    def arcp(self, v, f_local, iter_time):

        def my_sum(my_list):
            import random
            constant = 8.0 / 9.0
            alpha = 1.0 - constant
            local_weight_sum = 0.0
            neighbors_sum = 0.0
            for local_v in my_list:
                t = random.uniform(alpha, constant / len(my_list))
                neighbors_sum += t * local_v
                local_weight_sum += t
            return local_weight_sum, neighbors_sum

        if v in self.malicious_node:
            self.G.node[v]['value'].append(self.G.node[v]['value'][0])
            return

        neighbors = self.get_indegree_node_value(v, iter_time)
        v_self_value = self.G.node[v]['value'][iter_time]
        neighbors.sort()

        i = 0
        iRangeMin = 0
        while i < f_local and len(neighbors) > i:
            if neighbors[i] >= v_self_value:
                break
            i += 1
            iRangeMin += 1

        i = 0
        iRangeMax = len(neighbors)
        while i < f_local and iRangeMax > 0:
            if neighbors[iRangeMax - 1] <= v_self_value:
                break
            i += 1
            iRangeMax -= 1
        # ss = sum(neighbors[iRangeMin: iRangeMax])
        weight_sum, ss = my_sum(neighbors[iRangeMin: iRangeMax])
        # print "y = %f, statisfy: %d" % (-weight_sum, 1 if weight_sum <= constant else 0)
        self.G.node[v]['value'].append(ss + (-weight_sum) * v_self_value + v_self_value)

    def run_arcp(self, f_local=1, iter_time=30):
        for it in range(iter_time):
            for v in self.G.nodes():
                self.arcp(v, f_local, it)
        print('ok arcp')
        print(self.adjMatrix)
        print('-' * 100)
        for y in self.G.nodes():
            print(str(y) + ':', self.G.node[y]['value'])
        self.show_network()
        self.show_consensus(f_local, iter_time)

    def show_consensus(self, f_local, iter_time):
        x = range(iter_time + 1)
        import time
        plt.figure(time.time())
        plt.title('arcp with f-local=%d' % f_local)
        plt.xlim((0, 100))
        for y in self.G.nodes():
            plt.plot(x, self.G.node[y]['value'])
        plt.show()


class McaAlgo(NetworkAlgo):
    def __init__(self, vertex_num=5, p=0.9, network_file=None, vertex_value_file=None, adjmatrix=None):
        super(McaAlgo, self).__init__(vertex_num, p, network_file, vertex_value_file, adjMatrix=adjmatrix)

    def median_consensus_algo(self, v, iter_time):
        if v in self.malicious_node:
            self.G.node[v]['value'].append(self.G.node[v]['value'][0])
            return

        w1 = 0.5
        w2 = 0.5
        neighbors_value = self.get_indegree_node_value(v, iter_time)
        neighbors_value.sort()
        ll = len(neighbors_value)
        if ll % 2 == 0:
            median = int(
                (neighbors_value[len(neighbors_value) // 2] + neighbors_value[(len(neighbors_value) // 2) - 1]) / 2)
        else:
            median = neighbors_value[ll // 2]
        self.G.node[v]['value'].append(w1 * self.G.node[v]['value'][iter_time] + w2 * median)

    def run_median_consensus_algo(self, iter_time):
        for it in range(iter_time):
            for v in self.G.nodes():
                self.median_consensus_algo(v, it)

        print("Ok mac")
        for y in self.G.nodes():
            print(str(y) + ':', self.G.node[y]['value'])
        self.show_network()
        self.show_consensus(iter_time)

    def show_consensus(self, iter_time):
        x = range(iter_time + 1)
        import time
        plt.figure(time.time())

        plt.title('mac')
        for y in self.G.nodes():
            plt.plot(x, self.G.node[y]['value'])
        plt.show()


def main():
    # test = ArcpAlgo(10, 0.5)
    # test.set_malicious_node({2: 2.0})
    # test.run_arcp(1, 100)
    # test.show_network()
    # test.write_to_file(test.adjMatrix)
    # x = np.loadtxt("data.out",dtype=np.int)
    # print x.shape
    # print np.product(x, x.transpose())
    test2 = McaAlgo(20, 0.9)
    print(test2.adjMatrix)
    test2.set_malicious_node({2: 3.0})
    test2.run_median_consensus_algo(100)


if __name__ == '__main__':
    main()
