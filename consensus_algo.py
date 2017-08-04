# -*-coding:utf-8-*-


import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


class NetworkAlgo(object):

    def __init__(self, vertex_num=5, network_file=None, vertex_value_file=None, adjMatrix=None):
        """ the init constructor
        :param vertex_num: 
        :param network_file: 
        :param vertex_value_file: 
        
        :member variable
            self.adjMatrix
            self.G
            self.malicious_node
        """
        self.adjMatrix = self.init_network(vertex_num, network_file) if not adjMatrix else adjMatrix.copy()
        self.G = self.init_graph(self.adjMatrix)
        self.init_vertex_value(self.G, vertex_value_file)
        self.malicious_node = []

    @staticmethod
    def init_network(vertex_num, file_name):
        """ init the network by reading a file
        
        :param file_name: 
                the first line is the number of vertex
                the next lines of which the first number is the vertex as 
                the start point then the next are the end respectively
        :param vertex_num:
                
        :return:
            local_adjMatrix
        """
        local_adjMatrix = None
        if not file_name:   # init by file
            local_adjMatrix = np.random.randint(2, size=(vertex_num, vertex_num))
            for i in range(vertex_num):
                local_adjMatrix[i][i] = 0
        else:               # init by random
            with open(file_name, 'r') as fd:
                for line in fd.readlines():
                    tt = line.split(' ')

                    if len(tt) == 1:
                        vv = int(tt[0])
                        local_adjMatrix = np.zeros([vv, vv], dtype=np.int)
                        continue

                    for i in range(1, len(tt)):
                        local_adjMatrix[int(tt[0])-1][int(tt[i])-1] = 1

        return local_adjMatrix

    @staticmethod
    def init_graph(local_adjMatrix):
        local_G = nx.DiGraph()  # the init of networkx object
        for i in range(local_adjMatrix.shape[0]):
            for j in range(local_adjMatrix.shape[1]):
                if local_adjMatrix[i][j] != 0:
                    local_G.add_edge(i + 1, j + 1)
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
        if not isinstance(local_G, nx.DiGraph):
            return False

        import random
        if not file_name:
            for v in local_G.nodes():
                local_G.node[v]['value'] = [random.uniform(0.0, 1.0)]

        else:
            with open(file_name, 'r') as fd:
                for line in fd.readlines():
                    tt = line.split(' ')
                    local_G.node[int(tt[0])]['value'] = [float(tt[1])]
        return True

    def set_malicious_node(self, kwargs):
        for k, v in kwargs.iteritems():
            self.G.node[k]['value'][0] = v
            if k not in self.malicious_node:
                self.malicious_node.append(k)

    def get_indegree_node_value(self, node, iter_time):
        neighbors_value = []
        for k, v in self.G.in_edges(node):
            neighbors_value.append(self.G.node[k]['value'][iter_time])
        return neighbors_value

    def show_network(self):
        import time
        plt.figure(time.time())
        for v in self.G.nodes():
            self.G.node[v]['state'] = str(v)

        node_labels = nx.get_node_attributes(self.G, 'state')
        pos = nx.spring_layout(self.G)
        nx.draw_networkx_labels(self.G, pos, node_labels=node_labels)
        # nx.draw_networkx_edges(self.G, pos)
        # print self.adjMatrix
        nx.draw(self.G, pos)
        # plt.savefig('./assets/result.png')

        plt.show(block=False)


class ArcpAlgo(NetworkAlgo):

    def __init__(self, vertex_num=5, network_file=None, vertex_value_file=None):
        """
        :param vertex_num: 
        :param network_file: 
        :param vertex_value_file: 
        """
        super(ArcpAlgo, self).__init__(vertex_num, network_file, vertex_value_file)


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
        while i < f_local:
            if neighbors[i] >= v_self_value:
                break
            i += 1
            iRangeMin += 1

        i = 0
        iRangeMax = len(neighbors)
        while i < f_local:
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
        print 'ok arcp'
        self.show_network()
        self.show_consensus(f_local, iter_time)

    def show_consensus(self, f_local, iter_time):
        x = range(iter_time+1)
        import time
        plt.figure(time.time())
        plt.title('arcp with f-local=%d' % f_local)
        for y in self.G.nodes():
            plt.plot(x, self.G.node[y]['value'])
        plt.show()


class McaAlgo(NetworkAlgo):
    def __init__(self, vertex_num=5, network_file=None, vertex_value_file=None):
        super(McaAlgo, self).__init__(vertex_num, network_file, vertex_value_file)

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
            median = (neighbors_value[len(neighbors_value)/2] + neighbors_value[len(neighbors_value)/2-1])/2
        else:
            median = neighbors_value[ll/2]
        self.G.node[v]['value'].append(w1*self.G.node[v]['value'][iter_time]+w2*median)

    def run_median_consensus_algo(self, iter_time):
        for it in range(iter_time):
            for v in self.G.nodes():
                self.median_consensus_algo(v, it)

        print "Ok mac"
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

    test = ArcpAlgo(20)
    test.set_malicious_node({2: 2.0})
    test.run_arcp(2, 100)

    # test2 = McaAlgo(20)
    # test2.set_malicious_node({2:3.0})
    # test2.run_median_consensus_algo(100)

if __name__ == '__main__':
    main()