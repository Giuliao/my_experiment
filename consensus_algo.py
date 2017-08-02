# -*-coding:utf-8-*-


import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


class NetworkAlgo(object):

    def __init__(self, vertex_num=5, file_name=None):
        super(NetworkAlgo, self).__init__()
        self.adjMatrix = None
        self.G = None
        if file_name:
            self.init_network_by_file(file_name)
        else:
            self.adjMatrix = np.random.randint(2, size=(vertex_num, vertex_num))
            for i in range(vertex_num):
                self.adjMatrix[i][i] = 0

        self.G = nx.DiGraph()
        for i in range(self.adjMatrix.shape[0]):
            for j in range(self.adjMatrix.shape[1]):
                if self.adjMatrix[i][j] != 0:
                    self.G.add_edge(i + 1, j + 1)

    def init_network_by_file(self, file_name):
        """ init the network by reading a file
        
        :param file_name: 
                the first line is the number of vertex
                the next lines of which the first number is the vertex as 
                the start point then the next are the end respectively
        :return:
         
        """

        with open(file_name, 'r') as fd:
            for line in fd.readlines():
                tt = line.split(' ')

                if len(tt) == 1:
                    vv = int(tt[0])
                    self.adjMatrix = np.zeros([vv, vv], dtype=np.int)
                    continue

                for i in range(1, len(tt)):
                    self.adjMatrix[int(tt[0])-1][int(tt[i])-1] = 1

    def show_network(self):
        for v in self.G.nodes():
            self.G.node[v]['state'] = str(v)

        node_labels = nx.get_node_attributes(self.G, 'state')
        pos = nx.spring_layout(self.G)
        nx.draw_networkx_labels(self.G, pos, node_labels=node_labels)
        # nx.draw_networkx_edges(self.G, pos)
        print self.adjMatrix
        nx.draw(self.G, pos)
        plt.savefig('./assets/result.png')
        plt.show()


class ArcpAlgo(NetworkAlgo):

    def __init__(self, vertex_num=5, network_file=None, vertex_value_file=None):
        """
        :param vertex_num: 
        :param nework_file: 
        :param vertex_value_file: 
        """
        super(ArcpAlgo, self).__init__(vertex_num, network_file)
        self.malicious_node = [0]
        self.init_vertex_value(vertex_value_file)

    def init_vertex_value(self, file_name=None):
        """ init the value on every vertex
        
        :param file_name: 
            every line is like such:
                number of node  and the vlaue
                1 2
        :return: 
        """
        import random
        if not file_name:
            for v in self.G.nodes():
                self.G.node[v]['value'] = [random.uniform(0.0, 1.0)]
            return

        with open(file_name, 'r') as fd:
            for line in fd.readlines():
                tt = line.split(' ')
                self.G.node[int(tt[0])]['value'] = [float(tt[1])]


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

    def arcp(self, v, f_local, iter_time):

        def my_sum(my_list):
            import random
            constant = 8.0 / 9.0
            alpha = 1.0 - constant
            weight_sum = 0.0
            neighbors_sum = 0.0
            for v in my_list:
                t = random.uniform(alpha, constant / len(my_list))
                neighbors_sum += t * v
                weight_sum += t
            return weight_sum, neighbors_sum

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
        self.show_consensus(iter_time, f_local)

    def show_consensus(self, iter_time, f_local):
        x = range(iter_time+1)
        plt.figure(figsize=(8, 8), dpi=80)
        plt.title('arcp with f-local=%d' %f_local)
        for y in self.G.nodes():
            plt.plot(x, self.G.node[y]['value'])
        plt.show()



def main():

    test = ArcpAlgo(network_file="data/data.in")
    test.set_malicious_node({2: 2.0})
    test.run_arcp(f_local=0, iter_time=100)

if __name__ == '__main__':
    main()