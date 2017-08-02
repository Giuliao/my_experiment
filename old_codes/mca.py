# -*-coding:utf-8-*-
import matplotlib.pyplot as plt
import random


class ExcessGraph:
    def __init__(self, file_name=None):
        self.graph, self.value = self.construct_graph_from_file(file_name)
        self.constant = 8.0 / 9.0
        self.alpha = 1.0 - self.constant  # 1 - max_neighbor_number / (max_neighbor_number + 1)

    def construct_rs_excess_robustness(self, r=0, s=2):
        graph = {
            1: [2, 3, 4, 5, 6],
            2: [1, 3, 4, 5, 6],
            3: [1, 2, 4, 5, 6],
            4: [1, 2, 3, 5, 6],
            5: [],
            6: []
        }

        value = {}
        for i in range(1, 7):
            value[i] = [random.uniform(1.0, 7.0)]
        value[4][0] = '8.0'
        return graph, value

    def construct_graph_from_file(self, file_name=None):
        if not file_name:
            return self.construct_rs_excess_robustness()

        print 'not complete'
        return None, None

    def print_graph(self):
        for k, v in self.graph.iteritems():
            print k, '=>', v

        for k, v in self.value.iteritems():
            print k, '=>', v

    def get_in_neighbors(self, index):
        neighbors = []
        for k, v in self.graph.iteritems():
            if k != index:
                if index in v:
                    neighbors.append(k)

        return neighbors

    def get_in_neighbors_value(self, index, iter_time):
        neighbors_value = []
        neighbors = self.get_in_neighbors(index)

        for k in neighbors:
            neighbors_value.append(self.value[k][iter_time])

        return neighbors_value

    def median_consensus_algo(self, index, iter_time):
        if index == 4:
            self.value[index].append(self.value[index][0])
            return

        w1 = 0.5
        w2 = 0.5
        neighbors_value = self.get_in_neighbors_value(index, iter_time)
        ll = len(neighbors_value)
        if ll % 2 == 0:
            median = (neighbors_value[len(neighbors_value)/2] + neighbors_value[len(neighbors_value)/2-1])/2
        else:
            median = neighbors_value[ll/2]
        self.value[index].append(w1*self.value[index][iter_time]+w2*median)

    def my_sum(self, my_list):
        weight_sum = 0.0
        neighbors_sum = 0.0
        for v in my_list:
            t = random.uniform(self.alpha, self.constant / len(my_list))
            neighbors_sum += t * v
            weight_sum += t
        return weight_sum, neighbors_sum

    def arcp(self, index, f_local, iter_time):
        if index == 4:
            self.value[index].append(self.value[index][0])
            return

        v_self_value = self.value[index][iter_time]
        neighbors = self.get_in_neighbors_value(index, iter_time)
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
        weight_sum, ss = self.my_sum(neighbors[iRangeMin: iRangeMax])
        self.value[index].append(ss + (-weight_sum) * v_self_value + v_self_value)

    def __copy__(self, g2):
        for i in range(1, 7):
            self.value[i] = g2.value[i][:]
def w_msr():
    pass


def main():
    g1 = ExcessGraph()
    g2 = ExcessGraph()
    g2.__copy__(g1)
   # g2.print_graph()
    g1.print_graph()

    for iter_time in range(100):
        for i in range(1, 7):
            g1.median_consensus_algo(i, iter_time)
            g2.arcp(i, 1, iter_time)

    # g1.print_graph()
    # print '-'*100
    # g2.print_graph()
    x = range(101)
    plt.figure()
    plt.subplot(211)
    plt.title("mca")
    plt.xlim((0,20))
    for k, y in g1.value.iteritems():
        if k == 4:
            plt.plot(x, y, 'r:')
            continue
        plt.plot(x, y, 'k-')
    plt.subplot(212)
    plt.title("arcp with flocal=1")
    plt.xlim((0, 20))
    for k, y in g2.value.iteritems():
        if k == 4:
            plt.plot(x, y, 'r:')
            continue
        plt.plot(x, y, 'k-')

    plt.show()

if __name__ == '__main__':
    main()