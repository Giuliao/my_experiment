# -*-coding:utf-8-*-
# input: vertex, value, neighbors

import matplotlib.pyplot as plt
import random


class MyGraph(object):
    def __init__(self, in_file='data2.in'):
        self.vertex = {}
        self.in_file = in_file
        self.constant = 8.0 / 9.0
        self.alpha = 1.0 - self.constant  # 1 - max_neighbor_number / (max_neighbor_number + 1)

        for i in range(1, 15):
            self.vertex[i] = {'value': [],'neighbors': []}
        with open(self.in_file) as f:
            for line in f.readlines():
                tmp_input = line.strip('\n').split(' ')
                self.vertex[int(tmp_input[0])]['value'].append(float(tmp_input[1]))
                for num in range(2, len(tmp_input)):
                    self.vertex[int(tmp_input[0])]['neighbors'].append(int(tmp_input[num]))

    def print_graph(self):
        for k, v in self.vertex.iteritems():
            print k, "=>", v

    def judge_r_robustness(self):
        pass

    def judge_r_reachable(self):
        pass

    def judge_rs_robustness(self):
        pass

    def judge_rs_reachable(self):
        pass

    def get_neighbors_value(self, index, iter_time):
        neighbors = []
        if self.vertex[index]:
            for i in self.vertex[index]['neighbors']:
                if self.vertex[i]:
                    neighbors.append(self.vertex[i]['value'][iter_time])
        return neighbors

    def my_sum(self, my_list):
        weight_sum = 0.0
        neighbors_sum = 0.0
        for v in my_list:
            t = random.uniform(self.alpha, self.constant/len(my_list))
            neighbors_sum += t*v
            weight_sum += t
        return weight_sum, neighbors_sum

    def lcp(self, v, iter_time):
        if v == 14:
            self.vertex[v]['value'].append(self.vertex[v]['value'][0])
            return

        neighbors = self.get_neighbors_value(v, iter_time)
        v_self_value = self.vertex[v]['value'][iter_time]
    # neighbors.append(v_self_value)
        weight_sum, neighbors_sum = self.my_sum(neighbors)
    # print "y = %f, statisfy: %d" %(-weight_sum, 1 if weight_sum <= constant else 0)
        self.vertex[v]['value'].append((-weight_sum)*v_self_value+neighbors_sum+v_self_value)

    def arcp(self, v, f_local, iter_time):
        if v == 14:
            self.vertex[v]['value'].append(self.vertex[v]['value'][0])
            return

        neighbors = self.get_neighbors_value(v, iter_time)
        v_self_value = self.vertex[v]['value'][iter_time]
        # neighbors.append(v_self_value)
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
            if neighbors[iRangeMax-1] <= v_self_value:
                break
            i += 1
            iRangeMax -= 1
        # ss = sum(neighbors[iRangeMin: iRangeMax])
        weight_sum, ss = self.my_sum(neighbors[iRangeMin: iRangeMax])
        #print "y = %f, statisfy: %d" % (-weight_sum, 1 if weight_sum <= constant else 0)
        self.vertex[v]['value'].append(ss+(-weight_sum)*v_self_value+v_self_value)


if __name__ == '__main__':
    test1 = MyGraph()
    test2 = MyGraph()
    for iter_time in range(100):
        for v in range(1, 15):
            test2.arcp(v, 1, iter_time)
            test1.lcp(v, iter_time)

    x = range(101)
    plt.figure(figsize=(8, 8), dpi=80)
    plt.subplot(121)
    plt.title('lcp')
    for y in range(1, 15):
        plt.plot(x, test1.vertex[y]['value'])
    plt.subplot(122)
    plt.title('arcp with f-local=1')
    for y in range(1, 15):
        plt.plot(x, test2.vertex[y]['value'])

    plt.show()



