# -*-coding:utf-8-*-
# input : parent node, child nodes

import matplotlib.pyplot as plt

graph = {}
value = {}
input_file = 'data.in'


def print_graph():
    for k, val in graph.iteritems():
        print k, '=>', val
    for k, val in value.iteritems():
        print k, '=>', val


def init():
    global graph
    for i in range(1, 27):
        graph[i] = i
        value[i] = [i*10*1.0]
    with open(input_file, 'r') as f:
        for line in f.readlines():
            node = line.strip('\n').split(' ')
            tmp_value = 0.0
            for i in range(len(node)):
                graph[int(node[i])] = int(node[0])
                tmp_value += value[int(node[i])][0] if i != 0 else 0
            value[int(node[0])][0] = tmp_value/(len(node) - 1)


def find_peer2(index, iter_time):
    arrPeerVals = {}
    arrPeerVals[graph[index]]=value[graph[index]][iter_time]
    arrPeerVals[index] = value[index][iter_time]
    for k, val in graph.iteritems():
        if val == graph[index]:
            arrPeerVals[k] = value[k][iter_time]
    return arrPeerVals


def find_peer(index, iter_time):
    arrPeerVals = {}
    arrPeerVals[graph[index]] = value[graph[index]][iter_time]
    arrPeerVals[index] = value[index][iter_time]
    for k, val in graph.iteritems():
        if val == index:
            arrPeerVals[k] = value[k][iter_time]
    return arrPeerVals


def rarcp_algo(index, FLocal, iter_time):
    if index == 4 or index == 5 or index == 6 or index == 8 or index == 9:
        value[index].append(value[index][0])
        return

    cur_child_val, cur_child_parent_val = value[index][iter_time], value[graph[index]][iter_time]
    rangeMin, rangeMax = (cur_child_parent_val, cur_child_val) if cur_child_parent_val < cur_child_val else (cur_child_val, cur_child_parent_val)
    arrPeerVals = find_peer(index, iter_time)
    sorted_arrPeerVals = sorted(arrPeerVals.iteritems(), key=lambda tmp: tmp[1])

    i = 0
    iRangeMin = 0
    while i < FLocal:
        if sorted_arrPeerVals[iRangeMin][1] >= rangeMin:
            break
        i += 1
        iRangeMin += 1

    i = 0
    iRangeMax = len(sorted_arrPeerVals) - 1
    while i < FLocal:
        if sorted_arrPeerVals[iRangeMax][1] <= rangeMax:
            break
        i += 1
        iRangeMax -= 1
    tt = 0
    for k, v in sorted_arrPeerVals[iRangeMin: iRangeMax+1]:
        tt += v

    value[index].append(tt/(iRangeMax-iRangeMin+1))


def main():
    init()
    for iter_time in range(500):
        for j in range(1, 27):
            rarcp_algo(j, 3, iter_time)

    x = range(501)
    plt.xlim((0, 30))
    for k, v in value.iteritems():
        plt.plot(x, v)
    plt.show()


if __name__ == '__main__':
    init()
    main()
