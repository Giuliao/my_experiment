from __future__ import division
from __future__ import print_function

from model import neural_network
from model import config
import data_processing

if __name__ == '__main__':
    con = config.Config(json_file='./data/configurations/conf2.json')
    data = data_processing.DataGenerator(con)
    mm = neural_network.NeuralNetwork(con)
    mm.build_network_model()
    mm.init_session(con)
    mm.run_train(data)