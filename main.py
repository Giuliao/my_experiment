from __future__ import print_function
from __future__ import division
from config import Config
from data_processing import DataGenerator
from model import neural_network

# from model.sdne import SDNE
# from graph import Graph
# config = Config()
#
# graph_data = Graph(config.file_name_list, config.n_classes)
# config.structure[0] = 10
# model = SDNE(config)
# model.do_variables_init(graph_data, config.DBN_init)
#
# epochs = 0
# batch_n = 0
#
# while (True):
#     # graph_data.N = int(config.rN * graph_data.N)
#     mini_batch = graph_data.sample(config.batch_size)
#     loss = model.fit(mini_batch)
#     batch_n += 1
#     print("Epoch : %d, batch : %d, loss: %.3f" % (epochs, batch_n, loss))
#     if graph_data.is_epoch_end:
#         if epochs % config.display == 0:
#             embedding = None
#             while (True):
#                 mini_batch = graph_data.sample(config.batch_size, do_shuffle=False)
#                 loss += model.get_loss(mini_batch)
#                 if embedding is None:
#                     embedding = model.get_embedding(mini_batch).reshape(1, 50)
#                 else:
#                     embedding = np.vstack((embedding, model.get_embedding(mini_batch).reshape(1, 50)))
#
#                 if graph_data.is_epoch_end:
#                     break
#
#         if epochs == 100: # embedding
#             # yy = pd.DataFrame(graph_data.train_Y)
#             # xx = pd.DataFrame(embedding)
#             # xx.join(yy).to_csv("./data/r_5_representaion.csv")
#             embedding = np.column_stack((embedding, graph_data.train_Y))
#             print(embedding.shape)
#             pd.DataFrame(embedding).to_csv("./data/r_5_representaion.csv")
#             break
#
#         epochs += 1
#         batch_n = 0


if __name__ == '__main__':
    con = Config()
    data = DataGenerator(con)
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    # model = neural_network.NeuralNetwork(con)
    # model.run_train(data)

