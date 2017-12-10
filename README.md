![TEST](./assets/result1.png)
# my_experiment
- the judgment of (r, s)-robust by Deep learning
- here contains two branches that are my old codes just for memorizing =.=|
## CheckRobustness.py
- implemented from the paper '*Algorithms for DeterminingNetwork Robustness*'
## consensus_algo.py
- superclass: NetworkAlgo
- subclass: ArcpAlgo
- subclass: LcpAlgo(ToDo)
- subclass: RarcpAlgo(ToDo)
- subclass: MedianConsensusAlgo(Todo)

## tranin_data_generate.py
- generate the (r, s)-robust network and save as csv file
## data_processing.py
- class DataGenerator: generate train and test data by a batch-like way
- get_image_data: generate image data and save as a csv file
- test_image: just have a glance at the how the images look like
## cnn_with_tf.py and dnn_with_tf.py
- the origin code to implement a deep learning model
## graph.py
- will modified it or just remove it
- get from [here](https://github.com/suanrong/SDNE/blob/master/graph.py)

### model
- config.py: a configuration file, can be read from a [json file](https://github.com/Giuliao/my_experiment/tree/master/data/configurations)
- neural_network.py: generate a model
- rbm.py: see [here](https://gist.github.com/blackecho/db85fab069bd2d6fb3e7)
- sdne.py see [here](https://github.com/suanrong/SDNE/blob/master/model/sdne.py)
- utils.py see [here](https://gist.github.com/blackecho/db85fab069bd2d6fb3e7)

### data
- include the network data distinguished by (r, s)-robust

### assests
 

