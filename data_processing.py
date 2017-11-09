from __future__ import print_function
from __future__ import division
from sklearn.model_selection import train_test_split
import pandas as pd
import config



class DataGenerator:

    def __init__(self, con):
        self.file_name_list = con.file_name_list
        self.n_classes = con.n_classes
        self.train_X, self.train_Y, self.test_X, self.test_Y = self.read_from_csv_list()
        self.total_train = self.train_X.shape[0]

    def read_from_csv_list(self):
        pd_ll = []
        for local_file in self.file_name_list:
            pd_ll.append(pd.read_csv(local_file, header=0, index_col=0))

        df = pd.concat(pd_ll)
        valid_X = df.iloc[:, : -self.n_classes].reset_index(drop=True)
        valid_Y = df.iloc[:, -self.n_classes:].reset_index(drop=True)
        train_X, test_X, train_Y, test_Y = train_test_split(valid_X, valid_Y, test_size=0.2)

        return train_X, train_Y, test_X, test_Y

    def get_train_data(self, batch_size):
        count = 0
        batch_x = pd.DataFrame(columns=self.train_X.columns)
        batch_y = pd.DataFrame(columns=self.train_Y.columns)
        for i in range(self.total_train):
            batch_x = batch_x.append(self.train_X.iloc[i, :])
            batch_y = batch_y.append(self.train_Y.iloc[i, :])
            count += 1
            if count == batch_size:
                yield batch_x, batch_y
                batch_x.drop(batch_x.index, inplace=True)
                batch_y.drop(batch_y.index, inplace=True)
                count = 0

        if count != 0:
            yield batch_x, batch_y

        return

    def get_test_data(self, batch_size):
        return self.test_X, self.test_Y

if __name__ == '__main__':
    con = config.Config()
    data = DataGenerator(con)
    for x, y in data.get_train_data(100):
        print(x.shape)
        print(y.shape)
        print(x.head())
        print(y.head())

