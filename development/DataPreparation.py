import mysql.connector, pandas, re
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical


class DataPreparation():


    def __init__(self):
        pass


    def load_data_from_database(self, sql):
        mydb = mysql.connector.connect(host='localhost', database='news_dataset', user='root', password='')
        data_db = pandas.read_sql(sql, con=mydb)
        return data_db


    def clean_data(self, data, loop):
        for i in range(loop):
            data.loc[i] = ' '.join(data.loc[i].split('-'))
            data.loc[i] = re.sub(r'[^\w\s]', ' ', data.loc[i].lower())


    def create_data_label(self, size):

        labels = []

        for i in range(size):
            labels.append('Fake')
            labels.append('Real')

        return labels


    def encode_label(self, label):

        # label encode the target variable
        encoder = LabelEncoder()
        label = encoder.fit_transform(label)
        encoded_label = to_categorical(label, num_classes=4)

        return encoded_label


    def prepare_data_frame(self):

        # load data from database
        data = self.load_data_from_database('SELECT fake_news, real_news from news')

        # merge fake_news and real_news into single dataframe alternately
        data = pandas.concat([data.fake_news, data.real_news]).sort_index(kind='merge')

        # reset index bcoz of alternate merging process before
        data = data.reset_index(drop=True)

        # generate label for data in dataDF
        label = self.create_data_label(size=239)

        # prepare dataframe with news and label
        dataDF = pandas.DataFrame()
        dataDF['news'] = data
        dataDF['label'] = label

        # clean and lowercase data in dataDF
        self.clean_data(dataDF['news'], loop=478)

        return dataDF


    def prepare_stance_data_frame(self):

        # load data from database limit to 500 news only
        data_sebenarnya = self.load_data_from_database('SELECT penyataan, sumber  FROM stance_sebenarnya WHERE label = "tidak setuju" LIMIT 500')
        data_utusan = self.load_data_from_database('SELECT penyataan, sumber  FROM stance_utusan WHERE label = "setuju" LIMIT 500')

        # merge data_sebenarnya and data_utusan into single dataframe alternately
        dataDF = pandas.concat([data_sebenarnya, data_utusan]).sort_index(kind='merge')

        # reset index bcoz of alternate merging process before
        dataDF = dataDF.reset_index(drop=True)

        # generate label for data in dataDF
        label = self.create_data_label(size=500)

        # add column label in dataDF
        dataDF['label'] = pandas.DataFrame(label, index=dataDF.index)

        # clean and lowercase data in dataDF
        self.clean_data(dataDF['penyataan'], loop=1000)
        self.clean_data(dataDF['sumber'], loop=1000)

        return dataDF


    def prepare_data_normal_split(self, test_ratio):

        # get data_frame that stores news and label
        data_frame = self.prepare_data_frame()

        # split the dataset into training and validation datasets
        train_x, test_x, train_y, test_y = train_test_split(data_frame['news'], data_frame['label'], test_size=test_ratio, shuffle=False)

        # encode label
        train_y = self.encode_label(train_y)
        test_y = self.encode_label(test_y)

        return data_frame['news'], train_x, test_x, train_y, test_y


    def prepare_stance_data_normal_split(self, test_ratio):

        # get data_frame that stores data and label
        data_frame = self.prepare_stance_data_frame()
        merged_data_frame = self.merge_data_frame(data_frame['penyataan'], data_frame['sumber'])

        # split the dataset into training and validation datasets
        train_penyataan, test_penyataan, train_sumber, test_sumber, train_y, test_y = train_test_split(data_frame['penyataan'], data_frame['sumber'], data_frame['label'], test_size=test_ratio, shuffle=False)

        # encode label
        train_y = self.encode_label(train_y)
        test_y = self.encode_label(test_y)

        return merged_data_frame['news'], train_penyataan, test_penyataan, train_sumber, test_sumber, train_y, test_y


    def merge_data_frame(self, data1, data2):
        data = pandas.concat([data1, data2]).sort_index(kind='merge')
        data = data.reset_index(drop=True)
        dataDF = pandas.DataFrame()
        dataDF['news'] = data
        return dataDF


