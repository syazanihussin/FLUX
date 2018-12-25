from keras.models import Sequential, Model
from keras.layers import Embedding, Dropout, Conv1D, MaxPooling1D, GRU, GlobalMaxPooling1D, Dense, SimpleRNN, LSTM, Input
from keras.callbacks import EarlyStopping, TensorBoard, CSVLogger, ModelCheckpoint
from gensim.models import Word2Vec, FastText
from keras.wrappers.scikit_learn import KerasClassifier
from keras.constraints import maxnorm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, log_loss
import keras
import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix



class Models:



    def __init__(self):
        pass



    def build_embedding_layer(self, model, vocab_size, input_length, embedding_dim, embedding_matrix, trainable, embedding_dropout):
        model.add(Embedding(input_dim=vocab_size, input_length=input_length, output_dim=embedding_dim,
                            weights=[embedding_matrix], trainable=trainable))
        model.add(Dropout(embedding_dropout))



    def build_cnn_layer(self, model, filters, window_size, bias):
        model.add(Conv1D(filters=filters, kernel_size=window_size, padding='valid', activation='relu', use_bias=bias))
        model.add(MaxPooling1D(pool_size=window_size, padding='valid'))



    def build_rnn_layer(self, model, units, bias, kernel_constraint, recurrent_constraint, dropout, recurrent_dropout, many_output, backwards):
        model.add(SimpleRNN(units=units, use_bias=bias, kernel_constraint=maxnorm(kernel_constraint),
                            recurrent_constraint=maxnorm(recurrent_constraint), dropout=dropout, recurrent_dropout=recurrent_dropout,
                            return_sequences=many_output, go_backwards=backwards))



    def build_lstm_layer(self, model, units, bias, kernel_constraint, recurrent_constraint, dropout, recurrent_dropout, many_output, backwards,
                         forget_bias):
        model.add(LSTM(units=units, use_bias=bias, kernel_constraint=maxnorm(kernel_constraint), recurrent_constraint=maxnorm(recurrent_constraint),
                       dropout=dropout, recurrent_dropout=recurrent_dropout, return_sequences=many_output, go_backwards=backwards,
                       unit_forget_bias=forget_bias))



    def build_gru_layer(self, model, units, bias, kernel_constraint, recurrent_constraint, dropout, recurrent_dropout, backwards, many_output):
        model.add(GRU(units=units, use_bias=bias, kernel_constraint=maxnorm(kernel_constraint), recurrent_constraint=maxnorm(recurrent_constraint),
                      dropout=dropout, recurrent_dropout=recurrent_dropout, return_sequences=many_output, go_backwards=backwards))



    def build_flatten_layer(self, model):
        model.add(GlobalMaxPooling1D())



    def build_softmax_layer(self, model, bias):
        model.add(Dense(2, activation='softmax', use_bias=bias))



    def train_model(self, input_type, enable_cnn, model_type, enable_flatten, train_x, train_y, test_x, test_y, epoch=10, batch=32,
                    vocab_size=None, input_length=100, embedding_dim=300, embedding_matrix=None, embedding_dropout=0.2, trainable=False,
                    filters=100, window_size=4, cnn_bias=True, recurrent_units=100, recurrent_bias=True, kernel_constraint=None,
                    recurrent_constraint=None, dropout=0.0, recurrent_dropout=0.0, many_output=None, backwards=None, softmax_bias=True,
                    forget_bias=None, save=False, file_name=None):

        # build model
        model = Sequential()

        if (input_type == 'embedding'):
            self.build_embedding_layer(model, vocab_size, input_length, embedding_dim, embedding_matrix, trainable, embedding_dropout)

        if(enable_cnn == True):
            self.build_cnn_layer(model, filters, window_size, cnn_bias)

        if(model_type == 'rnn'):
            self.build_rnn_layer(model, recurrent_units, kernel_constraint, recurrent_constraint, recurrent_bias, dropout,
                                 recurrent_dropout, many_output, backwards)

        elif(model_type == 'lstm'):
            self.build_lstm_layer(model, recurrent_units, kernel_constraint, recurrent_constraint, recurrent_bias, dropout,
                                  recurrent_dropout, many_output, backwards, forget_bias)

        elif(model_type == 'gru'):
            self.build_gru_layer(model, recurrent_units, kernel_constraint, recurrent_constraint, recurrent_bias, dropout,
                                 recurrent_dropout, backwards, many_output)

        elif (model_type == 'double gru'):
            self.build_gru_layer(model, recurrent_units, kernel_constraint, recurrent_constraint, recurrent_bias,
                                 dropout, recurrent_dropout, backwards=True, many_output=True)
            self.build_gru_layer(model, recurrent_units, kernel_constraint, recurrent_constraint, recurrent_bias,
                                 dropout, recurrent_dropout, backwards=False, many_output=False)

        if(enable_flatten == True):
            self.build_flatten_layer(model)

        self.build_softmax_layer(model, softmax_bias)

        # compile the model
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

        #callbacks = [TensorBoard(log_dir='..\logs\{}'.format(time()), histogram_freq=1, write_graph=True, write_images=True)]callbacks=callbacks,

        #CSVLogger('..\result\result3.csv', append=True), ModelCheckpoint('..\model\weights.{epoch:02d}-{val_acc:.4f}.h5', monitor='val_acc')

        # summarize the model
        print(model.summary())

        # fit the model
        model.fit(train_x, train_y, validation_data=(test_x, test_y), epochs=epoch, batch_size=batch, verbose=1)

        # display confusion matrix graph showing TP, FN, FP, TN
        self.show_confusion_matrix(model, train_x, test_x, train_y, test_y)

        # save model as HDF5 file
        if(save == True):
            model.save(file_name)

        return model



    def train_stance_model(self, heads_len, article_len, vocab_size, embedding_matrix, heads_filters, heads_window_size, arts_filters,
                           arts_window_size, gru_arts_unit, gru_heads_unit, mlp_unit, heads_train_x, arts_train_x, train_y, heads_test_x,
                           arts_test_x, test_y, save=False, file_name=None):

        #build model layers
        input_heads = Input(shape=(heads_len,))
        input_arts = Input(shape=(article_len,))

        embedding_heads = Embedding(input_dim=vocab_size, input_length=heads_len, output_dim=300, weights=[embedding_matrix], trainable=False)(input_heads)
        embedding_arts = Embedding(input_dim=vocab_size, input_length=article_len, output_dim=300, weights=[embedding_matrix], trainable=False)(input_arts)

        dropout_heads = Dropout(0.2)(embedding_heads)
        dropout_arts = Dropout(0.2)(embedding_arts)

        conv1D_heads = Conv1D(filters=heads_filters, kernel_size=heads_window_size, padding='same', activation='relu')(dropout_heads)
        conv1D_arts = Conv1D(filters=arts_filters, kernel_size=arts_window_size, padding='same', activation='relu')(dropout_arts)

        max_pooling_head = MaxPooling1D(pool_size=heads_window_size, padding='valid')(conv1D_heads)
        max_pooling_arts = MaxPooling1D(pool_size=arts_window_size, padding='valid')(conv1D_arts)

        gru_heads = GRU(units=gru_heads_unit, return_sequences=False, go_backwards=True)(max_pooling_head)
        gru_arts = GRU(units=gru_arts_unit, return_sequences=False, go_backwards=True)(max_pooling_arts)

        merged = keras.layers.concatenate([gru_heads, gru_arts])
        linear = Dense(mlp_unit, activation='tanh')(merged)
        predictions = Dense(4, activation='softmax', use_bias=True)(linear)

        model = Model(inputs=[input_heads, input_arts], outputs=[predictions])

        # compile the model
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

        # summarize the model
        print(model.summary())

        # fit the model
        model.fit([heads_train_x, arts_train_x], [train_y], validation_data=([heads_test_x, arts_test_x], [test_y]), epochs=10, verbose=1)

        # display confusion matrix graph showing TP, FN, FP, TN
        self.show_confusion_matrix(model, [heads_train_x, arts_train_x], [heads_test_x, arts_test_x], train_y, test_y)

        # save model as HDF5 file
        if(save == True):
            model.save(file_name)

        return model



    def show_confusion_matrix(self, model, train_x, test_x, train_y, test_y):

        decoded_train_y, decoded_test_y = train_y.argmax(1), test_y.argmax(1)

        predicted_train = model.predict(train_x)
        predicted_test = model.predict(test_x)

        converted_predicted_train = self.convert_proba_prediction_to_label(predicted_train)
        converted_predicted_test = self.convert_proba_prediction_to_label(predicted_test)

        train_cm = confusion_matrix(decoded_train_y, converted_predicted_train)
        test_cm = confusion_matrix(decoded_test_y, converted_predicted_test)

        # Plot non-normalized confusion matrix
        #label = ['Fake', 'Real']
        label = ['tidak setuju', 'berkaitan', 'setuju', 'tidak berkaitan']
        np.set_printoptions(precision=2)

        plt.figure()
        self.plot_confusion_matrix(train_cm, classes=label, title='Train Confusion matrix')

        plt.figure()
        self.plot_confusion_matrix(test_cm, classes=label, title='Test Confusion matrix')

        plt.show()

        return train_cm, test_cm



    def convert_proba_prediction_to_label(self, prediction):

        converted_prediction = []

        for probability in prediction:
            '''fake_prob = probability[0]
            real_prob = probability[1]

            if(fake_prob > real_prob):
                converted_prediction.append(0)
            elif(real_prob > fake_prob):
                converted_prediction.append(1)'''
            if probability[0] == max(probability):
                converted_prediction.append(0)
            elif probability[1] == max(probability):
                converted_prediction.append(1)
            elif probability[2] == max(probability):
                converted_prediction.append(2)
            elif probability[3] == max(probability):
                converted_prediction.append(3)

        return converted_prediction



    def plot_confusion_matrix(self, cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):

        """ Normalization can be applied by setting `normalize=True`."""

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()



    def grid_search(self, x, y, vocabulary_size, max_length, embedding_size, embedding_matrix):

        # create model
        grid_search_model = KerasClassifier(build_fn=self.train_model, input_type='embedding', enable_cnn=True, model_type='gru',
                                            enable_flatten=False, epoch=10, batch=32, vocab_size=vocabulary_size, input_length=max_length,
                                            embedding_dim=embedding_size, embedding_matrix=embedding_matrix, embedding_dropout=0.2,
                                            trainable=False, filters=100, window_size=4, cnn_bias=True, recurrent_units=100, recurrent_bias=True,
                                            kernel_constraint=None, recurrent_constraint=None, dropout=0.0, recurrent_dropout=0.0,
                                            many_output=False, backwards=True, softmax_bias=True, forget_bias=None, verbose=1)

        # define the grid search parameters
        units = [50, 100, 150, 200]
        param_grid = dict(recurrent_units=units)
        #score = [accuracy_score, f1_score, precision_score, recall_score, log_loss]scoring=score, refit='f1_score'

        # execute grid search
        grid = GridSearchCV(estimator=grid_search_model, param_grid=param_grid, cv=10)
        grid_result = grid.fit(x, y)

        # summarize results
        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        means_train = grid_result.cv_results_['mean_train_score']
        means_test = grid_result.cv_results_['mean_test_score']
        stds_train = grid_result.cv_results_['std_train_score']
        stds_test = grid_result.cv_results_['std_test_score']
        params = grid_result.cv_results_['params']

        for mean_train, mean_test, stdev_train, stdev_test, param in zip(means_train, means_test, stds_train, stds_test, params):
            print("MEAN TRAIN: %f (%f), MEAN TEST: %f (%f) with: %r" % (mean_train, stdev_train, mean_test, stdev_test, param))

        return grid_result.best_estimator_



    def train_word2vec_model(self, dataDF, fname):

        # train word2vec model
        model = Word2Vec(dataDF, size=300, workers=10, window=3, min_count=1, sg=1, iter=10)

        # save model
        model.save(fname)



    def train_fasttext_model(self, dataDF, fname):

        # train word2vec model
        model = FastText(dataDF, size=300, workers=10, window=3, min_count=1, sg=1, iter=10)

        # save model
        model.save(fname)



    '''def train_glove_model(self):

        # load data
        data_preparation = DataPreparation()
        data_frame = data_preparation.prepare_data_frame(239)
        clean_data_frame = data_preparation.clean_data(data_frame['data'])

        # creating a corpus object
        corpus = Corpus()

        # training the corpus to generate the co-occurence matrix which is used in GloVe
        corpus.fit(clean_data_frame, window=3)

        # train glove
        glove = Glove(no_components=300, learning_rate=0.05)
        glove.fit(corpus.matrix, epochs=10, no_threads=10, verbose=True)
        glove.add_dictionary(corpus.dictionary)

        # save model
        glove.save('glove.model')
        
        
        elif(input_type == 'tfidf'):
            train_x = train_x.reshape((-1, vocab_size, 1))
            test_x = test_x.reshape((-1, vocab_size, 1))
            model.add(Conv1D(filters=4000, kernel_size=160, padding='same', activation='relu', use_bias=True, input_shape=(vocab_size, 1)))
            model.add(MaxPooling1D(pool_size=160, padding='valid'))
            model.add(GRU(units=100, return_sequences=True))
                
        '''