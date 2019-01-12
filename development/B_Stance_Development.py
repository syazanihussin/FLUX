from development.A_DataPreparation import DataPreparation
from development.A_FeatureEngineering import FeatureEngineering
from development.A_Models import Models
#from keras.models import load_model


data_preparation = DataPreparation()
feature_engineering = FeatureEngineering()
model = Models()


''' CONTENT BASED FAKE NEWS DETECTION '''

data, train_x, test_x, train_y, test_y = data_preparation.prepare_data_normal_split2(test_ratio=0.25)

#model.train_word2vec_model(dataDF=data, fname='..\model\word2vec.model')
'''
vocabulary_size, embedding_matrix, tokenizer = feature_engineering.prepare_embedding('use', data, initial_vocab_size=1000000,
                                                                                     embedding_size=300, embedding_type='word2vec',
                                                                                     embedding_dir='..\model\content_word2vec.model',
                                                                                     tokenizer_dir='..\model\content_word2vec_tokenizer.pickle',
                                                                                     embedding_matrix_dir='..\model\content_embedding_matrix.npy')


print(vocabulary_size, embedding_matrix.shape)

train_x_w2v = feature_engineering.tokenize_vectorize(tokenizer, data=train_x, max_length=100)
test_x_w2v = feature_engineering.tokenize_vectorize(tokenizer, data=test_x, max_length=100)
'''
'''
model.train_model(input_type='embedding', enable_cnn=True, model_type='gru', enable_flatten=False, train_x=train_x_w2v, train_y=train_y,
                  test_x=test_x_w2v, test_y=test_y, vocab_size=vocabulary_size, embedding_matrix=embedding_matrix, many_output=False,
                  backwards=True, save=False, file_name='..\model\content_model.h5')


content_model = load_model('..\model\content_model.h5')
print(content_model.evaluate(train_x_w2v, train_y))
print(content_model.evaluate(test_x_w2v, test_y))
model.show_confusion_matrix(content_model, train_x_w2v, test_x_w2v, train_y, test_y)
'''

''' STANCE BASED FAKE NEWS DETECTION '''

data2, train_penyataan, test_penyataan, train_sumber, test_sumber, train_y2, test_y2 = data_preparation.prepare_stance_data_normal_split2(test_ratio=0.25)

#model.train_word2vec_model(dataDF=data, fname='..\model\stance_word2vec.model')
'''
vocabulary_size, embedding_matrix, tokenizer = feature_engineering.prepare_embedding('use', data, initial_vocab_size=1000000,
                                                                                     embedding_size=300, embedding_type='word2vec',
                                                                                     embedding_dir='..\model\stance_word2vec.model',
                                                                                     tokenizer_dir='..\model\stance_word2vec_tokenizer.pickle',
                                                                                     embedding_matrix_dir='..\model\stance_embedding_matrix.npy')
print(vocabulary_size, embedding_matrix.shape)

train_penyataan_w2v = feature_engineering.tokenize_vectorize(tokenizer, data=train_penyataan, max_length=100)
test_penyataan_w2v = feature_engineering.tokenize_vectorize(tokenizer, data=test_penyataan, max_length=100)

train_sumber_w2v = feature_engineering.tokenize_vectorize(tokenizer, data=train_sumber, max_length=2000)
test_sumber_w2v = feature_engineering.tokenize_vectorize(tokenizer, data=test_sumber, max_length=2000)
'''
'''
model.train_stance_model(heads_len=100, article_len=2000, vocab_size=vocabulary_size, embedding_matrix=embedding_matrix, heads_filters=100,
                         heads_window_size=4, arts_filters=2000, arts_window_size=4, gru_arts_unit=100, gru_heads_unit=100, mlp_unit=100,
                         heads_train_x=train_penyataan_w2v, arts_train_x=train_sumber_w2v, train_y=train_y, heads_test_x=test_penyataan_w2v,
                         arts_test_x=test_sumber_w2v, test_y=test_y, save=True, file_name='..\model\stance_model.h5')

stance_model = load_model('..\model\stance_model.h5')
print(stance_model.evaluate([train_penyataan_w2v, train_sumber_w2v], train_y))
print(stance_model.evaluate([test_penyataan_w2v, test_sumber_w2v], test_y))
model.show_confusion_matrix(stance_model, [train_penyataan_w2v, train_sumber_w2v], [test_penyataan_w2v, test_sumber_w2v], train_y, test_y)
'''
