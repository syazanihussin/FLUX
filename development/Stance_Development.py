from development.DataPreparation import DataPreparation
from development.FeatureEngineering import FeatureEngineering
from development.Models import Models
from keras.models import load_model
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


data_preparation = DataPreparation()
feature_engineering = FeatureEngineering()
model = Models()


''' CONTENT BASED FAKE NEWS DETECTION '''

data, train_x, test_x, train_y, test_y = data_preparation.prepare_data_normal_split(test_ratio=0.25)

#model.train_word2vec_model(dataDF=data, fname='..\model\word2vec.model')

vocabulary_size, embedding_matrix, tokenizer = feature_engineering.prepare_embedding('use', data, initial_vocab_size=1000000,
                                                                                     embedding_size=300, embedding_type='word2vec',
                                                                                     embedding_dir='..\model\content_word2vec.model',
                                                                                     tokenizer_dir='..\model\content_word2vec_tokenizer.pickle',
                                                                                     embedding_matrix_dir='..\model\content_embedding_matrix.npy')

print(vocabulary_size, embedding_matrix.shape)
train_x_w2v = feature_engineering.tokenize_vectorize(tokenizer, data=train_x, max_length=100)
test_x_w2v = feature_engineering.tokenize_vectorize(tokenizer, data=test_x, max_length=100)

model.train_model(input_type='embedding', enable_cnn=True, model_type='gru', enable_flatten=False, train_x=train_x_w2v, train_y=train_y,
                  test_x=test_x_w2v, test_y=test_y, vocab_size=vocabulary_size, embedding_matrix=embedding_matrix, many_output=False,
                  backwards=True, save=False, file_name='..\model\content_model.h5')

'''
content_model = load_model('..\model\content_model.h5')
print(content_model.evaluate(train_x_w2v, train_y))
print(content_model.evaluate(test_x_w2v, test_y))
model.show_confusion_matrix(content_model, train_x_w2v, test_x_w2v, train_y, test_y)
'''

''' STANCE BASED FAKE NEWS DETECTION '''
'''
data, train_penyataan, test_penyataan, train_sumber, test_sumber, train_y, test_y = data_preparation.prepare_stance_data_normal_split(test_ratio=0.25)

#model.train_word2vec_model(dataDF=data, fname='..\model\stance_word2vec.model')

vocabulary_size, embedding_matrix, tokenizer = feature_engineering.prepare_embedding('use', data, initial_vocab_size=1000000,
                                                                                     embedding_size=300, embedding_type='word2vec',
                                                                                     embedding_dir='..\model\stance_word2vec.model',
                                                                                     tokenizer_dir='..\model\stance_word2vec_tokenizer.pickle',
                                                                                     embedding_matrix_dir='..\model\stance_embedding_matrix.npy')
print(vocabulary_size, embedding_matrix.shape)
train_penyataan_w2v = feature_engineering.tokenize_vectorize(tokenizer, data=train_penyataan, max_length=100)
test_penyataan_w2v = feature_engineering.tokenize_vectorize(tokenizer, data=test_penyataan, max_length=100)

train_sumber_w2v = feature_engineering.tokenize_vectorize(tokenizer, data=train_sumber, max_length=2000)
test_sumber_w2v = feature_engineering.tokenize_vectorize(tokenizer, data=test_sumber, max_length=2000)'''
'''
model.train_stance_model(heads_len=100, article_len=2000, vocab_size=vocabulary_size, embedding_matrix=embedding_matrix, heads_filters=100,
                         heads_window_size=4, arts_filters=2000, arts_window_size=4, gru_arts_unit=100, gru_heads_unit=100, mlp_unit=100,
                         heads_train_x=train_penyataan_w2v, arts_train_x=train_sumber_w2v, train_y=train_y, heads_test_x=test_penyataan_w2v,
                         arts_test_x=test_sumber_w2v, test_y=test_y, save=True, file_name='..\model\stance_model.h5')
'''
'''
stance_model = load_model('..\model\stance_model.h5')
print(stance_model.evaluate([train_penyataan_w2v, train_sumber_w2v], train_y))
print(stance_model.evaluate([test_penyataan_w2v, test_sumber_w2v], test_y))
model.show_confusion_matrix(stance_model, [train_penyataan_w2v, train_sumber_w2v], [test_penyataan_w2v, test_sumber_w2v], train_y, test_y)
'''

'''
# calculate acc for data stance out of knowledge
def detect_using_stance_consine_similarity(penyataan, sumber):

    title_searched_results = []
    title_searched_results.append(penyataan)
    title_searched_results.append(sumber)

    tfidf_vectorizer = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}')
    tfidf_matrix_train = tfidf_vectorizer.fit_transform(title_searched_results)

    cs = cosine_similarity(tfidf_matrix_train[0:1], tfidf_matrix_train)

    real_count, fake_count = 0, 0

    print(title_searched_results[1])
    print(cs[0][1])

    if cs[0][1] > 0.15:
        fake_count += 1
    else:
        real_count += 1

    print('Real: ', real_count, 'Fake: ', fake_count)

    label = ''

    if real_count > fake_count:
        label = 'Real'

    elif fake_count > real_count:
        label = 'Fake'

    elif fake_count == real_count:
        label = 'Real'

    return label


resultsss_train = []
for i in range(len(train_penyataan)):
    resultsss_train.append(detect_using_stance_consine_similarity(train_penyataan.values[i], train_sumber.values[i])),

resultsss_test = []
for i in range(len(test_penyataan)):
    resultsss_test.append(detect_using_stance_consine_similarity(test_penyataan.values[i], test_sumber.values[i]))

print(len(resultsss_train), resultsss_train)
print(len(resultsss_test), resultsss_test)


def cal_acc(size,label_correct, labels):
    count_correct = 0

    for i in range(size):
        if label_correct[i] == labels[i]:
            count_correct += 1

    print(count_correct/size)



cal_acc(750, resultsss_train, train_y.values)
cal_acc(250, resultsss_test, test_y.values)
'''