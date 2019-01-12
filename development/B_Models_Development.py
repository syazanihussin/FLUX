from development.A_DataPreparation import DataPreparation
from development.A_FeatureEngineering import FeatureEngineering
from development.A_Models import Models


data_preparation = DataPreparation()
feature_engineering = FeatureEngineering()
model = Models()

data, train_x, test_x, train_y, test_y = data_preparation.prepare_data_normal_split(test_ratio=0.25)

#model.train_word2vec_model(dataDF=data['news'], fname='..\model\word2vec.model')

vocabulary_size, embedding_matrix, tokenizer = feature_engineering.prepare_embedding('use', data, initial_vocab_size=1000000,
                                                                                     embedding_size=300, embedding_type='word2vec',
                                                                                     embedding_dir='..\model\content_word2vec.model',
                                                                                     tokenizer_dir='..\model\content_word2vec_tokenizer.pickle',
                                                                                     embedding_matrix_dir='..\model\content_embedding_matrix.npy')

train_x = feature_engineering.tokenize_vectorize(tokenizer, data=train_x, max_length=100)
test_x = feature_engineering.tokenize_vectorize(tokenizer, data=test_x, max_length=100)

'''
print('1. MODEL CNN-RNN ADOPTED : WORD2VEC')
model.train_model(input_type='embedding', enable_cnn=True, model_type='rnn', enable_flatten=True, train_x=train_x, train_y=train_y,
                  test_x=test_x, test_y=test_y, vocab_size=vocabulary_size, embedding_matrix=embedding_matrix, many_output=True,
                  backwards=False)


print('2. MODEL CNN-RNN ADOPTED BACKWARD : WORD2VEC')
model.train_model(input_type='embedding', enable_cnn=True, model_type='rnn', enable_flatten=True, train_x=train_x, train_y=train_y,
                  test_x=test_x, test_y=test_y, vocab_size=vocabulary_size, embedding_matrix=embedding_matrix, many_output=True,
                  backwards=True)


print('3. MODEL CNN-LSTM FORGET BIAS TRUE ADOPTED : WORD2VEC')
model.train_model(input_type='embedding', enable_cnn=True, model_type='lstm', enable_flatten=True, train_x=train_x, train_y=train_y,
                  test_x=test_x, test_y=test_y, vocab_size=vocabulary_size, embedding_matrix=embedding_matrix, many_output=True,
                  backwards=False, forget_bias=True)


print('4. MODEL CNN-LSTM FORGET BIAS TRUE ADOPTED BACKWARD : WORD2VEC')
model.train_model(input_type='embedding', enable_cnn=True, model_type='lstm', enable_flatten=True, train_x=train_x, train_y=train_y,
                  test_x=test_x, test_y=test_y, vocab_size=vocabulary_size, embedding_matrix=embedding_matrix, many_output=True,
                  backwards=True, forget_bias=True)


print('5. MODEL CNN-LSTM FORGET BIAS FALSE ADOPTED : WORD2VEC')
model.train_model(input_type='embedding', enable_cnn=True, model_type='lstm', enable_flatten=True, train_x=train_x, train_y=train_y,
                  test_x=test_x, test_y=test_y, vocab_size=vocabulary_size, embedding_matrix=embedding_matrix, many_output=True,
                  backwards=False, forget_bias=False)


print('6. MODEL CNN-LSTM FORGET BIAS FALSE ADOPTED BACKWARD : WORD2VEC')
model.train_model(input_type='embedding', enable_cnn=True, model_type='lstm', enable_flatten=True, train_x=train_x, train_y=train_y,
                  test_x=test_x, test_y=test_y, vocab_size=vocabulary_size, embedding_matrix=embedding_matrix, many_output=True,
                  backwards=True, forget_bias=False)


print('7. MODEL CNN-GRU ADOPTED : WORD2VEC')
model.train_model(input_type='embedding', enable_cnn=True, model_type='gru', enable_flatten=False, train_x=train_x, train_y=train_y,
                  test_x=test_x, test_y=test_y, vocab_size=vocabulary_size, embedding_matrix=embedding_matrix, many_output=False,
                  backwards=False)

'''
print('8. MODEL CNN-GRU ADOPTED BACKWARD: WORD2VEC ')
model.train_model(input_type='embedding', enable_cnn=True, model_type='gru', enable_flatten=True, train_x=train_x, train_y=train_y,
                  test_x=test_x, test_y=test_y, vocab_size=vocabulary_size, embedding_matrix=embedding_matrix, many_output=True,
                  backwards=True)
'''

print('9. MODEL CNN-RNN : WORD2VEC')
model.train_model(input_type='embedding', enable_cnn=True, model_type='rnn', enable_flatten=False, train_x=train_x, train_y=train_y,
                  test_x=test_x, test_y=test_y, vocab_size=vocabulary_size, embedding_matrix=embedding_matrix, many_output=False,
                  backwards=False)


print('10. MODEL CNN-RNN BACKWARD : WORD2VEC')
model.train_model(input_type='embedding', enable_cnn=True, model_type='rnn', enable_flatten=False, train_x=train_x, train_y=train_y,
                  test_x=test_x, test_y=test_y, vocab_size=vocabulary_size, embedding_matrix=embedding_matrix, many_output=False,
                  backwards=True)


print('11. MODEL CNN-LSTM FORGET BIAS TRUE : WORD2VEC')
model.train_model(input_type='embedding', enable_cnn=True, model_type='lstm', enable_flatten=False, train_x=train_x, train_y=train_y,
                  test_x=test_x, test_y=test_y, vocab_size=vocabulary_size, embedding_matrix=embedding_matrix, many_output=False,
                  backwards=False, forget_bias=True)


print('12. MODEL CNN-LSTM FORGET BIAS TRUE BACKWARD : WORD2VEC')
model.train_model(input_type='embedding', enable_cnn=True, model_type='lstm', enable_flatten=False, train_x=train_x, train_y=train_y,
                  test_x=test_x, test_y=test_y, vocab_size=vocabulary_size, embedding_matrix=embedding_matrix, many_output=False,
                  backwards=True, forget_bias=True)


print('13. MODEL CNN-LSTM FORGET BIAS FALSE : WORD2VEC')
model.train_model(input_type='embedding', enable_cnn=True, model_type='lstm', enable_flatten=False, train_x=train_x, train_y=train_y,
                  test_x=test_x, test_y=test_y, vocab_size=vocabulary_size, embedding_matrix=embedding_matrix, many_output=False,
                  backwards=False, forget_bias=False)


print('14. MODEL CNN-LSTM FORGET BIAS FALSE BACKWARD : WORD2VEC')
model.train_model(input_type='embedding', enable_cnn=True, model_type='lstm', enable_flatten=False, train_x=train_x, train_y=train_y,
                  test_x=test_x, test_y=test_y, vocab_size=vocabulary_size, embedding_matrix=embedding_matrix, many_output=False,
                  backwards=True, forget_bias=False)


print('15. MODEL CNN-GRU : WORD2VEC')
model.train_model(input_type='embedding', enable_cnn=True, model_type='gru', enable_flatten=False, train_x=train_x, train_y=train_y,
                  test_x=test_x, test_y=test_y, vocab_size=vocabulary_size, embedding_matrix=embedding_matrix, many_output=False,
                  backwards=False)

'''
print('16. MODEL CNN-GRU BACKWARD: WORD2VEC ')
model.train_model(input_type='embedding', enable_cnn=True, model_type='gru', enable_flatten=False, train_x=train_x, train_y=train_y,
                  test_x=test_x, test_y=test_y, vocab_size=vocabulary_size, embedding_matrix=embedding_matrix, many_output=False,
                  backwards=True)
'''
print('17. MODEL CNN-MULTILAYER GRU BACKWARD: WORD2VEC ')
model.train_model(input_type='embedding', enable_cnn=True, model_type='double gru', enable_flatten=False, train_x=train_x, train_y=train_y,
                  test_x=test_x, test_y=test_y, vocab_size=vocabulary_size, embedding_matrix=embedding_matrix, backwards=True)


print('18. MODEL CNN : WORD2VEC')
model.train_model(input_type='embedding', enable_cnn=True, model_type='', enable_flatten=True, train_x=train_x, train_y=train_y,
                  test_x=test_x, test_y=test_y, vocab_size=vocabulary_size, embedding_matrix=embedding_matrix)


print('19. MODEL RNN : WORD2VEC')
model.train_model(input_type='embedding', enable_cnn=False, model_type='rnn', enable_flatten=False, train_x=train_x, train_y=train_y,
                  test_x=test_x, test_y=test_y, vocab_size=vocabulary_size, embedding_matrix=embedding_matrix, many_output=False,
                  backwards=False)


print('20. MODEL RNN BACKWARD : WORD2VEC')
model.train_model(input_type='embedding', enable_cnn=False, model_type='rnn', enable_flatten=False, train_x=train_x, train_y=train_y,
                  test_x=test_x, test_y=test_y, vocab_size=vocabulary_size, embedding_matrix=embedding_matrix, many_output=False,
                  backwards=True)


print('21. MODEL RNN MANY : WORD2VEC')
model.train_model(input_type='embedding', enable_cnn=False, model_type='rnn', enable_flatten=True, train_x=train_x, train_y=train_y,
                  test_x=test_x, test_y=test_y, vocab_size=vocabulary_size, embedding_matrix=embedding_matrix, many_output=True,
                  backwards=False)


print('22. MODEL RNN MANY BACKWARD : WORD2VEC')
model.train_model(input_type='embedding', enable_cnn=False, model_type='rnn', enable_flatten=True, train_x=train_x, train_y=train_y,
                  test_x=test_x, test_y=test_y, vocab_size=vocabulary_size, embedding_matrix=embedding_matrix, many_output=True,
                  backwards=True)


print('23. MODEL LSTM FORGET BIAS TRUE : WORD2VEC')
model.train_model(input_type='embedding', enable_cnn=False, model_type='lstm', enable_flatten=False, train_x=train_x, train_y=train_y,
                  test_x=test_x, test_y=test_y, vocab_size=vocabulary_size, embedding_matrix=embedding_matrix, many_output=False,
                  backwards=False, forget_bias=True)


print('24. MODEL LSTM FORGET BIAS TRUE BACKWARD : WORD2VEC')
model.train_model(input_type='embedding', enable_cnn=False, model_type='lstm', enable_flatten=False, train_x=train_x, train_y=train_y,
                  test_x=test_x, test_y=test_y, vocab_size=vocabulary_size, embedding_matrix=embedding_matrix, many_output=False,
                  backwards=True, forget_bias=True)


print('25. MODEL LSTM FORGET BIAS TRUE MANY : WORD2VEC')
model.train_model(input_type='embedding', enable_cnn=False, model_type='lstm', enable_flatten=True, train_x=train_x, train_y=train_y,
                  test_x=test_x, test_y=test_y, vocab_size=vocabulary_size, embedding_matrix=embedding_matrix, many_output=True,
                  backwards=False, forget_bias=True)


print('26. MODEL LSTM FORGET BIAS TRUE MANY BACKWARD : WORD2VEC')
model.train_model(input_type='embedding', enable_cnn=False, model_type='lstm', enable_flatten=True, train_x=train_x, train_y=train_y,
                  test_x=test_x, test_y=test_y, vocab_size=vocabulary_size, embedding_matrix=embedding_matrix, many_output=True,
                  backwards=True, forget_bias=True)


print('27. MODEL LSTM FORGET BIAS FALSE : WORD2VEC')
model.train_model(input_type='embedding', enable_cnn=False, model_type='lstm', enable_flatten=False, train_x=train_x, train_y=train_y,
                  test_x=test_x, test_y=test_y, vocab_size=vocabulary_size, embedding_matrix=embedding_matrix, many_output=False,
                  backwards=False, forget_bias=False)


print('28. MODEL LSTM FORGET BIAS FALSE BACKWARD : WORD2VEC')
model.train_model(input_type='embedding', enable_cnn=False, model_type='lstm', enable_flatten=False, train_x=train_x, train_y=train_y,
                  test_x=test_x, test_y=test_y, vocab_size=vocabulary_size, embedding_matrix=embedding_matrix, many_output=False,
                  backwards=True, forget_bias=False)


print('29. MODEL LSTM FORGET BIAS FALSE MANY : WORD2VEC')
model.train_model(input_type='embedding', enable_cnn=False, model_type='lstm', enable_flatten=True, train_x=train_x, train_y=train_y,
                  test_x=test_x, test_y=test_y, vocab_size=vocabulary_size, embedding_matrix=embedding_matrix, many_output=True,
                  backwards=False, forget_bias=False)


print('30. MODEL LSTM FORGET BIAS FALSE MANY BACKWARD : WORD2VEC')
model.train_model(input_type='embedding', enable_cnn=False, model_type='lstm', enable_flatten=True, train_x=train_x, train_y=train_y,
                  test_x=test_x, test_y=test_y, vocab_size=vocabulary_size, embedding_matrix=embedding_matrix, many_output=True,
                  backwards=True, forget_bias=False)

'''
print('31. MODEL GRU : WORD2VEC')
model.train_model(input_type='embedding', enable_cnn=False, model_type='gru', enable_flatten=False, train_x=train_x, train_y=train_y,
                  test_x=test_x, test_y=test_y, vocab_size=vocabulary_size, embedding_matrix=embedding_matrix, many_output=False,
                  backwards=False)


print('32. MODEL GRU BACKWARD : WORD2VEC')
model.train_model(input_type='embedding', enable_cnn=False, model_type='gru', enable_flatten=False, train_x=train_x, train_y=train_y,
                  test_x=test_x, test_y=test_y, vocab_size=vocabulary_size, embedding_matrix=embedding_matrix, many_output=False,
                  backwards=True)


print('33. MODEL MULTILAYER GRU BACKWARD : WORD2VEC')
model.train_model(input_type='embedding', enable_cnn=False, model_type='double gru', enable_flatten=False, train_x=train_x, train_y=train_y,
                  test_x=test_x, test_y=test_y, vocab_size=vocabulary_size, embedding_matrix=embedding_matrix)


print('34. MODEL GRU MANY : WORD2VEC')
model.train_model(input_type='embedding', enable_cnn=False, model_type='gru', enable_flatten=True, train_x=train_x, train_y=train_y,
                  test_x=test_x, test_y=test_y, vocab_size=vocabulary_size, embedding_matrix=embedding_matrix, many_output=True,
                  backwards=False)


print('35. MODEL GRU MANY BACKWARD : WORD2VEC')
model.train_model(input_type='embedding', enable_cnn=False, model_type='gru', enable_flatten=True, train_x=train_x, train_y=train_y,
                  test_x=test_x, test_y=test_y, vocab_size=vocabulary_size, embedding_matrix=embedding_matrix, many_output=True,
                  backwards=True)




'''
def prepare_for_grid_search():
    train = np.concatenate((train_x, test_x), axis=0)
    test = np.concatenate((train_y, test_y), axis=0)

    model = Models()
    best_model = model.grid_search(train, test, vocabulary_size, max_length=100, embedding_size=300, embedding_matrix)
    loss, accuracy = best_model.evaluate(test_x, test_y, verbose=1)
    print('Accuracy: %f, Loss: %f' % (accuracy * 100, loss))


vocabulary_size, train_x, test_x = feature_engineering.prepare_count_vectorizer(data, train_x, test_x, initial_vocab_size=4000)
model.train_cnn_gru_model(train_x, train_y, test_x, test_y, vocab_size=4000, model_type ='cnn_gru', input_type='tfidf')

'''