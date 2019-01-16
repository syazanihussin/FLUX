#from development.A_DataPreparation import DataPreparation
#from development.A_FeatureEngineering import FeatureEngineering
#from development.A_Models import Models
import malaya, pickle
import matplotlib.pyplot as plt


'''
data_preparation = DataPreparation()
feature_engineering = FeatureEngineering()
model = Models()

data, train_penyataan, test_penyataan, train_sumber, test_sumber, train_y, test_y = data_preparation.prepare_stance_data_normal_split2(0.25)

pickle_out = open('train_penyataan.pickle',"wb")
pickle.dump(train_penyataan, pickle_out)

pickle_out = open('train_sumber.pickle',"wb")
pickle.dump(train_sumber, pickle_out)

pickle_out = open('test_penyataan.pickle',"wb")
pickle.dump(test_penyataan, pickle_out)

pickle_out = open('test_sumber.pickle',"wb")
pickle.dump(test_sumber, pickle_out)

pickle_out = open('train_y.pickle',"wb")
pickle.dump(train_y, pickle_out)

pickle_out = open('test_y.pickle',"wb")
pickle.dump(test_y, pickle_out)
'''
'''
vocabulary_size, embedding_matrix, tokenizer = feature_engineering.prepare_embedding_malaya()

train_penyataan_w2v = feature_engineering.tokenize_vectorize_malaya(tokenizer, data=train_penyataan, max_length=100)
test_penyataan_w2v = feature_engineering.tokenize_vectorize_malaya(tokenizer, data=test_penyataan, max_length=100)

print(train_penyataan.loc[48])
print(train_penyataan_w2v[48])

train_sumber_w2v = feature_engineering.tokenize_vectorize_malaya(tokenizer, data=train_sumber, max_length=200)
test_sumber_w2v = feature_engineering.tokenize_vectorize_malaya(tokenizer, data=test_sumber, max_length=200)

print(train_sumber.loc[48])
print(train_sumber_w2v[48])

model.train_stance_model(heads_len=100, article_len=200, vocab_size=vocabulary_size, embedding_matrix=embedding_matrix, heads_filters=100,
                         heads_window_size=4, arts_filters=200, arts_window_size=4, gru_arts_unit=100, gru_heads_unit=100, mlp_unit=100,
                         heads_train_x=train_penyataan_w2v, arts_train_x=train_sumber_w2v, train_y=train_y, heads_test_x=test_penyataan_w2v,
                         arts_test_x=test_sumber_w2v, test_y=test_y, save=False, file_name='..\model\stance_model2.h5', mode=3)
'''
'''

def get_NER_features(dataframe, filename):
    ner_features_list = []

    for news in dataframe:
        entity_network = malaya.deep_entities('entity-network')
        bahdanau = malaya.deep_entities('bahdanau')
        luong = malaya.deep_entities('luong')
        ner_features = malaya.voting_stack([entity_network, bahdanau, luong], news)
        print(type(ner_features), ner_features)
        ner_features_list.append(ner_features)

    pickle_out = open(filename,"wb")
    pickle.dump(ner_features_list, pickle_out)


get_NER_features(train_penyataan, 'train_penyataan_ner.pickle')
get_NER_features(train_sumber, 'train_sumber_ner.pickle')
get_NER_features(test_penyataan, 'test_penyataan_ner.pickle')
get_NER_features(test_sumber, 'test_sumber_ner.pickle')

'''

news = 'KUALA LUMPUR: Sempena sambutan Aidilfitri minggu depan, Perdana Menteri Tun Dr Mahathir Mohamad dan Menteri Pengangkutan Anthony Loke Siew Fook menitipkan pesanan khas kepada orang ramai yang mahu pulang ke kampung halaman masing-masing. Dalam video pendek terbitan Jabatan Keselamatan Jalan Raya (JKJR) itu, Dr Mahathir menasihati mereka supaya berhenti berehat dan tidur sebentar  sekiranya mengantuk ketika memandu.'
opinion = 'tun dr mahathir merupakan perdana menteri malaysia yang sangat hebat'

''' ENTITY RECOGNITION '''
entity_network = malaya.entity.deep_model('entity-network')
bahdanau = malaya.entity.deep_model('bahdanau')
luong = malaya.entity.deep_model('luong')

ner_features = malaya.stack.voting_stack([entity_network, bahdanau, luong], news)

print(type(ner_features), ner_features)
print(type(ner_features[0]), ner_features[0][1])

''' SUBJECTIVITY '''
xgb = malaya.subjective.xgb()
multinomial = malaya.subjective.multinomial()
hierarchical = malaya.subjective.deep_model('hierarchical')
print(multinomial.predict(opinion, get_proba=True))
print(malaya.stack.predict_stack([xgb, multinomial, hierarchical], news))
print(malaya.stack.predict_stack([xgb, multinomial, hierarchical], opinion))

''' SENTIMENT '''
bahdanau = malaya.sentiment.deep_model('bahdanau')
luong = malaya.sentiment.deep_model('luong')
entity = malaya.sentiment.deep_model('entity-network')
multinomial = malaya.sentiment.multinomial()
print(malaya.stack.predict_stack([bahdanau, luong, entity, multinomial], opinion))
print(max(malaya.stack.predict_stack([bahdanau, luong, entity, multinomial], opinion)))
print(malaya.stack.predict_stack([bahdanau, luong, entity, multinomial], news))

''' TOPIC N INFLUENCER N LOCATION '''
news = 'najib razak dan mahathir mengalami masalah air di kemamam terengganu'
print(malaya.topic_influencer.fuzzy_topic(opinion))
print(malaya.topic_influencer.fuzzy_influencer(opinion))

''' LANGUAGE '''
xgb = malaya.language_detection.xgb()
multinomial = malaya.language_detection.multinomial()
sgd = malaya.language_detection.sgd()
print(malaya.stack.predict_stack([xgb, multinomial, sgd], opinion))