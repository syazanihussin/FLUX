from development.A_DataPreparation import DataPreparation
from development.A_FeatureEngineering import FeatureEngineering
from development.A_Models import Models
import malaya


data_preparation = DataPreparation()
feature_engineering = FeatureEngineering()
model = Models()

data, train_penyataan, test_penyataan, train_sumber, test_sumber, train_y, test_y = data_preparation.prepare_stance_data_normal_split2(0.25)
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

def get_NER_features(dataframe):
    ner_features_list = []

    for news in dataframe:
        entity_network = malaya.deep_entities('entity-network')
        bahdanau = malaya.deep_entities('bahdanau')
        luong = malaya.deep_entities('luong')
        ner_features = malaya.voting_stack([entity_network, bahdanau, luong], news)
        print(type(ner_features), ner_features)
        ner_features_list.append(ner_features)
