#from development.DataPreparation import DataPreparation
#from development.FeatureEngineering import FeatureEngineering
#from development.Models import Models
#from system import search
import csv, pandas
import malaya
from sklearn.model_selection import train_test_split

'''
index = []
for i in range(120):
    if i%2!=0:
        index.append(i)

print(index)
penyataan_data = pandas.read_csv('../data/penyataan_csv.csv', skiprows=index)
print(penyataan_data)


searching_obj = search.Searching()

with open('../data/sumber_csv.csv', mode='w', encoding='utf-8') as file:
    file_writer = csv.writer(file, delimiter=',', lineterminator = '\n')

    for i in range(len(penyataan_data)):
        searched_results = searching_obj.search_news(keyword=penyataan_data['penyataan'][i])

        for searched_data in searched_results:
            print(searched_data['link'])
            print(searched_data['title'])
            print(searched_data['snippet'])
            print()
            key = i*2+1
            file_writer.writerow([key, searched_data['title'], searched_data['snippet'], searched_data['link']])


'''
'''
data_preparation = DataPreparation()
feature_engineering = FeatureEngineering()

model = Models()


content_data = pandas.read_csv('../data/content_csv.csv')
stance_data = pandas.read_csv('../data/stance_csv.csv')


print('RAW DATA')
print(content_data[['penyataan', 'label']])
print(stance_data[['penyataan','sumber', 'label']])

count_palsu_s, count_benar_s, count_xberkaitan_s, count_berkaitan_s = 0,0,0,0


for i in range(len(stance_data)):
    print(i)
    if(stance_data['label'].loc[i] == 'tidak setuju'):
        count_palsu_s+=1
    elif(stance_data['label'].loc[i] == 'setuju'):
        count_benar_s+=1
    elif(stance_data['label'].loc[i] == 'tidak berkaitan'):
        count_xberkaitan_s+=1
    elif(stance_data['label'].loc[i] == 'berkaitan'):
        count_berkaitan_s+=1


print(count_benar_s, count_palsu_s,count_berkaitan_s, count_xberkaitan_s)
'''
'''
# clean and lowercase data in dataDF
data_preparation.clean_data(stance_data['penyataan'], loop=len(content_data))

data_preparation.clean_data(stance_data['penyataan'], loop=len(stance_data))
data_preparation.clean_data(stance_data['sumber'], loop=len(stance_data))




for i in range(len(stance_data)):
    stance_data['penyataan'].loc[i] = malaya.sastrawi_stemmer(stance_data['penyataan'].loc[i])
    stance_data['sumber'].loc[i] = malaya.sastrawi_stemmer(stance_data['sumber'].loc[i])

for i in range(len(content_data)):
    content_data['penyataan'].loc[i] = malaya.sastrawi_stemmer(content_data['penyataan'].loc[i])
    
    
print('AFTER CLEANING AND STEMMING')
print(stance_data[['penyataan','sumber']])




''''''
# split the dataset into training and validation datasets
train_penyataan, test_penyataan, train_sumber, test_sumber, train_y, test_y = train_test_split(stance_data['penyataan'], stance_data['sumber'], stance_data['label'], test_size=0.25, shuffle=False)

# encode label
train_y = data_preparation.encode_label(train_y)
test_y = data_preparation.encode_label(test_y)

data = data_preparation.merge_data_frame(stance_data['penyataan'], stance_data['sumber'])
#model.train_word2vec_model(dataDF=data['news'], fname='..\model\stance_word2vec2.model')

vocabulary_size, embedding_matrix, tokenizer = feature_engineering.prepare_embedding('use', data['news'], initial_vocab_size=1000000,
                                                                                     embedding_size=300, embedding_type='word2vec',
                                                                                     embedding_dir='..\model\stance_word2vec2.model',
                                                                                     tokenizer_dir='..\model\stance_word2vec_tokenizer2.pickle',
                                                                                     embedding_matrix_dir='..\model\stance_embedding_matrix2.npy')

print(vocabulary_size, embedding_matrix.shape)
train_penyataan_w2v = feature_engineering.tokenize_vectorize(tokenizer, data=train_penyataan, max_length=100)
test_penyataan_w2v = feature_engineering.tokenize_vectorize(tokenizer, data=test_penyataan, max_length=100)

train_sumber_w2v = feature_engineering.tokenize_vectorize(tokenizer, data=train_sumber, max_length=100)
test_sumber_w2v = feature_engineering.tokenize_vectorize(tokenizer, data=test_sumber, max_length=100)

model.train_stance_model(heads_len=100, article_len=100, vocab_size=vocabulary_size, embedding_matrix=embedding_matrix, heads_filters=100,
                         heads_window_size=4, arts_filters=100, arts_window_size=4, gru_arts_unit=100, gru_heads_unit=100, mlp_unit=100,
                         heads_train_x=train_penyataan_w2v, arts_train_x=train_sumber_w2v, train_y=train_y, heads_test_x=test_penyataan_w2v,
                         arts_test_x=test_sumber_w2v, test_y=test_y, save=False, file_name='..\model\stance_model2.h5')





'''
