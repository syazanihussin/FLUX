from development.A_DataPreparation import DataPreparation
from development.A_FeatureEngineering import FeatureEngineering
from development.A_Models import Models
from keras.preprocessing.text import text_to_word_sequence
from sklearn.feature_selection import chi2
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import numpy as np
import malaya
import csv


''' COUNT TOTAL VOCAB & AVERAGE LENGTH '''

data_preparation = DataPreparation()
feature_engineering = FeatureEngineering()
model = Models()


scraped_content_data, train_x, test_x, train_y, test_y = data_preparation.prepare_data_normal_split(0.25)
scraped_stance_data, train_penyataan, test_penyataan, train_sumber, test_sumber, train_y2, test_y2 = data_preparation.prepare_stance_data_normal_split(0.25)

print(scraped_content_data)
print(scraped_stance_data)
print(len(scraped_content_data), len(scraped_stance_data))


scraped_content_list, scraped_stance_list = [], []
total_fake, total_real, total_news_avg = 0, 0, 0
total_fake2, total_real2, total_news_avg2 = 0, 0, 0


# VOCAB BEFORE STEMMING #
for i in range(len(scraped_content_data)):
    tokens = text_to_word_sequence(scraped_content_data[i], filters='!"#$%&()*+,-./:;<=>?@[\]“”^_`{|}~‘’''\t\n\xa0â\x80\x9c\x9d\'')

    for token in tokens:
        scraped_content_list.append(token)

    if i%2==0:
        total_fake += len(scraped_content_data[i].split())
    elif i%2!=0:
        total_real += len(scraped_content_data[i].split())

    total_news_avg += total_fake + total_real


for i in range(len(scraped_stance_data)):
    tokens = text_to_word_sequence(scraped_stance_data[i], filters='!"#$%&()*+,-./:;<=>?@[\]“”^_`{|}~‘’''\t\n\xa0â\x80\x9c\x9d\'')

    for token in tokens:
        scraped_stance_list.append(token)

    if i%2==0:
        total_fake2 += len(scraped_stance_data[i].split())
    elif i%2!=0:
        total_real2 += len(scraped_stance_data[i].split())

    total_news_avg2 += total_fake2 + total_real2


print(len(set(scraped_content_list)), set(scraped_content_list))
print(total_fake, total_real, total_news_avg)
print('avg fake news: ', total_fake/(len(scraped_content_data)/2))
print('avg real news: ', total_real/(len(scraped_content_data)/2))
print('avg total: ', total_news_avg/len(scraped_content_data))

print(len(set(scraped_stance_list)), set(scraped_stance_list))
print(total_fake2, total_real2, total_news_avg2)
print('avg fake news: ', total_fake2/(len(scraped_stance_data)/2))
print('avg real news: ', total_real2/(len(scraped_stance_data)/2))
print('avg total: ', total_news_avg2/len(scraped_stance_data))


scraped_content_list, scraped_stance_list = [], []
total_fake, total_real, total_news_avg = 0, 0, 0
total_fake2, total_real2, total_news_avg2 = 0, 0, 0


# VOCAB AFTER STEMMING #
for i in range(len(scraped_content_data)):
    news = malaya.sastrawi_stemmer(scraped_content_data[i])
    tokens = text_to_word_sequence(news, filters='!"#$%&()*+,-./:;<=>?@[\]“”^_`{|}~‘’''\t\n\xa0â\x80\x9c\x9d\'')

    for token in tokens:
        scraped_content_list.append(token)

    if i%2==0:
        total_fake += len(news.split())
    elif i%2!=0:
        total_real += len(news.split())

    total_news_avg += total_fake + total_real


for i in range(len(scraped_stance_data)):
    news = malaya.sastrawi_stemmer(scraped_stance_data[i])
    tokens = text_to_word_sequence(news, filters='!"#$%&()*+,-./:;<=>?@[\]“”^_`{|}~‘’''\t\n\xa0â\x80\x9c\x9d\'')

    for token in tokens:
        scraped_stance_list.append(token)

    if i%2==0:
        total_fake2 += len(news.split())
    elif i%2!=0:
        total_real2 += len(news.split())

    total_news_avg2 += total_fake2 + total_real2


print(len(set(scraped_content_list)), set(scraped_content_list))
print(total_fake, total_real, total_news_avg)
print('avg fake news: ', total_fake/(len(scraped_content_data)/2))
print('avg real news: ', total_real/(len(scraped_content_data)/2))
print('avg total: ', total_news_avg/len(scraped_content_data))

print(len(set(scraped_stance_list)), set(scraped_stance_list))
print(total_fake2, total_real2, total_news_avg2)
print('avg fake news: ', total_fake2/(len(scraped_stance_data)/2))
print('avg real news: ', total_real2/(len(scraped_stance_data)/2))
print('avg total: ', total_news_avg2/len(scraped_stance_data))



''' CREATE WORD VOCAB '''

def word_count(str):
    counts = dict()
    words = str.split()

    for word in words:
        if word in counts:
            counts[word] += 1
        else:
            counts[word] = 1

    return counts



''' TEST DIFFERENT EMBEDDING PERFORMANCE ON DIFFERENT DATASET '''

embedded = malaya.malaya_word2vec(256)

print(len(embedded['dictionary']), embedded['nce_weights'].shape)
word_vector = malaya.Word2Vec(embedded['nce_weights'], embedded['dictionary'])



''' FEATURE SELECTION '''

tvec = TfidfVectorizer(max_features=100000, ngram_range=(1, 3))
x_train_tfidf = tvec.fit_transform(train_x)
chi2score = chi2(x_train_tfidf, train_y)[0]

plt.figure(figsize=(15,10))
wscores = zip(tvec.get_feature_names(), chi2score)
wchi2 = sorted(wscores, key=lambda x:x[1])
topchi2 = list(zip(*wchi2[-20:]))
x = range(len(topchi2[1]))
labels = topchi2[0]
plt.barh(x,topchi2[1], align='center', alpha=0.2)
plt.plot(topchi2[1], x, '-o', markersize=5, alpha=0.8)
plt.yticks(x, labels)
plt.xlabel('$\chi^2$')


tvec = TfidfVectorizer(max_features=100000, ngram_range=(1, 3))
x_train_pen = tvec.fit_transform(train_penyataan)
x_train_sum = tvec.fit_transform(train_sumber)

x_train_pen = x_train_pen.toarray()
x_train_sum = x_train_sum.toarray()
concatenated = []
for data in range(len(x_train_sum)):
    concatenated.append(np.concatenate((x_train_pen, x_train_sum), axis=1))

chi2score = chi2(concatenated, train_y2)[0]

plt.figure(figsize=(15,10))
wscores = zip(tvec.get_feature_names(), chi2score)
wchi2 = sorted(wscores, key=lambda x:x[1])
topchi2 = list(zip(*wchi2[-20:]))
x = range(len(topchi2[1]))
labels = topchi2[0]
plt.barh(x,topchi2[1], align='center', alpha=0.2)
plt.plot(topchi2[1], x, '-o', markersize=5, alpha=0.8)
plt.yticks(x, labels)
plt.xlabel('$\chi^2$')
plt.show()



''' LDA TOPIC MODELLING '''

corpus = scraped_content_data.tolist()
lda2vec = malaya.lda2vec_topic_modelling(corpus, 10)
lda2vec.print_topics(5)
lda2vec.get_sentences(5)
lda2vec.visualize_topics(notebook_mode = True)



''' WRITING DATA TO CSV '''

with open('data500_csv.csv', mode='w', encoding='utf-8') as file:
    file_writer = csv.writer(file, delimiter=',', lineterminator = '\n')

    for i in range(len(scraped_content_data)):
        file_writer.writerow([scraped_content_data['fake_news'][i], 'Fake'])
        file_writer.writerow([scraped_content_data['real_news'][i], 'Real'])



