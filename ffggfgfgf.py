import mysql.connector, pandas, csv
from development.DataPreparation import DataPreparation
import malaya


def word_count(str):
    counts = dict()
    words = str.split()

    for word in words:
        if word in counts:
            counts[word] += 1
        else:
            counts[word] = 1

    return counts


mydb = mysql.connector.connect(host='localhost', database='news_dataset', user='root', password='')
data_db = pandas.read_sql('SELECT penyataan, sumber FROM stance_sebenarnya LIMIT 500', con=mydb)
data_db2 = pandas.read_sql('SELECT penyataan, sumber FROM stance_utusan LIMIT 500', con=mydb)

data = pandas.concat([data_db['penyataan'], data_db2['penyataan']]).sort_index(kind='merge')
data = data.reset_index(drop=True)
dataDF = pandas.DataFrame()
dataDF['Penyataan'] = data

data = pandas.concat([data_db['sumber'], data_db2['sumber']]).sort_index(kind='merge')
data = data.reset_index(drop=True)
dataDF['Sumber'] = data

print(dataDF)
print(len(dataDF))

data_preparation = DataPreparation()
data_preparation.clean_data(dataDF['Penyataan'], loop=len(dataDF))
data_preparation.clean_data(dataDF['Sumber'], loop=len(dataDF))

lengths, lengths2 = [], []
#str_news = ''

for data in dataDF['Penyataan']:
    lengths.append(len(data.split()))
    #str_news += data + ' '

for data in dataDF['Sumber']:
    lengths2.append(len(data.split()))

avg_fake, avg_real, total_avg = 0, 0, 0

for i in range(len(dataDF)):
    avg_fake += lengths[i]
    avg_real += lengths2[i]

    '''if i%2==0:
        avg_fake += lengths[i]
    elif i%2!=0:
        avg_real += lengths[i]
    '''

    total_avg += lengths[i] + lengths2[i]

print(avg_fake, avg_real, total_avg)
print('avg fake news: ', avg_fake/(len(dataDF)/2))
print('avg real news: ', avg_real/(len(dataDF)/2))
print('avg total: ', total_avg/len(dataDF))

'''
countss = word_count(str_news)
print(countss)
print(len(countss))
print(str_news.count('awani'), countss.get('awani'))

for i in range(len(dataDF['News'])):
    dataDF['News'].loc[i] = malaya.sastrawi_stemmer(dataDF['News'].loc[i])

print(dataDF)

lengths = []
str_news = ''

for data in dataDF['News']:
    lengths.append(len(data.split()))
    str_news += data + ' '

avg_fake, avg_real, total_avg = 0, 0, 0

for i in range(len(dataDF)):
    if i%2==0:
        avg_fake += lengths[i]
    elif i%2!=0:
        avg_real += lengths[i]

    total_avg += lengths[i]


print(avg_fake, avg_real, total_avg)
print('avg fake news: ', avg_fake/(len(dataDF)/2))
print('avg real news: ', avg_real/(len(dataDF)/2))
print('avg total: ', total_avg/len(dataDF))

countss = word_count(str_news)
print(countss)
print(len(countss))
print(str_news.count('awani'), countss.get('awani'))
'''


'''
corpus = dataDF['News'].tolist()
lda2vec = malaya.lda2vec_topic_modelling(corpus, 10)
lda2vec.print_topics(5)
lda2vec.get_sentences(5)
lda2vec.visualize_topics(notebook_mode = True)

with open('data500_csv.csv', mode='w', encoding='utf-8') as file:
    file_writer = csv.writer(file, delimiter=',', lineterminator = '\n')

    for i in range(len(data_db)):
        file_writer.writerow([data_db['fake_news'][i], 'Fake'])
        file_writer.writerow([data_db['real_news'][i], 'Real'])
        
'''