from system import search, detection, inputprocessing
from keras import backend as K
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import csv, pickle


news = ['najib diikat jamin', 'kfc tidak halal', 'vietnam menang dalam piala aff suzuki 2018', 'malaysia kalah dalam piala aff suzuki 2018', 'adib meninggal dunia', 'tun dr mahathir mati', 'najib mati', 'himpunan icerd tidak wujud', 'tun dr mahathir perdana menteri ke-7', 'anwar bakal ganti mahathir', 'cadburry diperbuat daripada babi', 'sushiking haram dimakan', 'sri serdang banjir tahun ini', 'nurul izzah letak jawatan naib presiden pkr', 'johor nafi ada ancaman tsunami bulan ini', 'viral "muhyiddin letak jawatan" adalah palsu', 'dewan rakyat lulus RUU mansuh akta anti berita tidak benar', 'polis nafi serbu apartmen mewah', 'pdrm beri jaminan proses pilihan raya berjalan lancar', 'malaysia juara piala aff suzuki 2010', 'pakatan harapan menang pru 14', 'bn tewas pru 14', 'pakatan harapan tewas pru 14', 'bn menang pru 14']
labels = ['Real', 'Fake', 'Real', 'Real', 'Real', 'Fake', 'Fake', 'Fake', 'Real', 'Real', 'Fake', 'Fake', 'Fake', 'Real', 'Real', 'Real', 'Real', 'Real', 'Real', 'Real', 'Real', 'Real', 'Fake', 'Fake']

searching_obj = search.Searching()
title_searched_results_all = []

for news_data in news:
    searched_results = searching_obj.search_news(keyword=news_data)

    title_searched_results = []

    for searched_data in searched_results:
        title_searched_results.append(searched_data['snippet'])

    title_searched_results_all.append(title_searched_results)


def detect_using_stance_consine_similarity(news, searched_results):

    title_searched_results = []
    title_searched_results.append(news)

    for data in searched_results:
        title_searched_results.append(data)

    tfidf_vectorizer = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}')
    tfidf_matrix_train = tfidf_vectorizer.fit_transform(title_searched_results)

    cs = cosine_similarity(tfidf_matrix_train[0:1], tfidf_matrix_train)

    real_count, fake_count = 0, 0

    for i in range(len(title_searched_results)):
        print(title_searched_results[i])
        print(cs[0][i])

        if cs[0][i] < 0.3:
            fake_count += 1
        elif cs[0][i] > 0.3:
            real_count += 1

    print('Real: ', real_count, 'Fake: ', fake_count)

    if real_count >= 3:
        label = 'Real'

    elif fake_count > real_count:
        label = 'Fake'

    return label, real_count, fake_count, cs[0]



def detect_using_stance_model(news, searched_results):

    inputprocessing_obj = inputprocessing.InputProcessing()
    clean_penyataan = inputprocessing_obj.preprocess_news(news)
    padded_penyataan = inputprocessing_obj.vectorize_news('stance', clean_penyataan, 100)
    print(clean_penyataan)

    real_count, fake_count, sum_real_prob, sum_fake_prob = 0, 0, 0, 0
    label_list, fake_probability_list, real_probability_list = [], [], []

    for data in searched_results:

        clean_sumber = inputprocessing_obj.preprocess_news(data)
        padded_sumber = inputprocessing_obj.vectorize_news('stance', clean_sumber, 2000)
        print(clean_sumber)

        detection_obj = detection.Detection()
        label, fake_probability, real_probability = detection_obj.detect_fake_news('stance', [padded_penyataan, padded_sumber])

        print(label, fake_probability, real_probability)

        label_list.append(label)
        fake_probability_list.append(fake_probability)
        real_probability_list.append(real_probability)

        sum_fake_prob += fake_probability
        sum_real_prob += real_probability

        if label == 'Fake':
            fake_count += 1
        elif label == 'Real':
            real_count += 1

        K.clear_session()

    if fake_count > 2 & fake_count < 6:
        considered_label = 'Fake'

    elif real_count > fake_count:
        considered_label = 'Real'

    return considered_label, real_count, fake_count, sum_real_prob, sum_fake_prob, label_list, real_probability_list, fake_probability_list



correct_prediction_cosine_similarity, correct_prediction_cosine_similarity_all = 0, []
correct_prediction_model, correct_prediction_model_all = 0, []

label_all, real_count_all, fake_count_all, cosine_similar_all = [], [], [], []
considered_label_all, real_count_list, fake_count_list, sum_real_prob_list, sum_fake_prob_list = [], [], [], [], []
label_list_all, real_probability_list_all, fake_probability_list_all = [], [], []

for i in range(len(news)):

    label, real_count, fake_count, cosine_similar = detect_using_stance_consine_similarity(news[i], title_searched_results_all[i])

    label_all.append(label) #24
    real_count_all.append(real_count) #24
    fake_count_all.append(fake_count) #24
    cosine_similar_all.append(cosine_similar[1:11]) # 24, inside 10

    if(label == labels[i]):
        correct_prediction_cosine_similarity += 1

    correct_prediction_cosine_similarity_all.append(correct_prediction_cosine_similarity)


    considered_label, real_count, fake_count, sum_real_prob, sum_fake_prob, label_list, real_probability_list, fake_probability_list = detect_using_stance_model(news[i], title_searched_results_all[i])

    considered_label_all.append(considered_label) #24
    real_count_list.append(real_count) #24
    fake_count_list.append(fake_count) #24
    sum_real_prob_list.append(sum_real_prob) #24
    sum_fake_prob_list.append(sum_fake_prob) #24
    label_list_all.append(label_list) # 24, inside 10
    real_probability_list_all.append(real_probability_list) # 24, inside 10
    fake_probability_list_all.append(fake_probability_list) # 24, inside 10

    if(considered_label == labels[i]):
        correct_prediction_model += 1

    correct_prediction_model_all.append(correct_prediction_model) #24



pickle_out = open("title_searched_results_all.pickle","wb")
pickle.dump(title_searched_results_all, pickle_out)

pickle_out2 = open("label_all.pickle","wb")
pickle.dump(label_all, pickle_out2)

pickle_out3 = open("real_count_all.pickle","wb")
pickle.dump(real_count_all, pickle_out3)

pickle_out4 = open("fake_count_all.pickle","wb")
pickle.dump(fake_count_all, pickle_out4)

pickle_out5 = open("cosine_similar_all.pickle","wb")
pickle.dump(cosine_similar_all, pickle_out5)

pickle_out6 = open("correct_prediction_cosine_similarity_all.pickle","wb")
pickle.dump(correct_prediction_cosine_similarity_all, pickle_out6)

pickle_out7 = open("considered_label_all.pickle","wb")
pickle.dump(considered_label_all, pickle_out7)

pickle_out8 = open("real_count_list.pickle","wb")
pickle.dump(real_count_list, pickle_out8)

pickle_out15 = open("fake_count_list.pickle","wb")
pickle.dump(fake_count_list, pickle_out15)

pickle_out9 = open("sum_real_prob_list.pickle","wb")
pickle.dump(sum_real_prob_list, pickle_out9)

pickle_out10 = open("sum_fake_prob_list.pickle","wb")
pickle.dump(sum_fake_prob_list, pickle_out10)

pickle_out11 = open("label_list_all.pickle","wb")
pickle.dump(label_list_all, pickle_out11)

pickle_out12 = open("real_probability_list_all.pickle","wb")
pickle.dump(real_probability_list_all, pickle_out12)

pickle_out13 = open("fake_probability_list_all.pickle","wb")
pickle.dump(fake_probability_list_all, pickle_out13)

pickle_out14 = open("correct_prediction_model_all.pickle","wb")
pickle.dump(correct_prediction_model_all, pickle_out14)


with open('result.csv', mode='w') as result_file:
    result_writer = csv.writer(result_file, delimiter=',')

    for i in range(len(news)):
        for j in range(len(title_searched_results_all[i])):
            csv_data = []
            
            if j == 0:
                csv_data.append(news[i])
                csv_data.append(labels[i])
                csv_data.append(title_searched_results_all[i][j])
                csv_data.append(label_all[i])
                csv_data.append(real_count_all[i])
                csv_data.append(fake_count_all[i])
                csv_data.append(cosine_similar_all[i][j])
                csv_data.append(correct_prediction_cosine_similarity_all[i])
                csv_data.append(correct_prediction_cosine_similarity_all[i]/len(news))

                csv_data.append(considered_label_all[i])
                csv_data.append(real_count_list[i])
                csv_data.append(fake_count_list[i])
                csv_data.append(sum_real_prob_list[i])
                csv_data.append(sum_fake_prob_list[i])
                csv_data.append(label_list_all[i][j])
                csv_data.append(real_probability_list_all[i][j])
                csv_data.append(fake_probability_list_all[i][j])
                csv_data.append(correct_prediction_model_all[i])
                csv_data.append(correct_prediction_model_all[i]/len(news))

            else:
                csv_data.append('')
                csv_data.append('')
                csv_data.append(title_searched_results_all[i][j])
                csv_data.append('')
                csv_data.append('')
                csv_data.append('')
                csv_data.append(cosine_similar_all[i][j])
                csv_data.append('')
                csv_data.append('')

                csv_data.append('')
                csv_data.append('')
                csv_data.append('')
                csv_data.append('')
                csv_data.append('')
                csv_data.append(label_list_all[i][j])
                csv_data.append(real_probability_list_all[i][j])
                csv_data.append(fake_probability_list_all[i][j])
                csv_data.append('')
                csv_data.append('')

            result_writer.writerow(csv_data)


# load pickle saved data to try different logic
pickle_out = open("title_searched_results_all.pickle","rb")
title_searched_results_all = pickle.load(pickle_out)

pickle_out2 = open("label_all.pickle","rb")
label_all = pickle.load(pickle_out2)

pickle_out3 = open("real_count_all.pickle","rb")
real_count_all = pickle.load(pickle_out3)

pickle_out4 = open("fake_count_all.pickle","rb")
fake_count_all = pickle.load(pickle_out4)

pickle_out5 = open("cosine_similar_all.pickle","rb")
cosine_similar_all = pickle.load(pickle_out5)

pickle_out6 = open("correct_prediction_cosine_similarity_all.pickle","rb")
correct_prediction_cosine_similarity_all = pickle.load(pickle_out6)

pickle_out7 = open("considered_label_all.pickle","rb")
considered_label_all = pickle.load(pickle_out7)

pickle_out8 = open("real_count_list.pickle","rb")
real_count_list = pickle.load(pickle_out8)

pickle_out15 = open("fake_count_list.pickle","rb")
fake_count_list = pickle.load(pickle_out15)

pickle_out9 = open("sum_real_prob_list.pickle","rb")
sum_real_prob_list = pickle.load(pickle_out9)

pickle_out10 = open("sum_fake_prob_list.pickle","rb")
sum_fake_prob_list = pickle.load(pickle_out10)

pickle_out11 = open("label_list_all.pickle","rb")
label_list_all = pickle.load(pickle_out11)

pickle_out12 = open("real_probability_list_all.pickle","rb")
real_probability_list_all = pickle.load(pickle_out12)

pickle_out13 = open("fake_probability_list_all.pickle","rb")
fake_probability_list_all = pickle.load(pickle_out13)

pickle_out14 = open("correct_prediction_model_all.pickle","rb")
correct_prediction_model_all = pickle.load(pickle_out14)


# LOGIC 1
label_correct = []

for i in range(len(news)):
    for j in range(len(title_searched_results_all[i])):

        if cosine_similar_all[i][j] > 0.2:
            maximum = max(cosine_similar_all[i])
            if cosine_similar_all[i][j] == maximum:
                label = label_list_all[i][j]
                label_correct.append({i:label})

count_correct = 0
for i in range(len(news)):
    if label_correct[i] == labels[i]:
        count_correct += 1

print(count_correct/len(news))


# LOGIC 2
label_correct = []
label = ''

for i in range(len(news)):
    if real_count_all[i] >= 3:
        label = 'Real'

    elif fake_count_all[i] > real_count_all[i]:
        label = 'Fake'

    if fake_count_all[i] > 7:
        if label == 'Real':
            label = 'Fake'
        elif label == 'Fake':
            label = 'Real'

    label_correct.append(label)


