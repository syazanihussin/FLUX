import pickle


news = ['najib diikat jamin', 'kfc tidak halal', 'vietnam menang dalam piala aff suzuki 2018', 'malaysia kalah dalam piala aff suzuki 2018', 'adib meninggal dunia', 'tun dr mahathir mati', 'najib mati', 'himpunan icerd tidak wujud', 'tun dr mahathir perdana menteri ke-7', 'anwar bakal ganti mahathir', 'cadburry diperbuat daripada babi', 'sushiking haram dimakan', 'sri serdang banjir tahun ini', 'nurul izzah letak jawatan naib presiden pkr', 'johor nafi ada ancaman tsunami bulan ini', 'viral "muhyiddin letak jawatan" adalah palsu', 'dewan rakyat lulus RUU mansuh akta anti berita tidak benar', 'polis nafi serbu apartmen mewah', 'pdrm beri jaminan proses pilihan raya berjalan lancar', 'malaysia juara piala aff suzuki 2010', 'pakatan harapan menang pru 14', 'bn tewas pru 14', 'pakatan harapan tewas pru 14', 'bn menang pru 14']
labels = ['Real', 'Fake', 'Real', 'Real', 'Real', 'Fake', 'Fake', 'Fake', 'Real', 'Real', 'Fake', 'Fake', 'Fake', 'Real', 'Real', 'Real', 'Real', 'Real', 'Real', 'Real', 'Real', 'Real', 'Fake', 'Fake']


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


