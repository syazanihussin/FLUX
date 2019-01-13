from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from gensim.models import Word2Vec, FastText
import pickle, numpy as np, malaya



class FeatureEngineering:


    def __init__(self):
        pass


    def prepare_count_vectorizer(self, data, train_x, test_x, initial_vocab_size):

        # create dictionary
        count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=initial_vocab_size)
        count_vect.fit(data)

        # save tokenizer for vocabulary indexing
        self.save_tokenizer(count_vect, '..\model\count_vectorizer.pickle')

        # tokenize and vectorize data using count vectorizer
        train_count = count_vect.transform(train_x).toarray()
        test_count = count_vect.transform(test_x).toarray()

        return len(count_vect.get_feature_names()), train_count, test_count


    def prepare_word_level_tfidf(self, data, train_x, test_x, initial_vocab_size):

        # create dictionary
        tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=initial_vocab_size)
        tfidf_vect.fit(data)

        # save tokenizer for vocabulary indexing
        self.save_tokenizer(tfidf_vect, '..\model\word_level_tfidf_vectorizer.pickle')

        # tokenize and vectorize data using vectorizer
        train_tfidf = tfidf_vect.transform(train_x).toarray()
        test_tfidf = tfidf_vect.transform(test_x).toarray()

        return len(tfidf_vect.get_feature_names()), train_tfidf, test_tfidf


    def prepare_ngram_level_tfidf(self, data, train_x, test_x, initial_vocab_size):

        # create dictionary
        tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2, 3), max_features=initial_vocab_size)
        tfidf_vect.fit(data)

        # save tokenizer for vocabulary indexing
        self.save_tokenizer(tfidf_vect, '..\model\ngram_level_tfidf_vectorizer.pickle')

        # tokenize and vectorize data using ngram level tfidf vectorizer
        train_tfidf = tfidf_vect.transform(train_x).toarray()
        test_tfidf = tfidf_vect.transform(test_x).toarray()

        return len(tfidf_vect.get_feature_names()), train_tfidf, test_tfidf


    def prepare_character_level_tfidf(self, data, train_x, test_x, initial_vocab_size):

        # create dictionary
        tfidf_vect = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=(2, 3), max_features=initial_vocab_size)
        tfidf_vect.fit(data)

        # save tokenizer for vocabulary indexing
        self.save_tokenizer(tfidf_vect, '..\model\char_level_tfidf_vectorizer.pickle')

        # tokenize and vectorize data using char level tfidf vectorizer
        train_tfidf = tfidf_vect.transform(train_x).toarray()
        test_tfidf = tfidf_vect.transform(test_x).toarray()

        return len(tfidf_vect.get_feature_names()), train_tfidf, test_tfidf


    def prepare_embedding(self, mode, data, initial_vocab_size, embedding_size, embedding_type, embedding_dir, tokenizer_dir, embedding_matrix_dir):

        if(mode == 'create'):

            # load embedding model
            if(embedding_type=='word2vec'):
                model = Word2Vec.load(embedding_dir)
            elif(embedding_type=='fasttext'):
                model = FastText.load(embedding_dir)
            #elif (embedding_type == 'glove'):
                #model = Glove.load('glove.model')

            # create vocababulary
            tokenizer = Tokenizer(num_words=initial_vocab_size, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}‘’''~\t\n\'', lower=True)
            tokenizer.fit_on_texts(data)
            word_index = tokenizer.word_index

            # save tokenizer for vocabulary indexing
            self.save_tokenizer(tokenizer, tokenizer_dir)

            # create embedding matrix
            vocabulary_size = min(len(word_index) + 1, initial_vocab_size)
            embedding_matrix = np.zeros((vocabulary_size, embedding_size))

            for word, i in word_index.items():
                try:
                    embedding_vector = model.wv[word]
                    embedding_matrix[i] = embedding_vector
                except KeyError:
                    embedding_matrix[i] = np.random.normal(0, np.sqrt(0.25), embedding_size)

                '''elif(embedding_type == 'glove'):
                    try:
                        embedding_vector = model.word_vectors[model.dictionary[word]]
                        embedding_matrix[i] = embedding_vector
                    except KeyError:
                        embedding_matrix[i] = np.random.normal(0, np.sqrt(0.25), embedding_size)'''

            np.save(embedding_matrix_dir, embedding_matrix)

        if(mode == 'use'):
            with open(tokenizer_dir, 'rb') as handle:
                tokenizer = pickle.load(handle)

            vocabulary_size = len(tokenizer.word_index) + 1
            embedding_matrix = np.load(embedding_matrix_dir)

        return vocabulary_size, embedding_matrix, tokenizer


    def tokenize_vectorize(self, tokenizer, data, max_length):

        data_temp = []

        for sentence in data:
            data_temp.append(text_to_word_sequence(sentence))

        # pad and truncate data to make sure the data comes in fixed size
        data_word2vec = pad_sequences(tokenizer.texts_to_sequences(data_temp), maxlen=max_length, padding='post', truncating='post', value=0)

        return data_word2vec


    def prepare_embedding_malaya(self):
        embedded = malaya.malaya_word2vec(256)
        return len(embedded['dictionary']), embedded['nce_weights'], embedded['dictionary']


    def tokenize_vectorize_malaya(self, tokenizer, data, max_length):
        data_temp = []

        for sentence in data:
            list = []
            for word in sentence.split():
                id = tokenizer.get(word)
                if id is not None:
                    list.append(id)
            data_temp.append(list)

        # pad and truncate data to make sure the data comes in fixed size
        data_word2vec = pad_sequences(data_temp, maxlen=max_length, padding='post', truncating='post', value=0)

        return data_word2vec


    def save_tokenizer(self, tokenizer, file_name):
        with open(file_name, 'wb') as handle:
            pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)