from interface import implements, Interface
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
import pickle


class IInputProcessing(Interface):

    def preprocess_news(self, news):
        pass

    def vectorize_news(self, type, news, length):
        pass



class InputProcessing(implements(IInputProcessing)):

    def preprocess_news(self, news):
        return text_to_word_sequence(news, filters='!"#$%&()*+,-./:;<=>?@[\]“”^_`{|}~‘’''\t\n\xa0â\x80\x9c\x9d\'')


    def vectorize_news(self, type, news, length):

        # load vectorizer
        vocab = self.load_vocab(type)

        # pad and truncate news
        padded_texts = pad_sequences(vocab.texts_to_sequences([news]), maxlen=length, padding='post', truncating='post', value=0)

        return padded_texts


    def load_vocab(self, type):

        if(type == 'content'):
            with open('./model/content_word2vec_tokenizer.pickle', 'rb') as handle:
                vocab = pickle.load(handle)

        elif(type == 'stance'):
            with open('./model/stance_word2vec_tokenizer.pickle', 'rb') as handle:
                vocab = pickle.load(handle)

        return vocab
