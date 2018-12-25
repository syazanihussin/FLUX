from interface import implements, Interface
from keras.models import load_model


class IDetection(Interface):

    def detect_fake_news(self, type, news):
        pass



class Detection(implements(IDetection)):

    def detect_fake_news(self, type, news):

        # load detection model
        detection_model = self.load_detection_model(type)

        # predict probability
        probabilities = detection_model.predict(news)

        # get probability according to its assosiated class
        class_label, fake_prob, real_prob = self.get_class_label(probabilities)

        return class_label, fake_prob, real_prob


    def load_detection_model(self, type):
        if(type == 'content'):
            return load_model('./model/content_model.h5')
        elif(type == 'stance'):
            return load_model('./model/stance_model.h5')


    def get_class_label(self, probabilities):

        for probability in probabilities:
            fake_prob = probability[0]
            real_prob = probability[1]

            if(fake_prob > real_prob):
                class_label = 'Fake'
            elif(real_prob > fake_prob):
                class_label = 'Real'

        return class_label, fake_prob, real_prob
