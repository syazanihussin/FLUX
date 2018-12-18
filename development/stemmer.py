import re
from development.tatabahasa import permulaan, hujung

class Stemmer():
    def naive_stemmer(word):
        assert (isinstance(word, str)), "input must be a string"
        try:
            word = re.findall(r'^(.*?)(%s)$'%('|'.join(hujung)), word)[0][0]
            mula = re.findall(r'^(.*?)(%s)'%('|'.join(permulaan[::-1])), word)[0][1]
            return word.replace(mula,'')
        except:
            return word
