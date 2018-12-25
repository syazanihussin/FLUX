from interface import implements, Interface
from googleapiclient.discovery import build


class ISearching(Interface):

    def search_news(self, keyword):
        pass



class Searching(implements(ISearching)):

    def search_news(self, keyword):

        # build custom search service
        service = build(serviceName="customsearch", version="v1", developerKey="AIzaSyAgtXBw-jGS7D7QoMG75KLZ3YiJhIDtnCs")

        # get response based on the keyword passed to the service
        response = service.cse().list(q=keyword, cx='016555104812731516391:lz6xrzmoeuc').execute()

        return response['items']


#  key:  AIzaSyAe1k87i3K7pPpFG8a_fCE_fJ6Ofukm6LI  ###  AIzaSyAhmbPn_vz7RVQE-XauMuWESFHJ0a9908k  ###  AIzaSyAgtXBw-jGS7D7QoMG75KLZ3YiJhIDtnCs
#  cx: 016555104812731516391:folkceyikgq (general search)  ###  016555104812731516391:lz6xrzmoeuc  ###  016555104812731516391:bzh4rgo03ke