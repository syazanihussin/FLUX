import scrapy
import mysql.connector
from mysql.connector import Error


class CrawlerSpider(scrapy.Spider):
    name = 'crawler'
    allowed_domains = ['sebenarnya.my']
    start_urls = ['https://sebenarnya.my/kekacauan-kaum-di-kluang/']

    def parse(self, response):
        #try:
            #get category
            title_category = response.xpath('//div[@class="td-post-content td-pb-padding-side"]/h4[1]/strong/text()').extract()

            if(len(title_category) == 0):
                title_category = response.xpath('//div[@class="td-post-content td-pb-padding-side"]/p[1]/strong/text()').extract()

            #extraction logic
            if (title_category[0] == 'PENJELASAN:' or title_category[0] == 'PENJELASAN :'):
                fake_data = response.css('header h1 ::text').extract_first()
                real_data1 = response.xpath('//div[@class="td-post-content td-pb-padding-side"]/p/text()').extract() + response.xpath('//div[@class="rtejustify"]/text()').extract()
                real_data2 = ''
                for data in real_data1:
                    real_data2 = real_data2 + data + ' '
                real_data = real_data2.replace("\xa0", " ")

                if(real_data == ''):
                    real_data1 = response.xpath('//div[@class="td-post-content td-pb-padding-side"]/div[@class="articleLead leadContainer"]/p/text()').extract() + response.xpath('//div[@class="rtejustify"]/text()').extract()
                    real_data2 = ''
                    for data in real_data1:
                        real_data2 = real_data2 + data + ' '
                    real_data = real_data2.replace("\xa0", " ")

            elif (title_category[0] == 'PALSU:'):
                fake_data1 = response.css('header h1 ::text').extract_first()
                fake_data2 = response.xpath("//p[preceding-sibling::h4[strong[contains(text(), 'PALSU:')]]][1]/text()").extract()
                fake_data3 = '' + fake_data2[0]
                fake_data = fake_data1 + ' ' + fake_data3.replace("\xa0", " ")
                real_data1 = response.xpath("//p[preceding-sibling::h4[strong[contains(text(), 'SEBENARNYA:')]]]/text()").extract()
                real_data2 = ''
                for data in real_data1:
                    real_data2 = real_data2 + data + ' '
                real_data = real_data2.replace("\xa0", " ")

                if (real_data == ''):
                    real_data1 = response.xpath("//div[@class='articleBody'][preceding-sibling::h4[strong[contains(text(), 'SEBENARNYA:')]]]/p/text()").extract()
                    real_data2 = ''
                    for data in real_data1:
                        real_data2 = real_data2 + data + ' '
                    real_data = real_data2.replace("\xa0", " ")

                if (real_data == ''):
                    real_data1 = response.xpath("//p[preceding-sibling::h4[strong[contains(text(), 'SEBENARNYA :')]]]/text()").extract()
                    real_data2 = ''
                    for data in real_data1:
                        real_data2 = real_data2 + data + ' '
                    real_data = real_data2.replace("\xa0", " ")

                if (real_data == ''):
                    real_data1 = response.xpath("//div[@class='articleBody'][preceding-sibling::h4[strong[contains(text(), 'SEBENARNYA :')]]]/p/text()").extract()
                    real_data2 = ''
                    for data in real_data1:
                        real_data2 = real_data2 + data + ' '
                    real_data = real_data2.replace("\xa0", " ")

            elif (title_category[0] == 'PALSU :'):
                fake_data1 = response.css('header h1 ::text').extract_first()
                fake_data2 = response.xpath("//p[preceding-sibling::h4[strong[contains(text(), 'PALSU :')]]][1]/text()").extract()
                fake_data3 = '' + fake_data2[0]
                fake_data = fake_data1 + ' ' + fake_data3.replace("\xa0", " ")
                real_data1 = response.xpath("//p[preceding-sibling::h4[strong[contains(text(), 'SEBENARNYA :')]]]/text()").extract()
                real_data2 = ''
                for data in real_data1:
                    real_data2 = real_data2 + data + ' '
                real_data = real_data2.replace("\xa0", " ")

                if (real_data == ''):
                    real_data1 = response.xpath("//div[@class='articleBody'][preceding-sibling::h4[strong[contains(text(), 'SEBENARNYA :')]]]/p/text()").extract()
                    real_data2 = ''
                    for data in real_data1:
                        real_data2 = real_data2 + data + ' '
                    real_data = real_data2.replace("\xa0", " ")

                if (real_data == ''):
                    real_data1 = response.xpath("//p[preceding-sibling::h4[strong[contains(text(), 'SEBENARNYA:')]]]/text()").extract()
                    real_data2 = ''
                    for data in real_data1:
                        real_data2 = real_data2 + data + ' '
                    real_data = real_data2.replace("\xa0", " ")

                if (real_data == ''):
                    real_data1 = response.xpath("//div[@class='articleBody'][preceding-sibling::h4[strong[contains(text(), 'SEBENARNYA:')]]]/p/text()").extract()
                    real_data2 = ''
                    for data in real_data1:
                        real_data2 = real_data2 + data + ' '
                    real_data = real_data2.replace("\xa0", " ")

            elif (title_category[0] == 'WASPADA:' or title_category[0] == 'WASPADA :'):
                fake_data = response.css('header h1 ::text').extract_first()
                real_data1 = response.xpath('//div[@class="td-post-content td-pb-padding-side"]/p/text()').extract()
                real_data2 = ''
                for data in real_data1:
                    real_data2 = real_data2 + data + ' '
                real_data = real_data2.replace("\xa0", " ")

            #display result
            yield {'url': response.url,  'fake_data': fake_data, 'real_data': real_data, 'category': title_category[0]}

            try:
                mydb = mysql.connector.connect(host='localhost',
                                       database='news',
                                       user='root',
                                       password='spectrum',
                                       auth_plugin='mysql_native_password')

                if(mydb.is_connected()):
                    print('Connected to MySQL database')
                    yield {'note': 'Connected to MySQL database'}

                    mycursor = mydb.cursor()

                    sql = "INSERT INTO news_table (url, fake_news, real_news, category) VALUES (%s, %s, %s, %s)"
                    val = (response.url, fake_data, real_data, title_category[0])
                    mycursor.execute(sql, val)

                    mydb.commit()

                    print(mycursor.rowcount, "record inserted.")


            except Error as e:
                print(e)

        #except:
            #yield {'note': 'something wrong happened'}
