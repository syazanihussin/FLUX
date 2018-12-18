from scrapy.spiders import CrawlSpider, Rule
from scrapy.linkextractors import LinkExtractor
import mysql.connector


class CrawlerSpider(CrawlSpider):
    name = 'crawler_awani'
    allowed_domains = ['www.astroawani.com']
    start_urls = ['http://www.astroawani.com/']

    rules = (
        Rule(LinkExtractor(), callback='parse_item', follow=True),
    )

    def parse_item(self, response):

            # extract category
            title_category2 = response.xpath('//a[@class="a2"]/text()').extract()
            title_category = ''.join(title_category2)

            # extract data from webpage
            real_data1 = response.xpath('//h1[@class="col-xs-12"]/text()').extract() + response.xpath('//div[@class="detail-body-content"]/text()').extract()
            real_data = ''.join(real_data1)
            real_data = real_data.replace('\n', '')
            fake_data = real_data


            #display result
            #yield {'url': response.url,  'fake_data': fake_data, 'real_data': real_data, 'category': title_category}


            # save news inside database
            if(real_data != ''):
                mydb = mysql.connector.connect(host='localhost', database='news_dataset', user='root', password='')
                mycursor = mydb.cursor()

                sql = "INSERT INTO news_table (url, fake_news, real_news, category) VALUES (%s, %s, %s, %s)"
                val = (response.url, fake_data, real_data, title_category)

                mycursor.execute(sql, val)
                mydb.commit()

                print(mycursor.lastrowid, " inserted.")