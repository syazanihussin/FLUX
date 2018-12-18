from scrapy.spiders import CrawlSpider, Rule
from scrapy.linkextractors import LinkExtractor
import mysql.connector

class CrawlerSpider(CrawlSpider):

    name = 'crawler_sebenarnya'
    allowed_domains = ['sebenarnya.my']
    start_urls = ['https://sebenarnya.my/']

    rules = (
        Rule(LinkExtractor(), callback='parse_item', follow=True),
    )

    def parse_item(self, response):

            try:
                # extract data from webpage
                fake_data = response.css('header h1 ::text').extract_first()

                real_data = response.xpath("//div[@class='td-post-content td-pb-padding-side']/p/text() | //div[@class='td-post-content td-pb-padding-side']/h4/strong/text() | //div[@class='rtejustify']/text() | //div[@class='td-post-content td-pb-padding-side']/div[@class='articleLead leadContainer']/p/text() | //div[@class='td-post-content td-pb-padding-side']/div[@class='articleLead leadContainer']/h4/strong/text()").extract()
                real_data = ' '.join(real_data)
                real_data = real_data.replace('\xa0', '')

                #display result
                yield {'url': response.url, 'fake_data': fake_data, 'real_data': real_data}

                # save news inside database
                mydb = mysql.connector.connect(host='localhost', database='news_dataset', user='root', password='')
                mycursor = mydb.cursor()

                sql = "INSERT INTO news_table2 (url, fake_news, real_news) VALUES (%s, %s, %s)"
                val = (response.url, fake_data, real_data)

                mycursor.execute(sql, val)
                mydb.commit()

                print(mycursor.lastrowid, " inserted.")

            except:
                yield {'message': 'sorry error'}