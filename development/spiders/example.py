import scrapy

class CrawlerSpider(scrapy.Spider):
    name = 'simple_scraper'
    allowed_domains = ['sebenarnya.my']
    start_urls = ['https://sebenarnya.my']

    def parse(self, response):

        #with open(stripped_filename, 'wb') as f:
            #f.write(response.body)

        fake_data = response.css('header h1 ::text').extract_first()
        real_data = response.xpath('//div[@class="td-post-content td-pb-padding-side"]/p/text()').extract()

        yield {'url': response.url, 'fake_data': fake_data, 'real_data': real_data}





