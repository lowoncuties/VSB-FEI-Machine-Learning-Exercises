# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy


class SrealityItem(scrapy.Item):
    # define the fields for your item here like:
    # name = scrapy.Field()
    pass


class ApartmentItem(scrapy.Item):
    listing_id = scrapy.Field()
    url = scrapy.Field()
    title = scrapy.Field()
    price_num = scrapy.Field()
    location = scrapy.Field()
    rooms = scrapy.Field()
    area_m2 = scrapy.Field()
    image_url = scrapy.Field()
    scraped_at = scrapy.Field()
    price_per_m2 = scrapy.Field()
