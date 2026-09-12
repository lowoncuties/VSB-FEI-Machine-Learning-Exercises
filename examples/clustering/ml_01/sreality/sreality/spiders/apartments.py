# sreality/sreality/spiders/apartments.py

import json
import time
import re
import scrapy
from sreality.items import ApartmentItem


class ApartmentsSpider(scrapy.Spider):
    name = "apartments"
    allowed_domains = ["www.sreality.cz"]

    def __init__(self, page=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        tms = int(time.time() * 1000)
        self.start_urls = [
            (
                "https://www.sreality.cz/api/cs/v2/estates"
                f"?category_main_cb=1"
                f"&category_type_cb=1"
                f"&per_page=50"
                f"&page={int(page)}"
                f"&tms={tms}"
            )
        ]

    def parse(self, response):
        data = json.loads(response.text)
        estates = data.get("_embedded", {}).get("estates", [])
        for e in estates:
            item = ApartmentItem()

            lid = str(e.get("hash_id"))
            item["listing_id"] = lid

            # Parse title & fallback
            title = e.get("name") or e.get("locality") or ""
            item["title"] = title

            # Build front-end detail URL:
            # rooms_slug = e.g. "3+1" or "2+kk"
            m_slug = re.search(r"bytu\s+([\d\+kkmxMEONet]+)", title)
            rooms_slug = m_slug.group(1) if m_slug else ""
            # slugify locality
            loc = e.get("locality", "") or ""
            loc_slug = loc.lower().replace(" ", "-")
            item["url"] = (
                f"https://www.sreality.cz/detail/"
                f"prodej/byt/{rooms_slug}/{loc_slug}/{lid}"
            )

            # Price
            item["price_num"] = e.get("price_czk", {}).get("value_raw")

            # Extract rooms (float) from title e.g. "3+1"
            m_rooms = re.search(r"bytu\s+([\d\.]+)", title)
            item["rooms"] = float(m_rooms.group(1)) if m_rooms else None

            # Extract area in m2: number before "m²"
            m_area = re.search(r"(\d+)\s?m", title)
            item["area_m2"] = float(m_area.group(1)) if m_area else None

            # Location
            item["location"] = loc

            # First image
            imgs = e.get("_links", {}).get("images") or []
            item["image_url"] = imgs[0].get("href") if imgs else None

            yield item

        # Automatic pagination through the v2 API
        paging = data.get("paging", {})
        current = paging.get("page", 1)
        total = paging.get("pagesCount", 1)
        if current < total:
            nxt = current + 1
            tms = int(time.time() * 1000)
            base = response.url.split("?")[0]
            qs = (
                "category_main_cb=1"
                "&category_type_cb=1"
                f"&per_page=50&page={nxt}&tms={tms}"
            )
            yield scrapy.Request(f"{base}?{qs}", callback=self.parse)
