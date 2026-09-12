# pipelines.py

from itemadapter import ItemAdapter
import sqlite3
from datetime import datetime


class SrealityPipeline:
    def process_item(self, item, spider):
        return item


class SQLitePipeline:
    def open_spider(self, spider):
        self.conn = sqlite3.connect("apartments.db")
        self.cur = self.conn.cursor()
        # note the new price_per_m2 column
        self.cur.execute(
            """
        CREATE TABLE IF NOT EXISTS apartments (
            id            INTEGER PRIMARY KEY,
            listing_id    TEXT    UNIQUE,
            url           TEXT,
            title         TEXT,
            price_num     INTEGER,
            location      TEXT,
            rooms         REAL,
            area_m2       REAL,
            image_url     TEXT,
            price_per_m2  REAL,
            scraped_at    TEXT
        )
        """
        )

    def close_spider(self, spider):
        self.conn.commit()
        self.cur.close()
        self.conn.close()

    def process_item(self, item, spider):
        # compute price_per_m2 if possible
        price = item.get("price_num")
        area = item.get("area_m2")
        if price is not None and area:
            item["price_per_m2"] = float(price) / area
        else:
            item["price_per_m2"] = None

        # timestamp
        item["scraped_at"] = datetime.utcnow().isoformat()

        # insert into SQLite, now including price_per_m2
        self.cur.execute(
            """
          INSERT OR REPLACE INTO apartments
            (listing_id, url, title,
             price_num, location, rooms,
             area_m2, image_url, price_per_m2,
             scraped_at)
          VALUES (?,?,?,?,?,?,?,?,?,?)
        """,
            (
                item["listing_id"],
                item.get("url"),
                item.get("title"),
                item.get("price_num"),
                item.get("location"),
                item.get("rooms"),
                item.get("area_m2"),
                item.get("image_url"),
                item.get("price_per_m2"),
                item.get("scraped_at"),
            ),
        )
        return item
