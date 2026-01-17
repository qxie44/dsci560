#Name: Hui Xie
#USD ID : 7956658480

import csv
from bs4 import BeautifulSoup

print("Reading Raw Data...")
with open( "../data/raw_data/web_data.html", "r", encoding= "utf-8") as file:
    content = file.read()

parsed = BeautifulSoup(content, "html.parser")

market_data = []
print("Filtering for Market Data...")
marketBan = parsed.find("div", class_="MarketsBanner-marketData")
marketC = marketBan.find_all("a", class_="MarketCard-container")

print("Storing Market Data...")
for c in marketC:
    spans = c.find_all("span")

    symbol = spans[0].text.strip() if len(spans) > 0 else ""
    position = spans[1].text.strip() if len(spans) > 1 else ""

    change_div = c.find("div", class_="MarketCard-changeData")
    change = change_div.get_text(strip=True) if change_div else ""

    market_data.append([symbol, position, change])

    
data_path = "../data/processed_data/market_data.csv"
#data_path = "market_data.csv"

print("Writing to market csv...")
with open(data_path, "w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["marketCard_symbol", "marketCard_stockPosition", "marketCard_changePct"])
    writer.writerows(market_data)
    
news_data= []
print("Filtering for News Data...")
news = parsed.find("ul", class_="LatestNews-list")
newsC = news.find_all("li", class_="LatestNews-item")

print("Storing News Data...")
for i in newsC:
    headline = i.find("a", class_="LatestNews-headline")
    timestamp = i.find("time")

    title = headline.text.strip() if headline else ""
    link = headline.get("href", "") if headline else ""

    news_data.append([
        timestamp.text.strip() if timestamp else "",
        title,
        link
    ])

news_path = "../data/processed_data/news_data.csv"
#news_path = "news_data.csv"

print("Writing to News csv...")
with open(news_path, "w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["LatestNews_timestamp", "title", "link"])
    writer.writerows(news_data)
