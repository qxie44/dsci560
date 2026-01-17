#Name: Hui Xie
#USD ID : 7956658480

import requests
import time
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

options = Options()
options.add_argument("--headless")
options.add_argument("--disable-gpu")
options.add_argument("--no-sandbox")

driver = webdriver.Chrome(
    service=Service(ChromeDriverManager().install()),
    options=options
)


url = "https://www.cnbc.com/world/?region=world"
driver.get(url)

print("Loading link...")
time.sleep(5)

html = driver.page_source

print("Saving html file...")

with open("web_data.html", "w", encoding="utf-8") as f:
    f.write(html)

driver.quit()
soup = BeautifulSoup(html, "html.parser")