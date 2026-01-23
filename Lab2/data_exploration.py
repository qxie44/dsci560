import requests
import pandas as pd
from bs4 import BeautifulSoup
import pdfplumber
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import os

#topic- books

#csv data
def get_csv_data():
    print("\n Loading library api...")

    url = "https://openlibrary.org/search.json?q=classic+literature&limit=20"
    response = requests.get(url)
    data = response.json()
    books = []
    for doc in data["docs"]:
        books.append({
            "title": doc.get("title"),
            "author": ", ".join(doc.get("author_name", [])),
            "first_publish_year": doc.get("first_publish_year"),
            "edition_count": doc.get("edition_count")
        })

    df = pd.DataFrame(books)
    print("Previewing dataset")
    print(df.head(5))
    print("Number of rows:", df.shape[0])
    print("Number of columns:", df.shape[1])
    print("Missing data: ", df.isnull().sum())
    print("Columns", df.columns)

    print("Dataset shape:", df.shape)
    df.to_csv("books_metadata.csv", index=False)
    print("Saved books_metadata.csv...")

#html
def get_html_data():
    print("\n Loading Gutenberg...")

    url = "https://www.gutenberg.org/files/1342/1342-h/1342-h.htm"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    text = soup.get_text(separator=" ", strip=True)

    with open("pride_and_prejudice_htmldoc.txt", "w", encoding="utf-8") as f:
        f.write(text)
    print("Saved html as txt...")

#pdf
def get_pdf_data():
    print("\n Opening scanned book...")

    pdf_file = "scanned_pride_and_prejudice.pdf"
    pdf_url = "https://dn790002.ca.archive.org/0/items/prideprejudice00aust/prideprejudice00aust.pdf"

    response = requests.get(pdf_url)
    with open(pdf_file, "wb") as f:
        f.write(response.content)

    # Convert PDF pages to images
    images = convert_from_path(pdf_file, first_page=11, last_page=15)

    ocr_text = ""

    for i, image in enumerate(images):
        text = pytesseract.image_to_string(image)
        ocr_text += f"\n--- Page {i+1} ---\n{text}"

    with open("extracted_pride_and_prejudice_ocrpdf.txt", "w", encoding="utf-8") as f:
        f.write(ocr_text)
    print("Txt saved...")

#run all
if __name__ == "__main__":
    get_csv_data()
    get_html_data()
    get_pdf_data()