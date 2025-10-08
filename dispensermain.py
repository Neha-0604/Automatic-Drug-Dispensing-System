import cv2
import pytesseract
import pandas as pd

# Path to tesseract executable (Windows users)
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load drug database
df = pd.read_csv('database.csv')

def preprocess_image(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Optional: Thresholding for better OCR
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    return thresh

def extract_text(img_path):
    img = preprocess_image(img_path)
    text = pytesseract.image_to_string(img)
    return text

def match_drug(text):
    text = text.lower()
    for index, row in df.iterrows():
        if row['drug_name'].lower() in text:
            return row
    return None

if __name__ == "__main__":
    img_path = input("Enter path to drug label image: ")
    extracted_text = extract_text(img_path)
    print("\nDetected Text:")
    print(extracted_text)

    drug = match_drug(extracted_text)
    if drug is not None:
        print("\nDrug Found!")
        print(f"Name: {drug['drug_name']}")
        print(f"Quantity: {drug['quantity']}")
        print(f"Schedule: {drug['schedule']}")
    else:
        print("\nDrug not found in database.")
