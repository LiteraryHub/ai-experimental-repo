"""
A script for extracting and processing Arabic text from PDF files, designed to support natural language processing (NLP) tasks involving Arabic literature.

The script uses PyMuPDF to render PDF pages as images, then applies OCR (Optical Character Recognition) using Tesseract configured for Arabic language to extract text. It preprocesses the extracted text by removing Arabic diacritics, punctuation, and non-Arabic characters, then divides the text into manageable chunks which are saved as JSON files.

Key Features:
- Processes PDF files to extract Arabic text using OCR.
- Preprocesses text by removing diacritics, punctuation, numbers, and Latin characters.
- Splits text into chunks, ensuring each chunk has a minimum number of tokens for consistency.
- Saves processed text chunks as JSON files, with metadata including book name and page number.

Functions:
    tokenizer(text): Splits a string into tokens based on whitespace.
    preprocess_arabic_text(text): Cleans Arabic text by removing diacritics, punctuation, and non-relevant characters.
    save_as_json(data, filename): Saves data in JSON format to a specified filename.
    process_pdf(pdf_path): Renders each page of a PDF as an image, extracts and preprocesses the text, then saves it in JSON format.

Usage:
    Set the `pytesseract.pytesseract.tesseract_cmd` variable to the path of the Tesseract executable on your system. The script is configured to process PDF files in a specified directory, extracting and preprocessing the text, and saving the processed data in a designated output directory. Each PDF page's text is saved in separate JSON files if the processed text chunk has at least 100 tokens after preprocessing.

Note:
- The script requires Tesseract-OCR configured for the Arabic language (`-l ara`) and PyMuPDF (fitz) for rendering PDF pages as images.
- It's designed for batch processing of PDF files located in a specific directory.
- Output JSON files are named according to the book name, page number, and chunk index to facilitate organization and subsequent processing.
"""
import os
import json
import io
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import re
import time

# Configure pytesseract to use the Arabic language and your Tesseract path
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
TESSERACT_CONFIG = '--oem 3 --psm 6 -l ara'


def tokenizer(text):
    # Simple whitespace-based tokenization
    return text.split()


def preprocess_arabic_text(text):
    """ 
    Preprocess Arabic text by removing diacritics and punctuation.
    
    Args:
        text (str): The input Arabic text.
    
    Returns:
        str: The preprocessed Arabic text.
    """
    # Remove Arabic diacritics (Tashkeel)
    arabic_diacritics = re.compile("""
                                ّ    | # Shadda
                                َ    | # Fatha
                                ً    | # Tanwin Fath
                                ُ    | # Damma
                                ٌ    | # Tanwin Damm
                                ِ    | # Kasra
                                ٍ    | # Tanwin Kasr
                                ْ    | # Sukun
                                ـ     # Tatweel/Kashida
                            """, re.VERBOSE)
    text = re.sub(arabic_diacritics, '', text)
    text = text.replace("\n", " ").replace("  ", " ")
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[a-zA-Z]', '', text)
    text = re.sub(r'[^\w\s]', '', text)

    # Remove punctuation
    punctuations = '''`÷×؛<>_()*&^%][ـ،/:"؟.,'{}~¦+|!”…“–ـ»«'''
    translator = str.maketrans('', '', punctuations)
    text = text.translate(translator)

    return text

def save_as_json(data, filename):
    """Save data as a JSON file."""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def process_pdf(pdf_path):
    """
    Process a PDF document and extract text chunks based on paragraphs for further analysis.

    Args:
        pdf_path (str): The path to the PDF file.

    Returns:
        None

    Raises:
        FileNotFoundError: If the specified PDF file does not exist.
        PermissionError: If the specified PDF file cannot be accessed due to insufficient permissions.

    Notes:
        This function uses the PyMuPDF library to open and process the PDF document.
        It extracts text from each page of the PDF, splits it into paragraphs,
        preprocesses the text, and saves each paragraph as a JSON object.

        The JSON object contains the following fields:
        - 'book_name': The name of the PDF file without the '.pdf' extension.
        - 'page_number': The page number of the paragraph within the PDF.
        - 'text_chunk': The processed text paragraph.

        The processed text paragraphs are saved as separate JSON files, with a unique filename
        based on the book name, page number, and paragraph index.

        This function relies on external libraries such as PyMuPDF, Pillow, and Tesseract.
        Make sure to install these dependencies before using this function.

    Example:
        process_pdf('/path/to/my_document.pdf')
    """
    book_name = os.path.basename(pdf_path).replace('.pdf', '')
    doc = fitz.open(pdf_path)

    for page_number, page in enumerate(doc, start=1):
        # Render the page as an image
        pix = page.get_pixmap()
        img_bytes = pix.tobytes("ppm")  # Convert the pixmap to bytes

        # Load it to PIL
        image = Image.open(io.BytesIO(img_bytes))
        text = pytesseract.image_to_string(image, config=TESSERACT_CONFIG)
        # Splitting text into paragraphs based on double newline
        paragraphs = text.strip().split('\n\n')

        # Process each paragraph separately
        for i, paragraph in enumerate(paragraphs):
            processed_paragraph = preprocess_arabic_text(paragraph)
            # Ensure the paragraph has at least 100 tokens
            if len(processed_paragraph.split()) >= 100:
                # Creating JSON object
                json_obj = {
                    'book_name': book_name,
                    'page_number': page_number,
                    'text_chunk': processed_paragraph
                }
                
                cwd = os.getcwd()
                filename = os.path.join(cwd, 'src', 'restricted_topic_detection', 'processed_dataset', f"{book_name}_page_{page_number}_paragraph_{i}.json")
                save_as_json(json_obj, filename)



if __name__ == "__main__":
    start_time = time.time()
    cwd = os.getcwd()
    
    directory_path = os.path.join(cwd, 'src', 'restricted_topic_detection', 'row_dataset')
    
    for filename in os.listdir(directory_path):
        if filename.endswith('.pdf'):
            process_pdf(os.path.join(directory_path, filename))
    
    print(f"Processing completed in {time.time() - start_time:.2f} seconds.")
