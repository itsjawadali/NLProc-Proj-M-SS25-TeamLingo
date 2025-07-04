# utils/file_loader.py

def load_text_file(path):
    """
    Loads and returns the content of a .txt file.
    """
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def load_pdf(path):
    """
    Loads and returns the text content from a PDF file.
    Requires pdfplumber.
    """
    import pdfplumber
    text = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text
