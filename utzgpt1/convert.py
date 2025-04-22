import os
from PyPDF2 import PdfReader

# Directory containing your book PDFs
PDF_DIR = "books"
# Output file
OUTPUT_FILE = "input.txt"

def extract_text_from_pdf_with_pages(pdf_path):
    try:
        reader = PdfReader(pdf_path)
        all_text = ""
        for i, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            all_text += f"--- Page {i} ---\n{text.strip()}\n\n"
        return all_text
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
        return ""

def main():
    with open(OUTPUT_FILE, "w", encoding="utf-8") as output_file:
        for filename in sorted(os.listdir(PDF_DIR)):
            if filename.lower().endswith(".pdf"):
                title = os.path.splitext(filename)[0]
                pdf_path = os.path.join(PDF_DIR, filename)
                print(f"Processing: {title}")
                text = extract_text_from_pdf_with_pages(pdf_path)
                output_file.write(f"=== {title} ===\n{text}\n")

    print(f"Done! Output written to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()