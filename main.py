import PyPDF2
from transformers import pipeline

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return text

def summarize_text(text, max_length=150):
    summarizer = pipeline("summarization")
    summary = summarizer(text, max_length=max_length, min_length=30, do_sample=False)
    return summary[0]['summary_text']

if __name__ == "__main__":
    pdf_path = "sample.pdf"  # Replace with your PDF file name
    print("Extracting text from PDF...")
    text = extract_text_from_pdf(pdf_path)

    if len(text) > 1000:
        print("Summarizing content...")
        summary = summarize_text(text)
        print("\n--- Summary ---\n", summary)
    else:
        print("Text is too short for summarization.")
        print("\n--- Full Text ---\n", text)
