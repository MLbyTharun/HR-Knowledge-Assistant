from pypdf import PdfReader
import streamlit as st

# extracts raw text from PDFs safely without crashing with large files
def extract_text_from_pdfs(files, max_chars_per_pdf=25000):
    texts = []
    for file in files:
        if file.size > 15 * 1024 * 1024:
            st.warning(f"📄 {file.name} too large, skipping.")
            continue

        pdf = PdfReader(file)
        text = ""
        page_num = 0

        for page in pdf.pages:
            page_text = page.extract_text() or ""
            page_num += 1
            
            text += f"\n{page_text}"

            if len(text) >= max_chars_per_pdf:
                st.info(f"📄 {file.name}: Using first {page_num} pages (content limit reached)")
                break

        texts.append((text, file.name))
    return texts
