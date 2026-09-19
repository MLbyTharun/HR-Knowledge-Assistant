from pypdf import PdfReader
import streamlit as st


def _read_txt_file(file):
    """Read a plain-text upload, handling encodings gracefully."""
    try:
        file.seek(0)
    except Exception:
        pass
    raw = file.read()
    if isinstance(raw, bytes):
        for encoding in ("utf-8", "utf-8-sig", "latin-1"):
            try:
                return raw.decode(encoding)
            except (UnicodeDecodeError, ValueError):
                continue
        return raw.decode("utf-8", errors="ignore")
    return str(raw)


# extracts raw text from PDFs safely without crashing with large files
def extract_text_from_pdfs(files, max_chars_per_pdf=25000):
    texts = []
    for file in files:
        file_name = getattr(file, "name", "unknown file")
        file_size = getattr(file, "size", 0) or 0
        if file_size > 15 * 1024 * 1024:
            st.warning(f"📄 {file_name} too large, skipping.")
            continue

        try:
            # Plain-text files: read directly instead of PdfReader.
            if file_name.lower().endswith(".txt"):
                text = _read_txt_file(file)
            else:
                try:
                    file.seek(0)
                except Exception:
                    pass
                pdf = PdfReader(file)
                if getattr(pdf, "is_encrypted", False):
                    try:
                        pdf.decrypt("")
                    except Exception:
                        st.warning(f"📄 {file_name} is encrypted, skipping.")
                        continue
                text = ""
                page_num = 0
                for page in pdf.pages:
                    try:
                        page_text = page.extract_text() or ""
                    except Exception:
                        page_text = ""
                    page_num += 1
                    text += f"\n{page_text}"
                    if len(text) >= max_chars_per_pdf:
                        st.info(f"📄 {file_name}: Using first {page_num} pages (content limit reached)")
                        break

            if not text.strip():
                st.warning(f"📄 {file_name} has no extractable text, skipping.")
                continue

            texts.append((text, file_name))
        except Exception as e:
            st.warning(f"📄 {file_name} could not be read ({e}), skipping.")
            continue
    return texts
