import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st

st.set_page_config(page_title="HR Policy Assistant", page_icon="🧑‍💼", layout="wide")

from embedding.embedding import load_embeddings
from models.hyde import hyde_query_expansion  
from text_extract.extract_text_from_pdfs import extract_text_from_pdfs  
from models.ai import generate_output
from langchain_text_splitters import RecursiveCharacterTextSplitter  
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# Groq API input
with st.sidebar:
    st.header("⚙️ Groq Settings")
    api_key = st.text_input(
        "Groq API Key", 
        type="password", 
        value="",
        help="Get a free key at console.groq.com"
    )


# Background image
bg_url = "https://images.unsplash.com/photo-1620641788421-7a1c342ea42e?w=1920&auto=format&fit=crop&q=80"

st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("{bg_url}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🧑‍💼 HR Policy Assistant")
st.caption("Instant, policy-backed answers to all your HR questions.")



# LAYOUT 
left_col, right_col = st.columns([1, 2], gap="large")

with left_col:
    st.subheader("📚 Upload HR Documents")
    col1, col2 = st.columns(2)
    with col1:
        st.info("""
        **📏 File Guidelines**
        - Max **15MB** per file
        - **Ideal**: 5-30 page policies  
        - **Large books**: First 20-25 pages only

        💡 Upload individual policies for best results
        """)
        uploaded_files = st.file_uploader(
            "Upload multiple policy PDFs (Leave Policy, Code of Conduct, etc.)",
            type=["pdf", "txt"],
            accept_multiple_files=True
        )

    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} document(s) loaded")
        for file in uploaded_files:
            file_size = getattr(file, "size", 0) or 0
            st.caption(f"• {file.name} ({file_size/1024:.1f} KB)")
    else:
        st.info("💡 Upload HR policy PDFs to get started")

with right_col:
    st.subheader("❓ Ask Your Policy Question")

    st.markdown("""
    <style>
    div[data-testid="stAlert"] p {
        color: black !important;   /* Change to any color */
    }
    </style>
    """, unsafe_allow_html=True)

    st.info("Make sure to set your Groq API key in the sidebar before generating messages.", icon="💡")
    with st.form("query_form"):
        query = st.text_input(
            "e.g., 'maternity leave policy' or 'grievance procedure'",
            placeholder="What HR policy do you need clarified?"
        )
        use_hyde = st.checkbox(
            "🧠 Enhanced retrieval (HyDE)",
            value=True,
            help="Uses AI to expand your query for better document matching. Slightly slower but more accurate."
        )
        submitted = st.form_submit_button("🔍 Get Policy Answer", use_container_width=True)


# VECTOR DB BUILD 
if uploaded_files:
    file_ids = [(f.name, getattr(f, "size", 0)) for f in uploaded_files]

    if "file_ids" not in st.session_state or st.session_state.file_ids != file_ids:
        with st.spinner("🔄 Processing HR policies..."):
            try:
                raw_texts = extract_text_from_pdfs(uploaded_files)

                if not raw_texts:
                    st.error("⚠️ No readable text found in the uploaded files. Try different PDFs or TXT files.")
                    st.session_state.pop("db", None)
                    st.session_state.pop("file_ids", None)
                else:
                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=600,
                        chunk_overlap=150,
                        separators=["\n\n", "\n", ". ", "! ", "? ", " "]
                    )

                    all_docs = []
                    for text, doc_name in raw_texts:
                        chunks = splitter.split_text(text)
                        for chunk in chunks:
                            if chunk.strip():
                                all_docs.append(Document(
                                    page_content=chunk,
                                    metadata={"source": doc_name}
                                ))

                    if not all_docs:
                        st.error("⚠️ Documents produced no indexable chunks.")
                        st.session_state.pop("db", None)
                        st.session_state.pop("file_ids", None)
                    else:
                        st.caption(f"📊 Indexed {len(all_docs)} chunks from {len(raw_texts)} document(s)")

                        embeddings = load_embeddings()
                        st.session_state.db = FAISS.from_documents(all_docs, embeddings)
                        st.session_state.file_ids = file_ids

                        st.success("🚀 HR Knowledge Base Ready!")
            except Exception as e:
                st.error(f"⚠️ Failed to process documents: {e}")
                st.session_state.pop("db", None)
                st.session_state.pop("file_ids", None)

    db = st.session_state.get("db")

    # GENERATE ANSWER
    # FIXED: Now properly gated on `submitted` — won't re-run on every Streamlit rerender
    if submitted:
        if not api_key:
            st.warning("⚠️ Please add Groq api key in sidebar")
        elif not query or not query.strip():
            st.warning("❓ Please enter your HR question first!")
        elif db is None:
            st.warning("📚 Please wait for documents to finish indexing, then ask again.")
        else:
            try:
                with st.spinner("📖 Searching policies..."):
                    search_query = query
                    hypothetical_doc = None
                    if use_hyde:
                        with st.spinner("🧠 Expanding query with HyDE..."):
                            hypothetical_doc = hyde_query_expansion(query, api_key)
                            search_query = hypothetical_doc or query

                    docs_with_scores = db.similarity_search_with_score(search_query, k=10)
                    if not docs_with_scores:
                        st.warning("🔍 No matching policy chunks found. Try rephrasing your question.")
                    else:
                        SCORE_THRESHOLD = 1.2
                        filtered_docs = [
                            doc for doc, score in docs_with_scores if score < SCORE_THRESHOLD
                        ]

                        # Fallback: if nothing passes threshold, use top 4 anyway
                        if not filtered_docs:
                            filtered_docs = [doc for doc, _ in docs_with_scores[:4]]

                        # MMR pass for diversity
                        try:
                            mmr_docs = db.max_marginal_relevance_search(
                                search_query, k=6, fetch_k=16
                            )
                        except Exception:
                            mmr_docs = []

                        # Merging: MMR for diversity + threshold-filtered for quality
                        seen_contents = set()
                        final_docs = []
                        for doc in (mmr_docs + filtered_docs):
                            key = doc.page_content[:100]  # Dedup by first 100 chars
                            if key not in seen_contents:
                                seen_contents.add(key)
                                final_docs.append(doc)
                            if len(final_docs) >= 6:
                                break

                        context_parts = []
                        total_chars = 0
                        MAX_CONTEXT = 10000

                        for doc in final_docs:
                            chunk = doc.page_content
                            source = doc.metadata.get("source", "Unknown")

                            labeled_chunk = f"[Source: {source}]\n{chunk}"
                            if total_chars + len(labeled_chunk) > MAX_CONTEXT:
                                break
                            context_parts.append(labeled_chunk)
                            total_chars += len(labeled_chunk)

                        context = "\n\n---\n\n".join(context_parts)
                        if not context.strip():
                            st.warning("🔍 Retrieved chunks were empty. Try rephrasing your question.")
                        else:
                            response = generate_output(context, query, api_key)
                            answer = response.choices[0].message.content

                            # ANSWER DISPLAY
                            st.markdown("---")
                            st.markdown("### 📋 **Policy Answer**")
                            st.markdown(answer)

                            with st.expander("🔍 View Retrieved Policy Context"):
                                st.markdown(f"**{len(final_docs)} chunks retrieved** from knowledge base")
                                if use_hyde and hypothetical_doc:
                                    st.markdown("**🧠 HyDE Expansion used:**")
                                    st.caption(hypothetical_doc)
                                st.markdown("---")
                                for i, doc in enumerate(final_docs):
                                    source = doc.metadata.get("source", "Unknown")
                                    st.markdown(f"**Chunk {i+1}** — `{source}`")
                                    st.text(doc.page_content[:400] + ("..." if len(doc.page_content) > 400 else ""))
                                    st.markdown("---")
            except Exception as e:
                st.error(f"⚠️ Could not generate an answer: {e}")
elif submitted:
    if not api_key:
        st.warning("⚠️ Please add Groq api key in sidebar")
    elif not query or not query.strip():
        st.warning("❓ Please enter your HR question first!")
    else:
        st.warning("📚 Please upload HR policy documents first!")

# FOOTER 
st.markdown("---")
st.markdown("*Powered by company's HR policies. Always verify critical decisions with the HR department.*")