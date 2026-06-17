import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter  
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document  
from groq import Groq
from dotenv import load_dotenv

from embedding import load_embeddings
from hyde import hyde_query_expansion  
from extract_text_from_pdfs import extract_text_from_pdfs  
from ai import z


load_dotenv()
with st.sidebar:
    st.header("⚙️ Groq Settings")
    api_key = st.text_input(
        "Groq API Key", 
        type="password", 
        value="", # can be setted in .env file or can be directly setted in web
        help="Get a free key at console.groq.com"
    )
    
# CLIENT
@st.cache_resource
def get_client():
    return Groq(api_key=os.getenv("GROQ_API_KEY"))

client = get_client()


st.set_page_config(page_title="HR Policy Assistant", page_icon="🧑‍💼", layout="wide")

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
            st.caption(f"• {file.name} ({file.size/1024:.1f} KB)")
    else:
        st.info("💡 Upload HR policy PDFs to get started")

with right_col:
    st.subheader("❓ Ask Your Policy Question")

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
    file_ids = [(f.name, f.size) for f in uploaded_files]

    if "file_ids" not in st.session_state or st.session_state.file_ids != file_ids:
        with st.spinner("🔄 Processing HR policies..."):
            raw_texts = extract_text_from_pdfs(uploaded_files)

            
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=600,
                chunk_overlap=150,
                separators=["\n\n", "\n", ". ", "! ", "? ", " "]
            )

            all_docs = []
            for text, doc_name in raw_texts:
                chunks = splitter.split_text(text)
                for chunk in chunks:
                    all_docs.append(Document(
                        page_content=chunk,
                        metadata={"source": doc_name}
                    ))

            total_chunks = len(all_docs)
            st.caption(f"📊 Indexed {total_chunks} chunks from {len(raw_texts)} document(s)")

            embeddings = load_embeddings()
            st.session_state.db = FAISS.from_documents(all_docs, embeddings)
            st.session_state.file_ids = file_ids
            st.session_state.uploaded_files = uploaded_files

        st.success("🚀 HR Knowledge Base Ready!")

    db = st.session_state.db

    # GENERATE ANSWER
    # FIXED: Now properly gated on `submitted` — won't re-run on every Streamlit rerender
    if submitted and query and "db" in st.session_state:
        with st.spinner("📖 Searching policies..."):

            
            search_query = query
            if use_hyde:
                with st.spinner("🧠 Expanding query with HyDE..."):
                    hypothetical_doc = hyde_query_expansion(query)
                    search_query = hypothetical_doc

            docs_with_scores = db.similarity_search_with_score(search_query, k=10)

           
            SCORE_THRESHOLD = 1.2
            filtered_docs = [
                doc for doc, score in docs_with_scores if score < SCORE_THRESHOLD
            ]

            # Fallback: if nothing passes threshold, use top 4 anyway
            if not filtered_docs:
                filtered_docs = [doc for doc, _ in docs_with_scores[:4]]

            # MMR pass for diversity
            mmr_docs = db.max_marginal_relevance_search(
                search_query, k=6, fetch_k=16
            )

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
            
            response = z(context,query)

            answer = response.choices[0].message.content

        # ANSWER DISPLAY
        st.markdown("---")
        st.markdown("### 📋 **Policy Answer**")
        st.markdown(answer)


        with st.expander("🔍 View Retrieved Policy Context"):
            st.markdown(f"**{len(final_docs)} chunks retrieved** from knowledge base")
            if use_hyde:
                st.markdown("**🧠 HyDE Expansion used:**")
                st.caption(hypothetical_doc)
            st.markdown("---")
            for i, doc in enumerate(final_docs):
                source = doc.metadata.get("source", "Unknown")
                st.markdown(f"**Chunk {i+1}** — `{source}`")
                st.text(doc.page_content[:400] + ("..." if len(doc.page_content) > 400 else ""))
                st.markdown("---")

    elif submitted and not query:
        st.warning("❓ Please enter your HR question first!")

# FOOTER 
st.markdown("---")
st.markdown("*Powered by company's HR policies. Always verify critical decisions with the HR department.*")