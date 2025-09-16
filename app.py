import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.chains import RetrievalQA

# Optional OpenAI support
from langchain_openai import OpenAI

# -------------------- Helper Functions --------------------

def extract_text_from_pdf(uploaded_file):
    """Extract text from uploaded PDF file"""
    pdf_reader = PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text


def create_vectorstore(text, use_openai=False):
    """Build FAISS vectorstore with embeddings"""
    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)

    # Use free local embeddings by default
    if use_openai and os.getenv("OPENAI_API_KEY"):
        from langchain_community.embeddings import OpenAIEmbeddings
        embeddings = OpenAIEmbeddings()
    else:
        embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    return FAISS.from_texts(chunks, embeddings)


# -------------------- Streamlit App --------------------

st.set_page_config(page_title="📄 Resume RAG Chatbot", layout="wide")
st.title("📄 Resume RAG Chatbot (LangChain + Streamlit)")

with st.sidebar:
    st.header("⚙️ Settings")
    use_openai = st.checkbox("Use OpenAI (needs API key + quota)", value=False)
    top_k = st.number_input("Top-k retrieved chunks", min_value=1, max_value=10, value=4)

uploaded_file = st.file_uploader("Upload a Resume PDF", type=["pdf"])

if "vectorstore" not in st.session_state:
    st.session_state["vectorstore"] = None
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

if uploaded_file:
    with st.spinner("📄 Extracting text from PDF..."):
        text = extract_text_from_pdf(uploaded_file)

    st.success("✅ Text extracted from PDF")

    with st.spinner("🔎 Creating vectorstore..."):
        st.session_state["vectorstore"] = create_vectorstore(text, use_openai=use_openai)

    st.success("✅ Vectorstore ready! Ask questions below:")

if st.session_state["vectorstore"]:
    question = st.text_input("Ask a question about the resume:")

    if st.button("Ask") and question.strip():
        retriever = st.session_state["vectorstore"].as_retriever(search_kwargs={"k": top_k})

        if use_openai and os.getenv("OPENAI_API_KEY"):
            llm = OpenAI(temperature=0)
            qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
            with st.spinner("🤖 Thinking with OpenAI..."):
                answer = qa.run(question)
        else:
            # Just return retrieved chunks (no AI generation)
            docs = retriever.get_relevant_documents(question)
            if not docs:
                answer = "No relevant information found."
            else:
                answer = "\n\n---\n\n".join([d.page_content for d in docs])

        st.session_state["chat_history"].append((question, answer))

    st.write("### 💬 Chat History")
    for q, a in reversed(st.session_state["chat_history"]):
        st.markdown(f"**Q:** {q}")
        st.write(a)
        st.write("---")

else:
    st.info("📤 Please upload a PDF resume to begin.")
