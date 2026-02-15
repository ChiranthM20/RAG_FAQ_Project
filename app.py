import streamlit as st
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_ollama import OllamaLLM


# =========================
# STREAMLIT CONFIG
# =========================

st.set_page_config(page_title="RAG FAQ System", layout="centered")

st.title("📚 FAQ System using RAG (Offline)")
st.write("Upload PDF and ask questions (No API, No Limits)")


# =========================
# FILE UPLOAD
# =========================

uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file:

    os.makedirs("data", exist_ok=True)

    file_path = os.path.join("data", uploaded_file.name)

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success("✅ File uploaded successfully!")


# =========================
# LOAD PDF
# =========================

def load_pdf(path):

    loader = PyPDFLoader(path)

    return loader.load()


# =========================
# SPLIT TEXT
# =========================

def split_docs(docs):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    return splitter.split_documents(docs)


# =========================
# CREATE DB
# =========================

def create_db(chunks):

    embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
    )


    db = FAISS.from_documents(chunks, embeddings)

    db.save_local("faiss_db")

    return db


# =========================
# LOAD DB
# =========================

def load_db():

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return FAISS.load_local(
        "faiss_db",
        embeddings,
        allow_dangerous_deserialization=True
    )


# =========================
# ASK LLAMA (OLLAMA)
# =========================

def ask_llama(context, question):

    llm = OllamaLLM(model="llama3")

    prompt = f"""
You are a helpful assistant.

Answer ONLY from the context below.
If not found, say: Not available in document.

Context:
{context}

Question:
{question}
"""

    response = llm.invoke(prompt)

    return response


# =========================
# PROCESS DOCUMENT
# =========================

if uploaded_file:

    with st.spinner("📄 Processing document..."):

        docs = load_pdf(file_path)

        chunks = split_docs(docs)

        create_db(chunks)

    st.success("✅ Document processed successfully!")


# =========================
# QUESTION INPUT
# =========================

st.divider()

question = st.text_input("Ask your question")


if st.button("Get Answer"):

    if not question:
        st.warning("⚠️ Please enter a question")

    elif not os.path.exists("faiss_db"):
        st.warning("⚠️ Please upload PDF first")

    else:

        with st.spinner("🔍 Finding answer..."):

            db = load_db()

            docs = db.similarity_search(question, k=3)

            context = ""

            for d in docs:
                context += d.page_content + "\n\n"

            answer = ask_llama(context, question)

        st.subheader("📌 Answer")
        st.write(answer)
