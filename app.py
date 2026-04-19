import streamlit as st
import os
from dotenv import load_dotenv
import tempfile

# LangChain components (stable)
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Groq LLM
from langchain_google_genai import ChatGoogleGenerativeAI

# -----------------------
# 🔑 Load API Key
# -----------------------
load_dotenv()
# os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# -----------------------
# 🧠 Session State
# -----------------------
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "llm" not in st.session_state:
    st.session_state.llm = None

if "messages" not in st.session_state:
    st.session_state.messages = []

# -----------------------
# 📄 Process PDFs
# -----------------------
def process_pdfs(uploaded_files):
    documents = []

    for file in uploaded_files:
        if file is None or file.size == 0:
            st.warning(f"{file.name} is empty or invalid. Skipping.")
            continue

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file.getbuffer())
            tmp.flush()  # ✅ VERY IMPORTANT

            try:
                loader = PyPDFLoader(tmp.name)
                documents.extend(loader.load())
            except Exception as e:
                st.error(f"Error processing {file.name}: {e}")

    # Split text
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    docs = splitter.split_documents(documents)

    # Embeddings (HuggingFace)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Vector store (FAISS)
    vectorstore = FAISS.from_documents(docs, embeddings)

    # Groq LLM
    llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite-preview",
    temperature=0.3,
    google_api_key=st.secrets["GROQ_API_KEY"]  # 👈 important
)
    return vectorstore, llm

# -----------------------
# 🎨 UI Setup
# -----------------------
st.set_page_config(page_title="PDF Chatbot", layout="wide", page_icon="📄")
st.title("📄 Chat with PDFs")

# -----------------------
# 📂 Sidebar Upload
# -----------------------
with st.sidebar:
    st.markdown("## ➕ Upload PDFs")

    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded_files:
        st.success("✅ Files uploaded:")
        for file in uploaded_files:
            st.write(f"📄 {file.name}")

        if st.button("Process PDFs"):
            with st.spinner("Processing PDFs..."):
                vs, llm = process_pdfs(uploaded_files)
                st.session_state.vectorstore = vs
                st.session_state.llm = llm
            st.success("✅ Ready to chat!")

# -----------------------
# 💬 Chat History
# -----------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# -----------------------
# ⌨️ Chat Input
# -----------------------
user_input = st.chat_input("Ask something about your PDFs...")

# -----------------------
# 🚀 Handle Query
# -----------------------
if user_input:
    if not st.session_state.vectorstore:
        st.warning("⚠️ Please upload and process PDFs first.")
    else:
        # Show user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Retrieve relevant docs
        docs = st.session_state.vectorstore.similarity_search(user_input, k=3)

        # Combine context
        context = "\n\n".join([doc.page_content for doc in docs])

        # Create prompt
        prompt = f"""
        Answer the question based only on the context below:

        {context}

        Question: {user_input}
        """

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = st.session_state.llm.invoke(prompt)

                if isinstance(result.content, list):
                   response = " ".join([chunk["text"] for chunk in result.content if chunk["type"] == "text"])
                else:
                   response = result.content
                st.markdown(response)

        # Save response
        st.session_state.messages.append({"role": "assistant", "content": response})
