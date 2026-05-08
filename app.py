import sys
import os

# Fix sqlite3 for chromadb on Windows
try:
    import pysqlite3
    sys.modules["sqlite3"] = pysqlite3
except ImportError:
    pass

from dotenv import load_dotenv
import streamlit as st
from streamlit_community_navigation_bar import st_navbar
import warnings
import tempfile

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader, TextLoader, PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

warnings.filterwarnings('ignore')
load_dotenv()

# ── Streamlit page config (must be first Streamlit call) ──────────────────
st.set_page_config(page_title="RAG with Conversational Memory", layout="wide")

# ── API environment setup ─────────────────────────────────────────────────
os.environ["LANGCHAIN_TRACING_V2"]  = os.getenv("LANGCHAIN_TRACING_V2",  "false")
os.environ["LANGCHAIN_API_KEY"]     = os.getenv("LANGCHAIN_API_KEY",      "na")
os.environ["LANGCHAIN_ENDPOINT"]    = os.getenv("LANGCHAIN_ENDPOINT",     "https://api.smith.langchain.com")
os.environ["LANGCHAIN_PROJECT"]     = os.getenv("LANGCHAIN_PROJECT",      "rag-chatbot")

# ── Session state ─────────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "store" not in st.session_state:
    st.session_state["store"] = {}
if "retriever" not in st.session_state:
    st.session_state["retriever"] = None

# ── Initialise Gemini ─────────────────────────────────────────────────────
try:
    gemini_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    model = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash-latest",
        convert_system_message_to_human=True
    )
except Exception as e:
    st.error(f"❌ Error initialising Gemini — check your GOOGLE_API_KEY in .env file.\n\n{e}")
    st.stop()

# ── Helpers ───────────────────────────────────────────────────────────────
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in st.session_state["store"]:
        st.session_state["store"][session_id] = ChatMessageHistory()
    return st.session_state["store"][session_id]

def build_rag_chain(retriever):
    # History-aware retriever
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", "Given a chat history and the latest user question, "
                   "formulate a standalone question which can be understood "
                   "without the chat history. Do NOT answer it, just reformulate."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(
        model, retriever, contextualize_q_prompt
    )
    # QA chain
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Use the following context to answer "
                   "the user's question accurately.\n\nContext:\n{context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    qa_chain = create_stuff_documents_chain(model, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)
    return RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

def load_and_index(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(chunks, embedding=gemini_embeddings)
    return vectorstore.as_retriever()

# ── CSS ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
body, .stApp { font-family: 'Segoe UI', Arial, sans-serif !important; color: #e0e6f0 !important; }
.heading-box {
    background-color:#35365a; color:#fff; border-radius:16px;
    padding:18px 28px; margin-bottom:18px; font-size:2rem; font-weight:700;
}
.card {
    background:#2d2d44; border-radius:12px; padding:20px 24px;
    margin-bottom:14px; border-left:4px solid #3e68ff;
}
.stButton>button { background-color:#3e68ff; color:white; border-radius:8px; border:none; padding:10px 20px; }
.stButton>button:hover { background-color:#5e78ff; }
[data-testid="stSidebar"] { background-color:#1e1e2e; border-right:1px solid #383850; }
.footer {
    position:fixed; left:0; bottom:0; width:100%;
    background-color:#1e1e2e; border-top:1px solid #383850;
    color:#FFC300; text-align:center; padding:10px 0; z-index:100;
}
.stApp { padding-bottom:50px !important; }
.chat-bubble-user { background:#3e68ff; color:white; border-radius:12px; padding:10px 16px; margin:6px 0; max-width:80%; margin-left:auto; }
.chat-bubble-ai { background:#2d2d44; color:#e0e6f0; border-radius:12px; padding:10px 16px; margin:6px 0; max-width:80%; }
</style>
""", unsafe_allow_html=True)

# ── Navigation ────────────────────────────────────────────────────────────
selected_page = st_navbar(["Home", "How to Use", "About Us", "Team", "Contact Us", "Future Enhancements"])

# ── Footer ────────────────────────────────────────────────────────────────
st.markdown("<div class='footer'>© 2025 ASHI JAIN — All rights reserved.</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# PAGE: HOME
# ══════════════════════════════════════════════════════════════════════════
if selected_page == "Home":
    st.markdown("<div class='heading-box'>🤖 Intelligent Chatbot using Retrieval Augmented Generation (RAG)</div>", unsafe_allow_html=True)
    st.markdown("##### 📌 Welcome to the RAG Chatbot with Memory!")
    st.markdown("Harness the power of **AI + Retrieval** to get precise, document-specific answers — whether you're researching, studying, or building intelligent systems.")
    st.markdown("---")

    # Sidebar — Data Source
    with st.sidebar:
        st.markdown("## 📂 Data Source")
        source_type = st.radio("Select a source:", ["Web URL", "Upload File"])

        if source_type == "Web URL":
            url = st.text_input("Enter the URL to scrape:")
            if st.button("Load URL"):
                if url:
                    with st.spinner("Loading URL..."):
                        try:
                            loader = WebBaseLoader(url)
                            docs = loader.load()
                            st.session_state["retriever"] = load_and_index(docs)
                            st.success("✅ URL loaded and indexed!")
                        except Exception as e:
                            st.error(f"Error loading URL: {e}")
                else:
                    st.warning("Please enter a URL.")

        else:
            uploaded_file = st.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt"])
            if st.button("Load File"):
                if uploaded_file:
                    with st.spinner("Processing file..."):
                        try:
                            suffix = ".pdf" if uploaded_file.name.endswith(".pdf") else ".txt"
                            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                                tmp.write(uploaded_file.read())
                                tmp_path = tmp.name
                            loader = PyPDFLoader(tmp_path) if suffix == ".pdf" else TextLoader(tmp_path)
                            docs = loader.load()
                            st.session_state["retriever"] = load_and_index(docs)
                            st.success("✅ File loaded and indexed!")
                        except Exception as e:
                            st.error(f"Error processing file: {e}")
                else:
                    st.warning("Please upload a file.")

        st.markdown("---")
        st.markdown("### 📞 Contact")
        st.markdown("📧 jainashi2005@gmail.com")
        st.markdown("📱 +91-82696-65428")
        st.markdown("[LinkedIn](https://www.linkedin.com/in/ashi-jain-84938a27a/) | [GitHub](https://github.com/Ashijain30)")

    # Chat area
    st.markdown("#### 💬 Ask your question:")
    user_input = st.text_input("Enter your question:", key="user_input", label_visibility="collapsed")

    if st.button("Ask"):
        if not st.session_state["retriever"]:
            st.warning("⚠️ Please load a URL or file first using the sidebar.")
        elif not user_input.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Thinking..."):
                try:
                    chain = build_rag_chain(st.session_state["retriever"])
                    response = chain.invoke(
                        {"input": user_input},
                        config={"configurable": {"session_id": "ashi_session"}}
                    )
                    answer = response["answer"]
                    st.session_state["chat_history"].append(("You", user_input))
                    st.session_state["chat_history"].append(("AI", answer))
                except Exception as e:
                    st.error(f"Error generating response: {e}")

    # Display chat history
    if st.session_state["chat_history"]:
        st.markdown("---")
        st.markdown("#### 🗂️ Conversation History")
        for role, msg in st.session_state["chat_history"]:
            if role == "You":
                st.markdown(f"<div class='chat-bubble-user'>🧑 <b>You:</b> {msg}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='chat-bubble-ai'>🤖 <b>AI:</b> {msg}</div>", unsafe_allow_html=True)

        if st.button("🗑️ Clear Chat"):
            st.session_state["chat_history"] = []
            st.session_state["store"] = {}
            st.rerun()

# ══════════════════════════════════════════════════════════════════════════
# PAGE: HOW TO USE
# ══════════════════════════════════════════════════════════════════════════
elif selected_page == "How to Use":
    st.markdown("<div class='heading-box'>📖 How to Use</div>", unsafe_allow_html=True)
    steps = [
        ("1️⃣ Choose a Data Source", "In the sidebar on the Home page, select either **Web URL** or **Upload File**."),
        ("2️⃣ Load Your Data", "Enter a URL or upload a PDF/TXT file, then click the Load button. The system will process and index your content."),
        ("3️⃣ Ask Questions", "Type your question in the chat box and click **Ask**. The AI will retrieve relevant information and answer contextually."),
        ("4️⃣ Multi-turn Conversation", "The chatbot remembers your conversation history, so you can ask follow-up questions naturally."),
        ("5️⃣ Clear & Restart", "Use the **Clear Chat** button to reset the conversation and start fresh with new content."),
    ]
    for title, desc in steps:
        st.markdown(f"<div class='card'><b>{title}</b><br>{desc}</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# PAGE: ABOUT US
# ══════════════════════════════════════════════════════════════════════════
elif selected_page == "About Us":
    st.markdown("<div class='heading-box'>🌟 About Us</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='card'>
    Welcome to our RAG Chatbot — an intelligent assistant that bridges human curiosity and machine knowledge 
    through cutting-edge AI.<br><br>
    We are passionate developers dedicated to making information retrieval smarter, faster, and more contextual.<br><br>
    <b>Our Mission:</b> To make AI more human-centric by combining advanced language models with intuitive 
    user interfaces and real-world usability.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🚀 Tech Stack")
    tech = [
        ("🧠 Google Gemini", "State-of-the-art LLM for natural language understanding and generation"),
        ("🔗 LangChain", "Framework for building LLM-powered applications with memory and retrieval"),
        ("🗃️ ChromaDB", "Vector database for semantic document search and retrieval"),
        ("🖥️ Streamlit", "Fast, interactive web UI for Python ML applications"),
    ]
    for name, desc in tech:
        st.markdown(f"<div class='card'><b>{name}</b><br>{desc}</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# PAGE: TEAM
# ══════════════════════════════════════════════════════════════════════════
elif selected_page == "Team":
    st.markdown("<div class='heading-box'>👥 Our Team</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='card'>
    <b>Ashi Jain</b> — Lead Developer<br>
    B.Tech CSE (AI/ML), VIT Bhopal University<br>
    📧 jainashi2005@gmail.com<br>
    🔗 <a href='https://www.linkedin.com/in/ashi-jain-84938a27a/' style='color:#3e68ff'>LinkedIn</a> | 
    <a href='https://github.com/Ashijain30' style='color:#3e68ff'>GitHub</a>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# PAGE: CONTACT US
# ══════════════════════════════════════════════════════════════════════════
elif selected_page == "Contact Us":
    st.markdown("<div class='heading-box'>📞 Contact Us</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='card'>
    <b>Ashi Jain</b><br><br>
    📧 <a href='mailto:jainashi2005@gmail.com' style='color:#3e68ff'>jainashi2005@gmail.com</a><br>
    📱 +91-82696-65428<br>
    🔗 <a href='https://www.linkedin.com/in/ashi-jain-84938a27a/' style='color:#3e68ff'>LinkedIn</a><br>
    💻 <a href='https://github.com/Ashijain30' style='color:#3e68ff'>GitHub — Ashijain30</a>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# PAGE: FUTURE ENHANCEMENTS
# ══════════════════════════════════════════════════════════════════════════
elif selected_page == "Future Enhancements":
    st.markdown("<div class='heading-box'>🛣️ Future Enhancements</div>", unsafe_allow_html=True)
    enhancements = [
        ("📁 More File Types", "Support for DOCX, Excel, CSV, and PowerPoint files"),
        ("🔐 User Authentication", "Login system with personalised chat history per user"),
        ("💾 Persistent Memory", "Long-term memory storage across sessions"),
        ("📤 Export Chat Logs", "Download conversation history as PDF or TXT"),
        ("🌐 Multi-language Support", "Answer questions in multiple languages"),
        ("☁️ Cloud Deployment", "Deploy on Streamlit Cloud or AWS for public access"),
    ]
    for title, desc in enhancements:
        st.markdown(f"<div class='card'><b>{title}</b><br>{desc}</div>", unsafe_allow_html=True)
