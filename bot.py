import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import requests
import os
from dotenv import load_dotenv

# --- SETUP ---
load_dotenv()
st.set_page_config(page_title="NUML Karachi GPT", page_icon="🏛️")

# --- RAG SETUP ---
@st.cache_resource
def setup_rag():
    try:
        with open("numl_data.txt", "r", encoding='utf-8') as f:
            numl_data = f.read()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            separators=["\n\n", "\n", " ", ""]
        )
        documents = text_splitter.split_text(numl_data)
        
        embeddings = HuggingFaceEmbeddings(
            model_name="paraphrase-MiniLM-L3-v2",
            encode_kwargs={'normalize_embeddings': True}
        )
        
        return Chroma.from_texts(
            documents, 
            embeddings, 
            persist_directory="./numl_db"
        )
    except Exception as e:
        st.error(f"Initialization failed: {str(e)}")
        return None

vector_db = setup_rag()

# --- CHAT INTERFACE ---
st.title("🏛️ NUML GPT")
st.markdown("""
<style>
    .title {
        color: #1a237e;
        font-size: 2.5rem;
        text-align: center;
        padding: 20px 0;
        border-bottom: 2px solid #1a237e;
        margin-bottom: 30px;
    }
</style>
""", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me about NUML Karachi!"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("Your question"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    with st.spinner("Thinking..."):
        try:
            docs = vector_db.similarity_search(prompt, k=2)
            context = "\n".join([doc.page_content for doc in docs])
            
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "deepseek/deepseek-r1:free",
                    "messages": [
                        {"role": "system", "content": f"Use this context: {context}"},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.2
                }
            )
            
            ai_response = response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            ai_response = f"Error: {str(e)}"

    st.session_state.messages.append({"role": "assistant", "content": ai_response})
    st.chat_message("assistant").write(ai_response)