import streamlit as st
import json
import os
import chromadb
from chromadb.utils import embedding_functions
import google.generativeai as genai

# ==========================================
# 1. CẤU HÌNH HỆ THỐNG
# ==========================================
st.set_page_config(page_title="Chatbot Hộ Chiếu Việt Nam", page_icon="🇻🇳")

if "GOOGLE_API_KEY" in st.secrets:
    api_key = st.secrets["GOOGLE_API_KEY"]
else:
    api_key = "AIzaSyCzcZwCm4cycmjT2Q1biZNYDfbI5sh9Cr4"

genai.configure(api_key=api_key)

# ------------------------------------------
@st.cache_resource
def find_available_model():
    try:
        models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        # Ưu tiên flash, rồi đến pro
        for m_name in models:
            if "1.5-flash" in m_name: return m_name
        for m_name in models:
            if "pro" in m_name: return m_name
        return models[0] if models else "gemini-pro"
    except Exception:
        return "gemini-1.5-flash" 

AVAILABLE_MODEL = find_available_model()

# ==========================================
# 2. XỬ LÝ DỮ LIỆU
# ==========================================
@st.cache_resource
def get_embedding_function():
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="paraphrase-multilingual-MiniLM-L12-v2"
    )

@st.cache_resource
def init_vector_db():
    if not os.path.exists("TAI_LIEU_RB.json"):
        return None
    client = chromadb.PersistentClient(path="chroma_db_data")
    emb_func = get_embedding_function()
    try:
        collection = client.get_collection(name="RAG_procedure", embedding_function=emb_func)
    except:
        collection = client.create_collection(name="RAG_procedure", embedding_function=emb_func)
        with open("TAI_LIEU_RB.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        collection.add(
            ids=[str(i) for i in range(len(data))],
            documents=[item["content_text"] for item in data],
            metadatas=[{"url": item["url"], "title": item["title"], "hierarchy": item["hierarchy"]} for item in data]
        )
    return collection

collection = init_vector_db()

# ==========================================
# 3. LOGIC CHATBOT
# ==========================================
def get_chatbot_response(user_query):
    results = collection.query(query_texts=[user_query], n_results=3)
    context_text = ""
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        context_text += f"\n[Nguồn: {meta['title']}]\n{doc}\nLink: {meta['url']}\n---\n"

    full_prompt = f"Ngữ cảnh: {context_text}\n\nCâu hỏi: {user_query}"

    model = genai.GenerativeModel(model_name=AVAILABLE_MODEL)
    response = model.generate_content(full_prompt)
    return response.text

# ==========================================
# 4. GIAO DIỆN
# ==========================================
st.title("🇻🇳 Trợ lý ảo Hộ chiếu")
st.write(f" Đang sử dụng model: `{AVAILABLE_MODEL}`")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Nhập câu hỏi về thủ tục hộ chiếu...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Đang tra cứu dữ liệu..."):
            try:
                answer = get_chatbot_response(user_input)
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error(f"Lỗi: {str(e)}")
