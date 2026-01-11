import streamlit as st
import json
import os
import chromadb
from chromadb.utils import embedding_functions
import google.generativeai as genai

# ==========================================
# 1. CẤU HÌNH HỆ THỐNG & BẢO MẬT
# ==========================================
st.set_page_config(
    page_title="Chatbot Hộ Chiếu Việt Nam",
    page_icon="🇻🇳",
    layout="centered"
)

# Lấy API Key từ Secrets của Streamlit Cloud (Bắt buộc để không bị lỗi Leaked Key)
if "GOOGLE_API_KEY" in st.secrets:
    api_key = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=api_key)
else:
    st.error("❌ Lỗi: Chưa tìm thấy API Key. Hãy thêm GOOGLE_API_KEY vào mục Secrets trên Streamlit Cloud.")
    st.info("Hướng dẫn: Settings -> Secrets -> Dán: GOOGLE_API_KEY = 'Mã_API_Của_Bạn'")
    st.stop()

# ------------------------------------------
# TỰ ĐỘNG TÌM MODEL KHẢ DỤNG
# ------------------------------------------
@st.cache_resource
def find_available_model():
    try:
        # Liệt kê các model mà Key của bạn có quyền sử dụng
        models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        # Ưu tiên lấy bản 1.5 flash, nếu không có thì lấy các bản khác
        for m_name in models:
            if "1.5-flash" in m_name: return m_name
        for m_name in models:
            if "pro" in m_name: return m_name
        return models[0] if models else "gemini-1.5-flash"
    except Exception:
        return "gemini-1.5-flash"

AVAILABLE_MODEL = find_available_model()

# ==========================================
# 2. XỬ LÝ DỮ LIỆU & VECTOR DB
# ==========================================
JSON_FILE = "TAI_LIEU_RB.json"
CHROMA_DB_PATH = "chroma_db_data"

@st.cache_resource
def get_embedding_function():
    # Model embedding đa ngôn ngữ nhẹ cho Cloud
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="paraphrase-multilingual-MiniLM-L12-v2"
    )

@st.cache_resource
def init_vector_db():
    if not os.path.exists(JSON_FILE):
        return None
    
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    emb_func = get_embedding_function()
    
    try:
        collection = client.get_collection(name="RAG_procedure", embedding_function=emb_func)
    except:
        collection = client.create_collection(name="RAG_procedure", embedding_function=emb_func)
        with open(JSON_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Nạp dữ liệu vào database
        collection.add(
            ids=[str(i) for i in range(len(data))],
            documents=[item["content_text"] for item in data],
            metadatas=[
                {"url": item["url"], "title": item["title"], "hierarchy": item["hierarchy"]}
                for item in data
            ]
        )
    return collection

collection = init_vector_db()

# ==========================================
# 3. LOGIC XỬ LÝ CHAT (RAG)
# ==========================================
def get_chatbot_response(user_query):
    # Tìm kiếm 3 đoạn thông tin liên quan nhất
    results = collection.query(query_texts=[user_query], n_results=3)
    
    context_text = ""
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        context_text += f"\n[Nguồn: {meta['title']}]\n{doc}\nLink: {meta['url']}\n---\n"

    full_prompt = f"""Bạn là chuyên gia tư vấn thủ tục hành chính tại Việt Nam. 
Hãy sử dụng thông tin trong Ngữ cảnh dưới đây để trả lời câu hỏi một cách chính xác và thân thiện.
Nếu thông tin không có trong Ngữ cảnh, hãy hướng dẫn người dùng liên hệ Cổng Dịch vụ công hoặc Cơ quan Công an.

NGỮ CẢNH:
{context_text}

CÂU HỎI: {user_query}
"""

    model = genai.GenerativeModel(model_name=AVAILABLE_MODEL)
    response = model.generate_content(full_prompt)
    return response.text

# ==========================================
# 4. GIAO DIỆN NGƯỜI DÙNG
# ==========================================
st.title("🇻🇳 Trợ lý ảo Thủ tục Hộ chiếu")
st.write(f"🤖 Đang sử dụng model: `{AVAILABLE_MODEL}`")

if collection is None:
    st.error(f"❌ Thiếu file dữ liệu `{JSON_FILE}` trên GitHub!")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Hiển thị lại các tin nhắn cũ
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Ô nhập câu hỏi
user_input = st.chat_input("Hỏi về cấp hộ chiếu, lệ phí, thủ tục...")

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

# Sidebar bổ sung
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/a/a1/Emblem_of_Vietnam.svg/512px-Emblem_of_Vietnam.svg.png", width=100)
    st.header("Thông tin")
    st.info("Ứng dụng hỗ trợ tra cứu các thủ tục hành chính về Hộ chiếu phổ thông.")
    if st.button("Làm mới Chat"):
        st.session_state.messages = []
        st.rerun()
