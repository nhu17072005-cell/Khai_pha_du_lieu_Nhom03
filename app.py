import streamlit as st
import json
import os
import chromadb
from chromadb.utils import embedding_functions
import google.generativeai as genai

# ==========================================
# 1. CẤU HÌNH HỆ THỐNG
# ==========================================
st.set_page_config(page_title="Hỗ trợ Hộ chiếu VN", page_icon="🇻🇳", layout="wide")

if "GOOGLE_API_KEY" in st.secrets:
    api_key = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=api_key)
else:
    st.error("❌ Thiếu API Key trong Secrets!")
    st.stop()

# ==========================================
# 2. KHỞI TẠO DỮ LIỆU (RAG)
# ==========================================
@st.cache_resource
def init_db():
    if not os.path.exists("TAI_LIEU_RB.json"):
        return None
    
    client = chromadb.PersistentClient(path="chroma_db_data")
    emb_func = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="paraphrase-multilingual-MiniLM-L12-v2"
    )
    
    try:
        # Tạo collection mới để tránh xung đột dữ liệu cũ
        collection = client.get_or_create_collection(name="passport_final_fix", embedding_function=emb_func)
        
        if collection.count() == 0:
            with open("TAI_LIEU_RB.json", "r", encoding="utf-8") as f:
                data = json.load(f)
            
            documents = [item["content_text"] for item in data]
            metadatas = [{"title": item["title"], "url": item["url"]} for item in data]
            ids = [str(i) for i in range(len(data))]
            
            collection.add(ids=ids, documents=documents, metadatas=metadatas)
    except Exception as e:
        st.error(f"Lỗi DB: {e}")
        return None
    return collection

collection = init_db()

# ==========================================
# 3. XỬ LÝ AI
# ==========================================
def get_ai_response(user_query):
    if collection is None:
        return "Dữ liệu chưa sẵn sàng.", None, None

    results = collection.query(query_texts=[user_query], n_results=1)
    if not results["documents"][0]:
        return "Không tìm thấy thông tin phù hợp.", None, None

    context = results["documents"][0][0]
    meta = results["metadatas"][0][0]
    
    prompt = f"Ngữ cảnh: {context}\n\nCâu hỏi: {user_query}\nTrả lời ngắn gọn, chính xác bằng tiếng Việt."

    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        return response.text, meta['url'], meta['title']
    except Exception as e:
        return f"Lỗi AI: {str(e)}", None, None

# ==========================================
# 4. GIAO DIỆN CHAT
# ==========================================
st.title("🇻🇳 Trợ lý ảo Hộ chiếu")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Hỏi về thủ tục hộ chiếu...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Đang tìm kiếm nguồn tin chính thống..."):
            answer, url, title = get_ai_response(user_input)
            
            # Khắc phục lỗi SyntaxError bằng cách nối chuỗi an toàn
            if url:
                full_res = answer + "\n\n---\n**Nguồn trích dẫn:** " + title + "\n🔗 [Link Dịch vụ công](" + url + ")"
            else:
                full_res = answer
            
            st.markdown(full_res)
            st.session_state.messages.append({"role": "assistant", "content": full_res})
