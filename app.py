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
        # Sử dụng collection mới để làm sạch dữ liệu
        collection = client.get_or_create_collection(name="passport_stable_v1", embedding_function=emb_func)
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
# 3. HÀM TÌM MODEL KHẢ DỤNG (SỬA LỖI 404)
# ==========================================
def get_available_model():
    try:
        # Liệt kê các model mà Key này có quyền sử dụng
        models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        
        # Ưu tiên bản Flash 1.5 (ổn định và quota cao nhất)
        for m in models:
            if "1.5-flash" in m:
                return m
        # Nếu không thấy Flash, lấy bất kỳ model nào có sẵn (Pro, v.v.)
        return models[0] if models else "models/gemini-1.5-flash"
    except Exception:
        # Nếu lỗi list_models, trả về tên phổ biến nhất
        return "models/gemini-1.5-flash"

# ==========================================
# 4. XỬ LÝ AI
# ==========================================
def get_ai_response(user_query):
    if collection is None: return "Dữ liệu chưa sẵn sàng.", None, None, None

    # Lấy 1 đoạn thông tin liên quan nhất để tiết kiệm Token
    results = collection.query(query_texts=[user_query], n_results=1)
    if not results["documents"][0]:
        return "Không tìm thấy thông tin phù hợp.", None, None, None

    context = results["documents"][0][0]
    meta = results["metadatas"][0][0]
    
    prompt = f"Ngữ cảnh: {context}\n\nCâu hỏi: {user_query}\nTrả lời ngắn gọn, chính xác."

    try:
        model_name = get_available_model()
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        return response.text, meta['url'], meta['title'], model_name
    except Exception as e:
        return str(e), None, None, None

# ==========================================
# 5. GIAO DIỆN CHAT
# ==========================================
st.title("🇻🇳 Trợ lý ảo Hộ chiếu")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Nhập câu hỏi về hộ chiếu...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Đang kết nối..."):
            answer, url, title, m_used = get_ai_response(user_input)
            
            if "429" in answer:
                full_res = "⚠️ Hệ thống đang hết lượt dùng miễn phí. Vui lòng chờ 60 giây."
            elif url:
                full_res = f"{answer}\n\n---\n**Nguồn:** {title}\n🔗 [Link Dịch vụ công]({url})"
            else:
                full_res = answer
            
            st.markdown(full_res)
            if m_used: st.caption(f"Đã sử dụng model: {m_used}")
            st.session_state.messages.append({"role": "assistant", "content": full_res})
