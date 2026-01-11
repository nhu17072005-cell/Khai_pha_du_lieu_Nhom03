import streamlit as st
import json
import os
import chromadb
from chromadb.utils import embedding_functions
import google.generativeai as genai

# ==========================================
# 1. CẤU HÌNH HỆ THỐNG
# ==========================================
st.set_page_config(page_title="Hỗ trợ Hộ chiếu VN", page_icon="🇻🇳")

if "GOOGLE_API_KEY" in st.secrets:
    api_key = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=api_key)
else:
    st.error("❌ Thiếu API Key trong Secrets!")
    st.stop()

# ==========================================
# 2. XỬ LÝ DỮ LIỆU (RAG)
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
        collection = client.get_collection(name="passport_rag_v5", embedding_function=emb_func)
    except:
        collection = client.create_collection(name="passport_rag_v5", embedding_function=emb_func)
        with open("TAI_LIEU_RB.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        collection.add(
            ids=[str(i) for i in range(len(data))],
            documents=[item["content_text"] for item in data],
            metadatas=[{"title": item["title"], "url": item["url"]} for item in data]
        )
    return collection

collection = init_db()

# ==========================================
# 3. HÀM GỌI AI VÀ TRÍCH XUẤT URL
# ==========================================
def get_ai_response_with_url(user_query):
    # Tìm kiếm dữ liệu
    results = collection.query(query_texts=[user_query], n_results=1)
    
    if not results["documents"][0]:
        return "Không tìm thấy thông tin phù hợp.", None, None

    context = results["documents"][0][0]
    # Lấy URL từ metadata đã lưu trong Vector DB
    source_url = results["metadatas"][0][0].get("url", "https://dichvucong.gov.vn")
    source_title = results["metadatas"][0][0].get("title", "Cổng Dịch vụ công")

    prompt = f"""Bạn là trợ lý ảo hành chính công. 
Dựa vào ngữ cảnh: {context}
Hãy trả lời câu hỏi: {user_query}
Lưu ý: Chỉ trả lời phần nội dung chính, không lặp lại link URL vì tôi sẽ tự chèn phía dưới."""

    # Tự động tìm model khả dụng
    try:
        available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        model_name = "models/gemini-1.5-flash" if "models/gemini-1.5-flash" in available_models else available_models[0]
        
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        return response.text, source_url, source_title
    except Exception as e:
        return f"Lỗi: {str(e)}", None, None

# ==========================================
# 4. GIAO DIỆN
# ==========================================
st.title("🇻🇳 Trợ lý Hộ chiếu Việt Nam")

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
        with st.spinner("Đang tra cứu..."):
            answer, url, title = get_ai_response_with_url(user_input)
            
            if url:
                # Định dạng câu trả lời kèm nút bấm hoặc link rõ ràng
                full_response = f"{answer}\n\n---\n🔗 **Chi tiết thủ tục tại Cổng DVC:** [{title}]({url})"
            else:
                full_response = answer
                
            st.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
