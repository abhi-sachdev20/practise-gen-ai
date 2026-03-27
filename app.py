import streamlit as st
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer, util

st.title("📄 PDF Chatbot")

uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file:
    reader = PdfReader(uploaded_file)
    
    text = ""
    for page in reader.pages:
        text += page.extract_text()

    # Better chunking (fixed size)
    chunk_size = 500
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(chunks)

    query = st.text_input("Ask a question")

    if query:
        query_emb = model.encode(query)
        scores = util.cos_sim(query_emb, embeddings)

        top_idx = scores.argmax()
        answer = chunks[top_idx]

        st.subheader("Answer:")
        st.write(answer)