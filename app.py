import streamlit as st
import faiss
import numpy as np
from openai import OpenAI

client = OpenAI(api_key="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")

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer, util

def generate_answer(context,query,chat_history):
    messages = [{"role" : "system", "content":"Answer based only on provided context. "}]
    messages += chat_history

    messages.append({
        "role" : "user",
        "content" : f"Context: {context} \n\nQuestion: {query}"
    })

    response = client.chat.completions.create(
        model = "gpt-4o-mini",
        messages = messages

    )
    return response.choices[0].message.content

st.title("PDF chatbot")

uploaded_files = st.file_uploader("Upload PDFs ", type = "pdf", accept_multiple_files=True)

all_text = ""
for uploaded_file in uploaded_files:
    reader = PdfReader(uploaded_file)

    

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            all_text+= page_text

    def chunk_text(text, chunk_size=500, overlap=50):
        chunks = []
        metadata = []

        for file in uploaded_files:
            reader = PdfReader(file)

        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()

            if text:
                page_chunks = chunk_text(text) 

                for chunk in page_chunks:
                    chunks.append(chunk)
                    metadata.append({
                            "page": page_num+1,
                            "source": file.name

                    })       

        for i in range(0, len(text), chunk_size-overlap):
            chunk = text[i:i +chunk_size]
            chunks.append(chunk)

        return chunks

    chunks = chunk_text(all_text)        

    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(chunks)
    embeddings = np.array(embeddings).astype("float32")

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    chat_history = []
    

    while True:
        query = input("\nAsk a question: (type 'exit' to quit):")

        if query:
            if query.lower() == "exit":
                break

        chat_history.append({"role":"user","content":query})

        query_emb = model.encode(query).astype("float32")

        D,I = index.search(query_emb, k=3)
        context = " ".join([chunks[i] for i in I[0]])

        answer = generate_answer(context, query, chat_history)

        chat_history.append({"role":"assistant","content":answer})

        st.write("### Answer:")
        st.write(answer)


        

        scores = util.cos_sim(query_emb,embeddings)

        if scores.max() < 0.3:
            print("I could not find relevant information.")
            continue

        best_idx = scores.argmax()


        top_indices = scores.argsort(descending = True)[0][:2]


        answer = " ".join([chunks[i] for i in top_indices])


        # print("\n Answer:")

        # print(answer)

