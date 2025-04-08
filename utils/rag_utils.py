import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

def create_vector_store_per_resume(resumes_texts, resume_names, api_key):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    
    for name, text in zip(resume_names, resumes_texts):
        chunks = splitter.split_text(text)
        db = FAISS.from_texts(chunks, embedding=embeddings)
        db.save_local(f"faiss_{name}")

def retrieve_relevant_chunks(jd_text, resume_names, api_key, k=5):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    relevant_chunks = {}
    for name in resume_names:
        db = FAISS.load_local(f"faiss_{name}", embeddings, allow_dangerous_deserialization=True)
        results = db.similarity_search(jd_text, k=k)
        relevant_chunks[name] = "\n".join([res.page_content for res in results])
    return relevant_chunks

def cosine_similarity(vec1, vec2):
    vec1, vec2 = np.array(vec1), np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def rank_resumes_by_similarity(jd_text, resumes_texts, resume_names, api_key):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    jd_embedding = embeddings.embed_query(jd_text)

    scores = []
    for name, resume_text in zip(resume_names, resumes_texts):
        resume_embedding = embeddings.embed_query(resume_text)
        similarity = cosine_similarity(jd_embedding, resume_embedding)
        scores.append((name, similarity))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores
