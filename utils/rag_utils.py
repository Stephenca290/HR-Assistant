
import sys
import pysqlite3
sys.modules["sqlite3"] = sys.modules["pysqlite3"]




import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI
import os

openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.getenv("OPENAI_API_KEY"),
    model_name="text-embedding-ada-002"
)


def create_vector_store(docs, names):
    client = chromadb.Client()
    collection = client.get_or_create_collection(name="resumes", embedding_function=openai_ef)
    for i, doc in enumerate(docs):
        collection.add(documents=[doc], ids=[str(i)], metadatas=[{"name": names[i]}])
    return collection

def query_resume_similarity(vectordb, job_desc, top_k=5):
    results = vectordb.query(query_texts=[job_desc], n_results=top_k)
    ranked = []
    for i in range(len(results["documents"][0])):
        content = results["documents"][0][i]
        score = results["distances"][0][i]
        name = results["metadatas"][0][i]["name"]
        ranked.append((name, 1-score, content))  # higher is better
    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked
