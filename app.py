import streamlit as st
from utils.pdf_parser import extract_text_from_pdf
from utils.rag_utils import (
    create_vector_store_per_resume,
    retrieve_relevant_chunks,
    rank_resumes_by_similarity
)
from utils.question_generator import generate_hr_questions_rag
import tempfile

st.set_page_config(page_title="AI HR Assistant", layout="wide")
st.title("🤖 AI-Powered HR Assistant")

api_key = st.secrets["api_key"]

st.sidebar.header("Step 1: Input Job Description")
jd_text_input = st.sidebar.text_area("Paste Job Description", height=300)

st.sidebar.header("Step 2: Upload Resumes")
resume_files = st.sidebar.file_uploader("Upload Resumes (PDF)", type=["pdf"], accept_multiple_files=True)

if st.sidebar.button("Analyze"):
    if jd_text_input and resume_files:
        with st.spinner("Processing resumes..."):
            resumes_texts = [extract_text_from_pdf(resume) for resume in resume_files]
            resume_names = [resume.name for resume in resume_files]

            # Vector DB Creation
            create_vector_store_per_resume(resumes_texts, resume_names, api_key)

            # Retrieve RAG chunks
            relevant_chunks = retrieve_relevant_chunks(jd_text_input, resume_names, api_key)

            # Ranking
            ranking = rank_resumes_by_similarity(jd_text_input, resumes_texts, resume_names, api_key)

        st.subheader("📊 Resume Ranking by Similarity")
        for rank, (name, score) in enumerate(ranking, 1):
            st.markdown(f"{rank}. **{name}** - Similarity Score: `{score:.2f}`")

        st.subheader("🎯 HR Interview Questions")
        for name in resume_names:
            st.markdown(f"**{name}**")
            questions = generate_hr_questions_rag(relevant_chunks[name], jd_text_input, api_key)
            for q in questions.strip().split("\n"):
                if q.strip().endswith("?"):
                    st.write(f"- {q.strip('-• ').strip()}")
    else:
        st.warning("Please enter job description and upload at least one resume.")
