import streamlit as st
import os
from utils.pdf_utils import extract_text_from_pdf
from utils.rag_utils import create_vector_store, query_resume_similarity
from utils.question_generator import generate_hr_questions

st.set_page_config(page_title="AI HR Assistant", layout="wide")

st.title("🤖 AI HR Assistant")
st.markdown("Upload resumes (PDF) and enter a job description to rank candidates and generate HR interview questions.")

uploaded_files = st.file_uploader("Upload Resume PDFs", type=["pdf"], accept_multiple_files=True)
job_description = st.text_area("Paste Job Description", height=200)

if st.button("Analyze"):
    with st.spinner("Processing resumes..."):
        resume_texts = [extract_text_from_pdf(file) for file in uploaded_files]
        resume_names = [file.name for file in uploaded_files]

        # Step 1: Create Chroma vector store
        vectordb = create_vector_store(resume_texts, resume_names)

        # Step 2: Query using job description
        ranked_resumes = query_resume_similarity(vectordb, job_description)

        # Step 3: Generate questions
        st.subheader("📊 Ranked Resumes & HR Questions")
        for i, (name, score, content) in enumerate(ranked_resumes):
            st.markdown(f"### {i+1}. {name} (Score: {score:.2f})")
            questions = generate_hr_questions(job_description, content)
            st.markdown("**Suggested Questions:**")
            for q in questions:
                st.markdown(f"- {q}")
