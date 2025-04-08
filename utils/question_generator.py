import google.generativeai as genai

def generate_hr_questions_rag(resume_chunk, job_description, api_key):
    genai.configure(api_key=api_key)

    prompt = f"""
You are an AI HR Assistant.

Generate 5 personalized HR interview questions based on the following resume content and job description.

Resume (Relevant Extracted Content):
{resume_chunk}

Job Description:
{job_description}

Only provide the questions as bullet points.
"""
    model = genai.GenerativeModel("gemini-2.0-flash-lite")
    response = model.generate_content(prompt)
    return response.text
