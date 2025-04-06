import openai
import os
openai.api_key = os.getenv("OPENAI_API_KEY")

def generate_hr_questions(job_desc, resume_content):
    prompt = f"""You are an HR interviewer.
Based on the following job description and candidate resume, generate 5 relevant and insightful HR interview questions.

Job Description:
{job_desc}

Resume:
{resume_content}

Questions:"""

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
        temperature=0.7
    )

    output = response.choices[0].message.content
    return output.strip().split("\n")
