# To run this app: uvicorn main:app --reload
# Dependencies: pip install fastapi uvicorn pypdf ollama python-dotenv python-multipart

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from pypdf import PdfReader
from fastapi.responses import HTMLResponse
import ollama
from dotenv import load_dotenv
import json
import io
import os

load_dotenv()

app = FastAPI(docs_url=None, redoc_url=None)

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("index.html") as f:
        return HTMLResponse(content=f.read(), status_code=200)

def get_ollama_response(input_text):
    try:
        response = ollama.chat(model='llama3.2', messages=[{'role': 'user', 'content': input_text}])
        return response['message']['content']
    except Exception as e:
        print(f"Error calling Ollama: {e}")
        raise HTTPException(status_code=500, detail="Error communicating with the Ollama model.")

def input_pdf_text(uploaded_file_stream):
    try:
        reader = PdfReader(uploaded_file_stream)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
        return text
    except Exception as e:
        print(f"Error reading PDF: {e}")
        raise HTTPException(status_code=400, detail="Error processing the PDF file.")

input_prompt = """
Hey Act Like a skilled or very experience ATS(Application Tracking System)
with a deep understanding of tech field,software engineering,data science ,data analyst
and big data engineer. Your task is to evaluate the resume based on the given job description.
You must consider the job market is very competitive and you should provide 
best assistance for improving thr resumes. Assign the percentage Matching based 
on Jd and
the missing keywords with high accuracy
resume:{text}
description:{jd}

I want the response in one single string having the structure. Do not output any other text.
{{"JD Match":"%","MissingKeywords:[]","Profile Summary":""}}
"""

@app.post("/analyze-resume/")
async def analyze_resume(
    jd: str = Form(...),
    resume: UploadFile = File(...)
):
    if not resume.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a PDF.")

    pdf_contents = await resume.read()
    resume_text = input_pdf_text(io.BytesIO(pdf_contents))
    if not resume_text.strip():
        raise HTTPException(status_code=400, detail="Could not extract text from the provided resume PDF.")

    ollama_response_str = get_ollama_response(input_prompt.format(text=resume_text, jd=jd))

    try:
        return json.loads(ollama_response_str)
    except json.JSONDecodeError:
        return {"error": "Failed to parse the model's response as JSON.", "raw_response": ollama_response_str}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
