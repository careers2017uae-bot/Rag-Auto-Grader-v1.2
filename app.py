# app.py
"""
RAG-based Student Work Auto-Grader (Streamlit)
Enhanced: Rubric Upload / Paste (PDF, DOCX, TXT)
No JSON exposure to teachers
"""

import streamlit as st
import os
import tempfile
from typing import List, Dict
import docx2txt
from PyPDF2 import PdfReader
import re

# ----------------------------
# Utility functions
# ----------------------------
def parse_rubric_text(rubric_text: str) -> List[Dict]:
    """
    Parses rubric text into internal structure for grading.
    Expected format:
    Band | Task Response | Coherence & Cohesion | Lexical Resource | Grammar
    """
    lines = [line.strip() for line in rubric_text.splitlines() if line.strip()]
    parsed_rubric = []
    
    for line in lines:
        if re.match(r"^\d+", line):  # Line starts with band number
            parts = [part.strip() for part in re.split(r"\|", line)]
            if len(parts) >= 5:  # Ensure minimum 5 columns
                parsed_rubric.append({
                    "band": int(parts[0]),
                    "task_response": parts[1],
                    "cohesion": parts[2],
                    "lexical": parts[3],
                    "grammar": parts[4]
                })
    return parsed_rubric

def read_uploaded_file(uploaded_file) -> str:
    """
    Reads PDF, DOCX, or TXT content as string
    """
    if uploaded_file.type == "application/pdf":
        reader = PdfReader(uploaded_file)
        text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
        return text
    elif uploaded_file.type in ["application/vnd.openxmlformats-officedocument.wordprocessingml.document", "application/msword"]:
        return docx2txt.process(uploaded_file)
    elif uploaded_file.type == "text/plain":
        return uploaded_file.getvalue().decode("utf-8")
    else:
        st.error("Unsupported file type. Please upload PDF, DOCX, or TXT.")
        return ""

# ----------------------------
# Streamlit App
# ----------------------------
st.title("📝 RAG Auto-Grader with Human-Readable Rubric")

# Step 1: Upload or paste rubric
st.header("Step 1: Provide Rubric (No JSON needed)")

rubric_input_method = st.radio(
    "How would you like to provide the rubric?",
    ("Upload File (PDF/DOCX/TXT)", "Paste Rubric Text")
)

rubric_text = ""
if rubric_input_method == "Upload File (PDF/DOCX/TXT)":
    uploaded_rubric = st.file_uploader("Upload rubric file", type=["pdf", "docx", "txt"])
    if uploaded_rubric is not None:
        rubric_text = read_uploaded_file(uploaded_rubric)
elif rubric_input_method == "Paste Rubric Text":
    rubric_text = st.text_area("Paste rubric here (grid/table format with `|` separator)", height=300)

# Step 2: Parse rubric
parsed_rubric = []
if rubric_text:
    parsed_rubric = parse_rubric_text(rubric_text)
    if parsed_rubric:
        st.success("Rubric parsed successfully! ✅")
        st.table(parsed_rubric)
    else:
        st.warning("Could not parse rubric. Please ensure it follows the format: Band | Task Response | Coherence & Cohesion | Lexical Resource | Grammar")

# ----------------------------
# Step 3: Student answer input
# ----------------------------
st.header("Step 2: Student Submission")
student_answer = st.text_area("Paste student's answer here", height=200)

# ----------------------------
# Step 4: Auto-Grading (existing workflow)
# ----------------------------
if st.button("Grade Answer"):
    if not parsed_rubric:
        st.error("Please provide a valid rubric first.")
    elif not student_answer.strip():
        st.error("Please provide a student's answer to grade.")
    else:
        # Placeholder for your existing RAG grading logic
        # Example: Compare student_answer with model_answer and compute semantic similarity
        # Here we just mock a grade for demonstration
        import random
        assigned_band = random.choice([band["band"] for band in parsed_rubric])
        st.success(f"✅ Assigned Band: {assigned_band}")
        st.info("Full grading details would be displayed here according to the parsed rubric.")
