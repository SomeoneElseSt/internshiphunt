"""File reading utilities for resume processing."""

import PyPDF2
import docx


def read_pdf(file):
    """Extract text content from a PDF file."""
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


def read_docx(file):
    """Extract text content from a DOCX file."""
    doc = docx.Document(file)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text
