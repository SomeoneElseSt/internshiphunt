"""
Internship Hunt - AI-powered internship search and application assistant.

This application helps candidates find relevant internships and generate 
application materials using AI-powered analysis and document generation.
"""

import re
import asyncio
import time
import streamlit as st

# Client imports
from clients.github_client import GithubTool

# Service imports
from services.resume_processor import ResumeProcessor

# Utility imports
from utils.file_readers import read_pdf, read_docx

# UI component imports
from ui.internship_display import display_internships


def process_responses(processor):
    """Process user responses and create candidate profile."""
    profile = {
        "resume_data": st.session_state['resume_data'],
        "responses": st.session_state.responses
    }

    processor.database[st.session_state['resume_data'].get('name', 'Anonymous')] = profile
    return profile


def main():
    """Main application entry point - orchestrates the Streamlit UI."""
    st.title("Internship Hunt w/Agno📚, Gemini 🧠, and AgentQL🦾")

    if 'initialized' not in st.session_state:
        st.session_state.update({
            'initialized': True,
            'form_submitted': False,
            'resume_processed': False,
            'responses': {},
            'questions': [],
            'resume_data': None,
            'processor': ResumeProcessor()
        })

    with st.form("resume_upload_form"):
        uploaded_file = st.file_uploader("Upload Resume", type=["pdf", "docx"])

        if st.form_submit_button("Analyze Resume") and uploaded_file:
            with st.spinner("Processing resume..."):
                file_content = read_pdf(uploaded_file) if uploaded_file.type == "application/pdf" else read_docx(uploaded_file)

                st.session_state.resume_data = st.session_state.processor.parse_resume(file_content)
                st.session_state.questions = st.session_state.processor.generate_questions(st.session_state.resume_data)
                st.session_state.resume_processed = True

    if st.session_state.resume_processed and not st.session_state.form_submitted:
        with st.form("responses_form"):
            st.subheader("Follow-up Questions")
            for i, question in enumerate(st.session_state.questions):
                key = f"q_{i}"
                if key not in st.session_state.responses:
                    st.session_state.responses[key] = ""
                st.session_state.responses[key] = st.text_area(
                    label=f"Q{i+1}: {question}",
                    value=st.session_state.responses[key],
                    key=key
                )

            if st.form_submit_button("Submit Responses"):
                with st.spinner("Processing your answers..."):
                    st.session_state.form_submitted = True
                    profile = process_responses(st.session_state.processor)
                    st.session_state.job_links = st.session_state.processor.enhanced_search(profile)
                st.rerun()

    if st.session_state.form_submitted:
        st.success("Responses saved successfully!")

        for i, question in enumerate(st.session_state.questions):
            st.text_area(
                label=f"Q{i+1}: {question}",
                value=st.session_state.responses.get(f"q_{i}", ""),
                disabled=True,
                key=f"readonly_{i}"
            )

        st.subheader("Recommended Internships")

        with st.spinner("Fetching and analyzing internship opportunities..."):
            github_tool = GithubTool()
            internship_data = asyncio.run(github_tool.fetch_internship_data())

        if internship_data["status"] == "success":
            profile = {
                "resume_data": st.session_state.resume_data,
                "responses": st.session_state.responses
            }



            with st.spinner("Analyzing internships for best matches. This may take a few minutes..."):


                # Delay to avoid Gemini rate limiting
                time.sleep(2)
                
                recommended_internships = github_tool.recommend_internships(
                    internship_data["internships"], 
                    profile
                )

            asyncio.run(github_tool.close_session())

            valid_internships = []
            for job in recommended_internships:
                if 'application_link' not in job or not job['application_link']:
                    continue
                
                application_url = job['application_link'].strip()

                if application_url.startswith('$'):
                    continue

                if 'href=' in application_url:
                    url_match = re.search(r'href=["\'](.*?)["\']', application_url)
                    if url_match:
                        application_url = url_match.group(1).strip()

                application_url = application_url.replace('https://https://', 'https://')
                application_url = application_url.replace('http://https://', 'https://')
                application_url = application_url.replace('https://http://', 'https://')

                if 'href=' in job.get('company', ''):
                    company_match = re.search(r'<strong>(.*?)</strong>', job['company'])
                    if company_match:
                        job['company'] = company_match.group(1)
                    else:
                        job['company'] = re.sub(r'<[^>]+>', '', job['company'])

                if not application_url.startswith(('http://', 'https://')):
                    application_url = 'https://' + application_url

                job['application_link'] = application_url
                valid_internships.append(job)

            print(f"Total recommended internships: {len(recommended_internships)}")
            print(f"Valid internships to display: {len(valid_internships)}")

            if valid_internships:
                display_internships(
                    valid_internships,
                    st.session_state.resume_data,
                    st.session_state.responses
                )


if __name__ == "__main__":
    main()
