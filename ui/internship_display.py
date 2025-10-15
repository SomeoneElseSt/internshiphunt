"""Internship display component for rendering job listings with requirements."""

import base64
import streamlit as st
from typing import List, Dict
from clients.agentql_client import analyze_application_link
from services.document_generator import (
    generate_cover_letter,
    generate_research_statement,
    generate_why_us_statement
)
from utils.document_writers import create_docx_cover_letter


def display_internships(valid_internships: List[Dict], resume_data: Dict, responses: Dict):
    """
    Display internship listings with requirement analysis and document generation.
    
    Args:
        valid_internships: List of validated internship opportunities
        resume_data: Parsed resume information
        responses: User's responses to follow-up questions
    """
    for job in valid_internships:
        application_url = job.get('application_link', '')
        if not (application_url and 
                application_url.startswith(('http://', 'https://')) and 
                not application_url.startswith('$')):
            continue

        # Create a container for this job
        container = st.container()

        with container:
            st.markdown(f"### {job['company']} - {job['role']}")
            st.markdown(f"📍 {job['location']}")
            
            with st.spinner("Analyzing application requirements..."):
                # Analyze requirements for this specific internship
                requirements, should_display = analyze_application_link(application_url)
                if not should_display:
                    continue

                job['requirements'] = requirements
        
                requires_cover_letter = requirements.get('requires_cover_letter', False)
                if requires_cover_letter:
                    st.info("📝 Cover Letter Required")
                    cover_letter_key = f"cover_letter_{job['company']}_{job['role']}"
                    if cover_letter_key not in st.session_state:
                        st.session_state[cover_letter_key] = ""
                else:
                    st.success("✅ No Cover Letter Required")

                # Display research statement requirement status
                requires_research_statement = requirements.get('requires_research_statement', False)
                if requires_research_statement:
                    st.info("🔬 Research Statement Required")
                    research_statement_key = f"research_statement_{job['company']}_{job['role']}"
                    if research_statement_key not in st.session_state:
                        st.session_state[research_statement_key] = ""
                else:
                    st.success("✅ No Research Statement Required")

                # Display "why us" statement requirement status
                requires_why_us_statement = requirements.get('requires_why_us_statement', False)
                if requires_why_us_statement:
                    st.info("🎯 'Why Us' Statement Required")
                    why_us_statement_key = f"why_us_statement_{job['company']}_{job['role']}"
                    if why_us_statement_key not in st.session_state:
                        st.session_state[why_us_statement_key] = ""
                else:
                    st.success("✅ No 'Why Us' Statement Required")

            # Display apply button
            st.markdown(f"<a href='{job['application_link']}' target='_blank'>🔗 Apply Now</a>", unsafe_allow_html=True)

            # Generate cover letter if requirements say it's needed
            if requirements.get('requires_cover_letter'):
                cover_letter_key = f"cover_letter_{job['company']}_{job['role']}"

                # Auto-generate cover letter if not already generated
                if cover_letter_key not in st.session_state or not st.session_state[cover_letter_key]:
                    with st.spinner("Generating your cover letter..."):
                        st.session_state[cover_letter_key] = generate_cover_letter(
                            {"resume_data": resume_data, "responses": responses}, 
                            job
                        )

                # Display cover letter
                st.text_area(
                    "Generated Cover Letter",
                    value=st.session_state[cover_letter_key],
                    height=200,
                    key=f"cl_display_{job['company']}_{job['role']}"
                )

                # Generate download links
                dl_col1_cl, _ = st.columns(2)

                # Store files in session state
                files_key_cl = f"files_cl_{job['company']}_{job['role']}"
                if files_key_cl not in st.session_state:
                    profile_data = {"resume_data": resume_data}
                    st.session_state[files_key_cl] = {
                        'docx': create_docx_cover_letter(
                            st.session_state[cover_letter_key],
                            profile_data
                        )
                    }

                with dl_col1_cl:
                    company_safe = "".join(c for c in job['company'] if c.isalnum())
                    role_safe = "".join(c for c in job['role'] if c.isalnum())
                    st.markdown(
                        f'<a href="data:application/vnd.openxmlformats-officedocument.wordprocessingml.document;base64,{base64.b64encode(st.session_state[files_key_cl]["docx"]).decode()}" download="cover_letter_{company_safe}_{role_safe}.docx" target="_blank">📝 Cover Letter (DOCX)</a>',
                        unsafe_allow_html=True
                    )

            # Generate research statement if required
            if requirements.get('requires_research_statement'):
                research_statement_key = f"research_statement_{job['company']}_{job['role']}"

                # Auto-generate research statement if not already generated
                if research_statement_key not in st.session_state or not st.session_state[research_statement_key]:
                    with st.spinner("Generating your research statement..."):
                        st.session_state[research_statement_key] = generate_research_statement(
                            {"resume_data": resume_data, "responses": responses},
                            job
                        )

                # Display research statement
                st.text_area(
                    "Generated Research Statement",
                    value=st.session_state[research_statement_key],
                    height=200,
                    key=f"rs_display_{job['company']}_{job['role']}"
                )

                # Generate download links for research statement
                dl_col1_rs, _ = st.columns(2)
                files_key_rs = f"files_rs_{job['company']}_{job['role']}"
                if files_key_rs not in st.session_state:
                    profile_data = {"resume_data": resume_data}
                    st.session_state[files_key_rs] = {
                        'docx': create_docx_cover_letter(
                            st.session_state[research_statement_key],
                            profile_data
                        )
                    }

                with dl_col1_rs:
                    company_safe = "".join(c for c in job['company'] if c.isalnum())
                    role_safe = "".join(c for c in job['role'] if c.isalnum())
                    st.markdown(
                        f'<a href="data:application/vnd.openxmlformats-officedocument.wordprocessingml.document;base64,{base64.b64encode(st.session_state[files_key_rs]["docx"]).decode()}" download="research_statement_{company_safe}_{role_safe}.docx" target="_blank">📝 Research Statement (DOCX)</a>',
                        unsafe_allow_html=True
                    )

            # Generate "why us" statement if required
            if requirements.get('requires_why_us_statement'):
                why_us_statement_key = f"why_us_statement_{job['company']}_{job['role']}"

                # Auto-generate "why us" statement if not already generated
                if why_us_statement_key not in st.session_state or not st.session_state[why_us_statement_key]:
                    with st.spinner("Generating your 'Why Us' statement..."):
                        st.session_state[why_us_statement_key] = generate_why_us_statement(
                            {"resume_data": resume_data, "responses": responses},
                            job
                        )

                # Display "why us" statement
                st.text_area(
                    "Generated 'Why Us' Statement",
                    value=st.session_state[why_us_statement_key],
                    height=200,
                    key=f"wu_display_{job['company']}_{job['role']}"
                )

                # Generate download links for "why us" statement
                dl_col1_wu, _ = st.columns(2)
                files_key_wu = f"files_wu_{job['company']}_{job['role']}"
                if files_key_wu not in st.session_state:
                    profile_data = {"resume_data": resume_data}
                    st.session_state[files_key_wu] = {
                        'docx': create_docx_cover_letter(
                            st.session_state[why_us_statement_key],
                            profile_data
                        )
                    }

                with dl_col1_wu:
                    company_safe = "".join(c for c in job['company'] if c.isalnum())
                    role_safe = "".join(c for c in job['role'] if c.isalnum())
                    st.markdown(
                        f'<a href="data:application/vnd.openxmlformats-officedocument.wordprocessingml.document;base64,{base64.b64encode(st.session_state[files_key_wu]["docx"]).decode()}" download="why_us_statement_{company_safe}_{role_safe}.docx" target="_blank">📝 Why Us Statement (DOCX)</a>',
                        unsafe_allow_html=True
                    )
            
            st.divider()

