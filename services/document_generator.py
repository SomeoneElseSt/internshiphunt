"""Application document generation services."""

from typing import Dict
from clients.gemini_client import model


def generate_cover_letter(profile: Dict, job: Dict) -> str:
    """Generate a professional cover letter using Gemini based on profile and job details."""
    prompt = f"""Generate a professional cover letter for this job:
    Company: {job.get('company', 'N/A')}
    Role: {job.get('role', 'N/A')}
    Location: {job.get('location', 'N/A')}

    Using this candidate's profile:
    Skills: {profile.get('resume_data', {}).get('skills', [])}
    Experiences: {profile.get('resume_data', {}).get('experiences', [])}
    Interests: {profile.get('resume_data', {}).get('interests', [])}

    Write a concise, professional cover letter highlighting relevant skills and experiences.
    Keep it under 400 words and follow standard cover letter format."""

    response = model.generate_content(prompt)
    return response.text


def generate_research_statement(profile: Dict, job: Dict) -> str:
    """Generate a research statement using Gemini based on profile and job details."""
    prompt = f"""Generate a research statement for this job application:
    Company: {job.get('company', 'N/A')}
    Role: {job.get('role', 'N/A')}
    Location: {job.get('location', 'N/A')}

    Using this candidate's profile:
    Skills: {profile.get('resume_data', {}).get('skills', [])}
    Experiences: {profile.get('resume_data', {}).get('experiences', [])}
    Interests: {profile.get('resume_data', {}).get('interests', [])}

    Write a concise, professional research statement highlighting relevant research experiences, skills, and interests that align with the job role and company.
    Focus on past research projects, methodologies, and outcomes, and how they make the candidate suitable for a research-oriented role.
    Keep it under 500 words and follow a standard research statement format."""

    response = model.generate_content(prompt)
    return response.text


def generate_why_us_statement(profile: Dict, job: Dict) -> str:
    """Generate a 'Why Us' statement using Gemini based on profile and job details."""
    prompt = f"""Generate a 'Why Us' statement for this job application:
    Company: {job.get('company', 'N/A')}
    Role: {job.get('role', 'N/A')}
    Location: {job.get('location', 'N/A')}

    Using this candidate's profile:
    Skills: {profile.get('resume_data', {}).get('skills', [])}
    Experiences: {profile.get('resume_data', {}).get('experiences', [])}
    Interests: {profile.get('resume_data', {}).get('interests', [])}
    Career Goals: {profile.get('responses', {}).get('q_4', 'N/A')} # Assuming Q4 is about career goals

    Write a compelling 'Why Us' statement explaining why the candidate is interested in working for this specific company and in this role.
    Highlight alignment between the candidate's career goals, skills, and the company's mission, values, and opportunities.
    Mention specific aspects of the company or role that are particularly appealing to the candidate, based on their profile and stated career interests.
    Keep it under 300 words and maintain a professional yet enthusiastic tone."""

    response = model.generate_content(prompt)
    return response.text

