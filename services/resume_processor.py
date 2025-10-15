"""Resume processing and question generation services."""

import json
from typing import Dict, List
from agno.agent import Agent
from agno.models.google import Gemini
from clients.gemini_client import GOOGLE_API_KEY
from clients.github_client import GithubTool
from services.job_search_manager import JobSearchManager


class ResumeProcessor:
    """Process resumes and generate follow-up questions for candidates."""
    
    def __init__(self):
        self.agent = Agent(
            model=Gemini(
                id="gemini-2.0-flash",
                api_key=GOOGLE_API_KEY
            ),
            description="You are an expert resume analyzer and career counselor.",
            markdown=True,
            tools=[GithubTool()]
        )
        self.database = {}
        self.github_tool = GithubTool()
        self.job_searcher = JobSearchManager()

    def parse_resume(self, file_content: str) -> Dict:
        """Extract structured information from resume text."""
        max_retries = 3
        for attempt in range(max_retries):
            prompt = f"Analyze this resume and extract key information including skills, experiences, and interests. This information will be provided to another artificial intelligence to generate follow up questions about the applicant, so you want to be very specific. Return the response as a JSON object with keys: skills, experiences, interests. In your JSON return do not include `````` at the end or start of the JSON. Only raw JSON, literally nothing else besides RAW JSON. You will create parsing issues, so only RAW JSON, no markdown code blocks whatsoever! Applicant Content: {file_content}"
            response = self.agent.run(prompt)
            print(f"Raw Gemini Response (Attempt {attempt + 1}):", response.content)

            if not response.content.strip():
                if attempt < max_retries - 1:
                    continue
                return {
                    "skills": [],
                    "experiences": [],
                    "interests": []
                }

            try:
                cleaned_content = response.content.replace("```json", "").replace("```", "").strip()
                return json.loads(cleaned_content)
            except json.JSONDecodeError as e:
                print(f"JSON Decode Error (Attempt {attempt + 1}):", e)
                if attempt < max_retries - 1:
                    continue
                return {
                    "skills": [],
                    "experiences": [],
                    "interests": []
                }

    def generate_questions(self, resume_data: Dict) -> List[str]:
        """Generate follow-up questions based on resume data."""
        prompt = f"""Based on this resume data: {json.dumps(resume_data)}
        Generate only 5 follow-up questions about:
        1. Role/Experience details about experiences that were in their resume. Specifically, use the STAR methodology to generate questions. You want to obtain information that you can later use to create cover letters for the applicant, so you want to be very specific. 
        2. Specific skills and their applications. You want to know specific times the applicant has applied their skills succesfully.
        3. Career interests and goals. This type of question is only to detail what type of ideal role the applicant projects themselves on or is working towards, both role-wise and skills-wise.  

        Consider that the applicant is looking for internships during the summer. These questions should ideally reflect some questions they may see interviewing or in application pages. Do not add Q1, Q2 at the beguinning of each question.  

        Return ONLY a JSON array of strings, formatted exactly like this: ["question1", "question2", "question3", "question4", "question5"]"""
        response = self.agent.run(prompt)
        print("Raw Questions Response:", response.content)
        try:
            cleaned_response = response.content.replace("```json", "").replace("```", "").strip()
            return json.loads(cleaned_response)
        except json.JSONDecodeError as e:
            print(f"JSON Decode Error: {e}")
            return [
                "Could you elaborate on your most recent role?",
                "What specific projects have you worked on using Python?",
                "How have you applied your data science skills in real projects?",
                "What are your career goals in the next 3-5 years?",
                "Which of your listed skills are you most eager to develop further?"
            ]


    def enhanced_search(self, profile: Dict) -> Dict[str, List[Dict]]:
        """Trigger job search based on candidate profile."""
        resume_data = profile.get('resume_data', {})
        responses = profile.get('responses', {})
        return self.job_searcher.compile_job_links(resume_data, responses)

