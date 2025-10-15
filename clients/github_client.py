"""GitHub API client for fetching internship listings."""

import os
import re
import base64
import json
import asyncio
import time
import aiohttp
import streamlit as st
from typing import Dict, List, Any
from dotenv import load_dotenv
from agno.agent import Agent
from agno.models.google import Gemini
from agno.tools import Toolkit

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


class GithubTool(Toolkit):
    """Toolkit for fetching and analyzing internship data from GitHub repositories."""

    def __init__(self):
        super().__init__()
        self.name = "github"
        self.headers = {}
        self.session = None
        self.repo_links = [
            {
                "owner": "speedyapply",
                "repo": "2025-AI-College-Jobs",
                "branch": "main",
                "path": "README.md"
            },
            {
                "owner": "SimplifyJobs",
                "repo": "Summer2025-Internships",
                "branch": "main",
                "path": "README.md"
            }
        ]
        self.params = {
            "get_repository": True,
            "search_repositories": True,
            "list_repositories": True
        }
        self.agent = Agent(
            model=Gemini(
                id="gemini-2.0-pro",
                api_key=GOOGLE_API_KEY
            ),
            description="You are an expert at analyzing internship opportunities and matching them to candidate profiles.",
            markdown=True
        )

        github_key = os.getenv("GITHUB_KEY")
        if github_key:
            self.headers = {"Authorization": f"token {github_key}"}
        else:
            st.warning("No Github key found. Using unauthenticated access (rate limits may apply).")

    async def fetch_readme(self, repo_owner: str, repo_name: str, branch: str = "main", path: str = "README.md") -> Dict[str, Any]:
        """Fetch a README file from a Github repository."""
        if self.session is None:
            self.session = aiohttp.ClientSession()

        try:
            repo_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}"
            async with self.session.get(repo_url, headers=self.headers) as repo_response:
                if repo_response.status != 200:
                    print(f"Repository not found or inaccessible: {repo_url}")
                    return {"content": None, "status": "error", "message": "Repository not found"}

                content_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/contents/{path}?ref={branch}"
                async with self.session.get(content_url, headers=self.headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        if 'content' in data:
                            content = base64.b64decode(data['content']).decode('utf-8')
                            print(f"Successfully fetched content from {content_url}")
                            return {"content": content, "status": "success"}
                    print(f"Failed to fetch content: {response.status}")
                    return {"content": None, "status": "error", "message": f"Failed to fetch content: {response.status}"}
        except Exception as e:
            print(f"Error fetching from Github: {e}")
            return {"content": None, "status": "error", "message": str(e)}

    async def close_session(self):
        """Close the aiohttp session."""
        if self.session:
            await self.session.close()
            self.session = None

    def parse_internship_data(self, content: str) -> List[Dict[str, str]]:
        """Parse internship listings from markdown table content."""
        internships = []
        lines = content.split('\n')
        current_internship = {}

        for line in lines:
            if '|' in line:
                parts = [p.strip() for p in line.split('|')]
                if len(parts) >= 4:
                    if all(p.startswith('---') for p in parts if p):
                        continue
                    current_internship = {
                        "company": parts[1] if len(parts) > 1 else "",
                        "role": parts[2] if len(parts) > 2 else "",
                        "location": parts[3] if len(parts) > 3 else "",
                        "application": parts[4] if len(parts) > 4 else ""
                    }
                    if current_internship["company"] and current_internship["role"]:
                        internships.append(current_internship)

        return internships

    async def fetch_internship_data(self) -> Dict[str, Any]:
        """Fetch and process internship data from multiple sources in parallel."""

        async def fetch_and_parse_repo(repo):
            """Helper function to fetch and parse a single repository."""
            data = await self.fetch_readme(
                repo["owner"],
                repo["repo"],
                repo["branch"],
                repo["path"]
            )
            if data["status"] == "success":
                internships = self.parse_internship_data(data["content"])
                return [{**i, "source": repo["owner"]} for i in internships]
            return []

        repo_results = await asyncio.gather(
            *[fetch_and_parse_repo(repo) for repo in self.repo_links],
            return_exceptions=True
        )

        all_internships = []
        for result in repo_results:
            if isinstance(result, Exception):
                print(f"Error fetching repository: {result}")
                continue
            all_internships.extend(result)

        print("Fetched internships from GitHub:", all_internships)
        return {
            "status": "success", 
            "internships": all_internships,
            "total_count": len(all_internships)
        }

    def recommend_internships(self, internships: List[Dict], profile: Dict) -> List[Dict]:
        """Use custom Gemini agent to recommend internships based on profile with retry logic."""


        recommendation_agent = Agent(
            model=Gemini(
                id='gemini-2.5-pro',
                api_key=GOOGLE_API_KEY
            ),
            description="You are an expert at matching internship opportunities to candidate profiles, with deep understanding of tech industry requirements and career progression.",
            markdown=True
        )

        prompt = f"""Given this candidate profile:
        {json.dumps(profile)}

        And these internships:
        {json.dumps(internships)}

        Go through the available internships and return at maximum 10 (but any amount is okay) internships that are most relevant to the candidate's profile, based on what you know about them. 

        Avoid returning internships that are not relevant to the candidate's application range. For example, don't recommend master internships if you know explicitely they are not master students. 

        Return a JSON array containing only the selected internships, formatted exactly like this:
        [
            {{"company": "Company Name", "role": "Role Title", "location": "Location", "application_link"}}
        ] 

        Your response should only be JSON.
        """

        cleaned_response = ""
        try:
            response = recommendation_agent.run(prompt)
            print("Raw recommendation response:", response.content)
            cleaned_response = response.content.replace("```json", "").replace("```", "").strip()

            recommendations = json.loads(cleaned_response)

            processed_recommendations = []
            for job in recommendations:
                processed_job = {}

                if 'company' in job:
                    company_match = re.search(r'<strong>(.*?)</strong>', job['company'])
                    if company_match:
                        processed_job['company'] = company_match.group(1)
                    else:
                        processed_job['company'] = re.sub(r'<[^>]+>', '', job['company'])

                processed_job['role'] = job.get('role', '')
                processed_job['location'] = job.get('location', '')

                if 'application' in job or 'application_link' in job:
                    app_data = job.get('application') or job.get('application_link', '')

                    if not app_data or app_data == 'null':
                        continue

                    if app_data.startswith('$'):
                        processed_job['application_link'] = app_data
                    else:
                        url_match = re.search(r'href=["\'](.*?)["\']', app_data)
                        if url_match:
                            url = url_match.group(1)
                            url = url.replace('https://https://', 'https://')
                            url = url.replace('http://https://', 'https://')
                            url = url.replace('https://http://', 'https://')
                            if not url.startswith(('http://', 'https://')):
                                url = 'https://' + url
                            processed_job['application_link'] = url
                        else:
                            processed_job['application_link'] = app_data.strip()

                processed_recommendations.append(processed_job)

            return processed_recommendations[:10]
        except Exception as e:
            print(f"Error processing recommendations: {e}")
            print(f"Cleaned response that caused error: {cleaned_response}")
            return [internships[i] for i in range(min(10, len(internships)))]
