"""
Client modules for external API integrations.

This package contains client implementations for all external services used
for Internship Hunt.

Modules:
    gemini_client: Google Gemini AI API configuration and model initialization.
        Exports: model, GOOGLE_API_KEY
        
    github_client: GitHub API integration for fetching and recommending internship listings.
        Exports: GithubTool (class)
        Methods: fetch_readme, close_session, parse_internship_data,
                 fetch_internship_data, recommend_internships
        
    agentql_client: AgentQL API for analyzing application link requirements.
        Determines if a cover letter, research statement, or why us statement is required.
        Exports: analyze_application_link (function)


Role:
    Provides an interface to external APIs, handling authentication,
    requests, and response parsing. Each client is responsible for its own
    API key management and error handling.
"""
