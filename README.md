# Internship Hunt w/ Agno 📚, Gemini 🧠, and AgentQL 🦾

> 🏆 **2nd place — Best Use of Agno** at the MultiModal AI Agents Hackathon in San Francisco. [See the project on Devpost](https://devpost.com/software/internship-hunt-w-agno-gemini-and-agentql).

**Internship Hunt** is an AI-assisted internship search and application preparation tool, co-created by [Hammon Dutra](https://github.com/hammondutra) and [Stiven Triana](https://github.com/SomeoneElseSt).

## The idea 💡

As college freshmen, we found it challenging to translate academic experience into compelling internship applications. We built Internship Hunt to help students identify relevant roles and prepare personalized application materials using AI.

## What the app does 💻

1. **Builds a candidate profile.** The user uploads a PDF or DOCX résumé. A Gemini-powered Agno agent extracts skills, experience, and interests, then generates five follow-up questions to fill in the details.
2. **Finds potential matches.** The app reads internship listings from two configured GitHub repositories and uses a Gemini-powered agent to recommend up to ten relevant opportunities based on the candidate profile.
3. **Checks application pages.** AgentQL analyzes application links for cover-letter, research-statement, and "Why Us" fields. The UI displays the detected document requirements.
4. **Drafts application materials.** When those fields are detected, Gemini generates tailored drafts and the user can download them as DOCX files.

The app assists with researching roles and preparing documents; **it does not automatically submit applications**.

### Internship sources

The current implementation fetches README-based listings from these two repositories, configured in `clients/github_client.py`:

- [speedyapply/2025-AI-College-Jobs](https://github.com/speedyapply/2025-AI-College-Jobs)
- [SimplifyJobs/Summer2025-Internships](https://github.com/SimplifyJobs/Summer2025-Internships)

**Current limitation:** Source repositories are hardcoded rather than user-selectable. They reference the **2025** internship cycle, so listings may be outdated or unavailable. The code does not currently accept arbitrary repository URLs through the UI.

## Tech stack ⚙️

- **Python + [Streamlit](https://streamlit.io):** Interactive web interface and user session state.
- **[Agno](https://docs.agno.com) + [Google Gemini](https://ai.google.dev/):** Résumé analysis, follow-up questions, internship recommendations, and document generation.
- **[GitHub API](https://docs.github.com/en/rest) + aiohttp:** Fetching internship listings from repository README files.
- **[AgentQL](https://www.agentql.com):** Inspecting application pages for written-document fields.
- **python-docx and PyPDF2:** Reading uploaded documents and exporting generated materials.

[Replit](https://replit.com) was also used during the hackathon development process.

## Run locally 🚀

1. Clone the repository and install its dependencies:

   ```bash
   git clone https://github.com/SomeoneElseSt/internshiphunt.git
   cd internshiphunt
   python -m pip install -r requirements.txt
   ```

2. Create your local environment file:

   ```bash
   cp env.example .env
   ```

3. Edit `.env` and configure:

   | Variable | Purpose |
   | --- | --- |
   | `GOOGLE_API_KEY` | Required for Gemini-powered features |
   | `AGENT_QL_API_KEY` | Required for AgentQL application-page checks |
   | `GITHUB_KEY` | Optional; raises GitHub API rate limits |

   Never commit your actual API keys. The `.env` file is excluded by `.gitignore`.

4. Start the app:

   ```bash
   streamlit run main.py
   ```

5. Upload a PDF/DOCX résumé, answer the follow-up questions, and review the suggested internships and generated documents.

External services require working API credentials and network access. Output quality and availability depend on those services and on the source listings.

## Project structure 🗂️

```text
main.py                    # Streamlit application flow
clients/
  agentql_client.py        # Application-page analysis
  gemini_client.py         # Gemini model configuration
  github_client.py         # GitHub sources and internship matching
services/
  resume_processor.py      # Résumé parsing and follow-up questions
  document_generator.py    # Application-document drafts
  job_search_manager.py    # Profile/job-search helpers (partly placeholder)
ui/
  internship_display.py    # Listing cards and DOCX downloads
utils/
  file_readers.py          # PDF and DOCX ingestion
  document_writers.py      # DOCX export
env.example                # Environment variable template
```

## Roadmap 🎅

- [x] Deploy a Streamlit version of the prototype.
- [x] Refactor the original monolithic script into clients, services, UI, and utilities.
- [ ] Let users provide their own Gemini and AgentQL API keys through the app.
- [ ] Let users select the GitHub internship repositories they want to search.
- [ ] Update internship sources to current recruiting cycles.
- [ ] Improve reliability around external APIs and application-link parsing.

## Hackathon team 🤝

Built by **[Hammon Dutra](https://github.com/hammondutra)** and **[Stiven Triana](https://github.com/SomeoneElseSt)** at the MultiModal AI Agents Hackathon in San Francisco, where Internship Hunt received **2nd place for Best Use of Agno**. [Hackathon submission and award details on Devpost](https://devpost.com/software/internship-hunt-w-agno-gemini-and-agentql).
