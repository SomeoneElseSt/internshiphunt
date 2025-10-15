"""AgentQL client for analyzing application link requirements."""

import os
import requests
import streamlit as st
from typing import Dict, Tuple
from dotenv import load_dotenv

load_dotenv()

AGENTQL_API_ENDPOINT = "https://api.agentql.com/v1/query-data"
DEFAULT_WAIT_TIME = 15
REQUEST_TIMEOUT = 25


def convert_to_bool(value):
    """Convert string or unknown values to boolean."""
    if value == 'True':
        return True
    if value == 'False':
        return False
    if value == 'null' or value is None:
        return False
    return bool(value)


def analyze_application_link(url: str) -> Tuple[Dict[str, bool], bool]:
    """
    Send application link to AgentQL for analysis of document requirements.
    
    Returns:
        Tuple of (requirements_dict, should_display_bool)
    """
    print(f"\n--- AgentQL Analysis for URL: {url} ---")

    api_key = os.environ.get("AGENT_QL_API_KEY")
    if not api_key:
        print("Warning: AGENT_QL_API_KEY not found in environment")
        return {"written_requirements": []}, True

    headers = {
        "x-api-key": api_key,
        "Content-Type": "application/json"
    }

    payload = {
        "url": url,
        # The query is the dict schema we are asking for from AgentQL, ie. what to extract
        "query": """
            {
              written_requirements {
                accepts_cover_letter(True/False)
                accepts_research_statement(True/False)
                accepts_why_us_statement(True/False)
              }
            }
        """,
        "params": {
            "mode": "standard",
            "wait_for": DEFAULT_WAIT_TIME,
            "is_scroll_to_bottom_enabled": True
        }
    }

    try:
        response = requests.post(
            AGENTQL_API_ENDPOINT,
            json=payload,
            headers=headers,
            timeout=REQUEST_TIMEOUT
        )

        print(f"===AgentQL Response Status: {response.status_code}===")

        if response.status_code == 401:
            print(f"Authentication failed. Response: {response.text}")
            st.error("AgentQL authentication failed. Please check your AGENT_QL_API_KEY.")
            return {"written_requirements": []}, True

        if response.status_code == 200:
            response_json = response.json()
            print("\n=== AgentQL Response Analysis ===")
            print(f"Full Response: {response_json}")

            written_reqs = response_json.get("data", {}).get("written_requirements", {})

            requires_cover_letter = convert_to_bool(written_reqs.get("accepts_cover_letter"))
            requires_research_statement = convert_to_bool(written_reqs.get("accepts_research_statement"))
            requires_why_us_statement = convert_to_bool(written_reqs.get("accepts_why_us_statement"))

            print(f"\nCover Letter Required: {requires_cover_letter}")
            print(f"Research Statement Required: {requires_research_statement}")
            print(f"Why Us Statement Required: {requires_why_us_statement}")

            result = {
                "requires_cover_letter": requires_cover_letter,
                "requires_research_statement": requires_research_statement,
                "requires_why_us_statement": requires_why_us_statement
            }

            print("=" * 30 + "\n")
            return result, True

        if 'speedyapply' in url and response.status_code == 404:
            return None, False

        st.warning(f"AgentQL request failed for {url}: {response.status_code}")
        return {"written_requirements": []}, True

    except Exception as e:
        st.error(f"Error analyzing application link {url}: {str(e)}")
        return {"written_requirements": []}, True
