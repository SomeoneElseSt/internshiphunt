"""
UI components for Streamlit interface.

This package contains modular Streamlit UI components that handle
specific sections of front-end.

Modules:
    internship_display: Internship listing and application display component.
        Exports: display_internships (function)
        Handles: Job cards, requirement analysis, document generation UI,
                 download links for cover letters and statements

Role:
    Separates complex UI rendering logic from main.py to improve readability
    and maintainability. Components are designed to be reusable and accept
    data as parameters rather than accessing session state directly.
"""
