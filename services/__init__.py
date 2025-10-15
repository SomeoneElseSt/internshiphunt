"""
Service modules for business logic and data processing.

This package contains the core business logic services for the app core functionalities.

Modules:
    resume_processor: Resume analysis and question generation service.
        Exports: ResumeProcessor (class)
        Methods: parse_resume, generate_questions, enhanced_search
        
    document_generator: AI-powered document generation for applications.
        Exports: generate_cover_letter, generate_research_statement, 
                 generate_why_us_statement (functions)
        
    job_search_manager: Job search operations and profile management.
        Exports: JobSearchManager (class)
        Methods: create_profile, compile_job_links

Role:
    Implements the core application logic by coordinating between clients
    and processing data. Services consume client APIs and provide high-level
    functionality to the UI layer.
"""
