# job_searcher.py — Job Search Tool Execution
# Handles job search functionality for the agent

import json
import os
from dotenv import load_dotenv
from linkup import LinkupClient

load_dotenv()


class JobSearcher:
    """Manages job search operations using Linkup API."""
    
    def __init__(self):
        """Initialize job searcher with Linkup API key from .env."""
        self.api_key = os.getenv("LINKUP_API_KEY")
        if not self.api_key:
            raise ValueError("LINKUP_API_KEY not found in .env")
        self.client = LinkupClient(api_key=self.api_key)
    
    def execute_search(self, params: dict) -> str:
        """Execute job search with structured output schema."""
        role = params.get("role", "Machine Learning Engineer")
        company = params.get("company", "")
        location = params.get("location", "United States")
        
        try:
            # Build the search query
            query = f"Search for {role} job listings on official company career pages and job board platforms (NOT LinkedIn posts or feed). Find positions at major tech companies in the {location} posted within the last 7 days."
            if company:
                query += f" Prioritize {company} positions."
            
            query += """

CRITICAL RULES:
- applicationUrl MUST be a direct link to an actual job posting page where someone clicks "Apply"
- ONLY return URLs from: company career sites, boards.greenhouse.io, jobs.lever.co, myworkdayjobs.com, icims.com, or linkedin.com/jobs/view/XXXXXXXXX
- NEVER return: linkedin.com/posts/, linkedin.com/feed/, lnkd.in/ shortened links, or mailto: links
- Each result must be a real, individual job posting (not a search results page)
- Verify each job is Machine Learning related: ML Engineer, Applied ML, ML Scientist, AI Engineer, Deep Learning Engineer"""
            
            # Define structured output schema
            schema = {
                "type": "object",
                "properties": {
                    "jobs": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "jobTitle": {
                                    "type": "string",
                                    "description": "Exact job title as posted"
                                },
                                "companyName": {
                                    "type": "string",
                                    "description": "Hiring company name"
                                },
                                "location": {
                                    "type": "string",
                                    "description": "City, State or Remote"
                                },
                                "postingDate": {
                                    "type": "string",
                                    "description": "Date posted in YYYY-MM-DD format"
                                },
                                "applicationUrl": {
                                    "type": "string",
                                    "description": "Direct apply URL. MUST be from: company careers page, greenhouse.io, lever.co, workday, icims, or linkedin.com/jobs/view/NUMBERS. NEVER linkedin.com/posts/ or lnkd.in/ links."
                                },
                                "jobDescriptionSummary": {
                                    "type": "string",
                                    "description": "Key requirements and responsibilities in 2-3 sentences"
                                },
                                "salaryRange": {
                                    "type": "string",
                                    "description": "Salary range if available, otherwise empty string"
                                },
                                "experienceLevel": {
                                    "type": "string",
                                    "description": "Entry, Mid, Senior, Lead, or Staff"
                                },
                                "source": {
                                    "type": "string",
                                    "description": "Where listing was found: company_careers, greenhouse, lever, workday, indeed, linkedin_jobs"
                                }
                            },
                            "required": ["jobTitle", "companyName", "location", "applicationUrl", "jobDescriptionSummary", "source"]
                        }
                    },
                    "totalJobsFound": {
                        "type": "integer"
                    },
                    "searchDate": {
                        "type": "string"
                    }
                },
                "required": ["jobs", "totalJobsFound", "searchDate"]
            }
            
            # Call Linkup API with structured output
            response = self.client.search(
                query=query,
                depth="standard",
                output_type="structured",
                include_images=False,
                structured_output_schema=json.dumps(schema),
                include_sources=False,
            )
            
            print(f"\n✅ Search completed for: {role}")
            if company:
                print(f"   Company: {company}")
            print(f"   Location: {location}\n")
            print(response)
            
            # Return structured response
            return json.dumps({
                "status": "success",
                "query": f"{role} in {location}" + (f" at {company}" if company else ""),
                "response": response,
                "next_steps": "I can research the company, tailor your resume, or draft a cover letter for any of these roles.",
            }, indent=2)
        
        except Exception as e:
            return json.dumps({
                "status": "error",
                "error": str(e),
                "message": "Failed to fetch job results. Please check your API key and try again.",
            }, indent=2)
