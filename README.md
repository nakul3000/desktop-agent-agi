# Pathfind AI

A local-first desktop agent that can search, summarize, and orchestrate tools -- built for the Hack with DC hackathon. It keeps a privacy-friendly memory (SQLite) of conversations, artifacts, and facts for better follow-ups.

---

## Getting Started

Follow these steps to get the app running on your computer.

### 1. Clone the repo

Open a terminal and run:

```bash
git clone https://github.com/your-username/desktop-agent-agi.git
cd desktop-agent-agi
```

### 2. Create a virtual environment

This keeps your dependencies isolated so nothing breaks on your system.

```bash
python3 -m venv venv
source venv/bin/activate
```

On Windows use:

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

The app needs two API keys to work. Create a `.env` file in the project folder:

```bash
cp .env.example .env
```

Then open `.env` and fill in your keys:

```
LINKUP_API_KEY=your-linkup-api-key-here
HF_TOKEN=your-huggingface-token-here
```

- **LINKUP_API_KEY** -- get one from [Linkup](https://www.linkup.so/)
- **HF_TOKEN** -- get one from [Hugging Face](https://huggingface.co/settings/tokens)

You can also add your resume path so the agent knows where to find it:

```
RESUME_PATH=/absolute/path/to/your/resume.pdf
```

### 5. Add your resume (optional)

If you want to tailor your own resume, drop your resume file into the `outputs/` folder using these exact names:

```
outputs/tailored_resume.docx
outputs/tailored_resume_ats.pdf
```

Put both a `.docx` and a `.pdf` version with the same naming. The Resume Tailor Agent will read from here and write the tailored version back to the same spot.

### 6. Run the app

```bash
streamlit run streamlit_app.py
```

The app will open in your browser automatically (usually at `http://localhost:8501`). That's it!

---

## Project Overview

### Memory

- Every conversation turn, artifact, and fact is saved to a local SQLite database (`memory.db`).
- Data is scoped by session so different runs don't collide.
- Call `memory.init_db()` once to create the tables. The Streamlit app does this for you.

### Agents

The app is powered by three specialized agents that work together:

- **Role Search Agent** -- Searches for open job postings across the web using Linkup. You give it job titles, location, and keywords and it returns a deduplicated list of matching roles with company, URL, and requirements.

- **Job Description Agent** -- Takes a selected job posting and fetches the full, clean job description text (responsibilities, requirements, skills). It can pull directly from a URL or search for the listing. Returns up to 3 variants so you can pick the best one.

- **Resume Tailor Agent** -- The core of the app. Give it your resume (PDF) and a job description (text or URL) and it will tailor your resume to match the role. It extracts keywords from the JD, rewrites your bullet points to highlight relevant experience, and exports the result as both DOCX and ATS-friendly PDF.

### Email / Recruiter Outreach

- Finds recruiter contacts (name, email, LinkedIn) for a company and role using Linkup.
- Filters out junk emails and prefers same-company contacts.
- Can draft personalized outreach emails using an LLM.
- Optional Gmail integration for reading/summarizing threads (requires Google OAuth credentials).

### Key Files

| File | What it does |
|---|---|
| `streamlit_app.py` | Chat UI (Streamlit) |
| `app.py` | Core agent logic |
| `agents/role_search_agent.py` | Searches for open job postings |
| `agents/job_description_agent.py` | Fetches and cleans job descriptions |
| `agents/resume_tailor_agent.py` | Tailors your resume to a job description |
| `memory.py` | SQLite memory layer |
| `linkup_client.py` | Linkup API wrapper |
| `email_handler.py` | Recruiter lookup + email drafting |
| `requirements.txt` | Python dependencies |


### Architecture Diagram
<img width="683" height="547" alt="image" src="https://github.com/user-attachments/assets/e7e76054-61bc-4706-af24-0bd5f916f44d" />
