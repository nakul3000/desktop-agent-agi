# Entry point for the personalized agent app
import os
from agent_core import AgentCore
from linkup_client import LinkupClient
import memory
from email_handler import EmailHandler
from document_handler import DocumentHandler
from calendar_handler import CalendarHandler

# Load config and initialize components (placeholders)
memory.init_db()
linkup_client = LinkupClient(api_key=os.getenv("LINKUP_API_KEY"))
email_handler = EmailHandler(hf_token=os.getenv("HF_TOKEN"), auto_authenticate_gmail=False)
document_handler = DocumentHandler()
calendar_handler = CalendarHandler()

agent = AgentCore(memory, linkup_client, email_handler, document_handler, calendar_handler)

if __name__ == "__main__":
    # Placeholder for main loop or API server
    print("Agent is running.")
