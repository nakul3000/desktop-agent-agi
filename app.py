# Entry point for the personalized agent app
from agent_core import AgentCore
from linkup_client import LinkupClient
from memory import Memory
from email_handler import EmailHandler
from document_handler import DocumentHandler
from calendar_handler import CalendarHandler

# Load config and initialize components (placeholders)
linkup_client = LinkupClient(api_key='YOUR_LINKUP_API_KEY')
memory = Memory()
email_handler = EmailHandler()
document_handler = DocumentHandler()
calendar_handler = CalendarHandler()

agent = AgentCore(memory, linkup_client, email_handler, document_handler, calendar_handler)

if __name__ == "__main__":
    # Placeholder for main loop or API server
    print("Agent is running.")
