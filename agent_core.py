# Core agent logic for multi-turn conversation and tool orchestration

class AgentCore:
    def __init__(self, memory, linkup_client, email_handler, document_handler, calendar_handler):
        self.memory = memory
        self.linkup_client = linkup_client
        self.email_handler = email_handler
        self.document_handler = document_handler
        self.calendar_handler = calendar_handler

    def handle_message(self, message, user_context=None):
        # Main workflow: decide which tool/domain to use, call Linkup if needed, update memory
        # Minimal routing so EmailHandler integrates with the broader app scaffolding.
        # Accepts dict-style messages for structured orchestration.
        if not isinstance(message, dict):
            return {"status": "unsupported", "error": "Message should be a dict payload."}

        intent = message.get("intent")

        if intent == "job_outreach":
            required = ["role_title", "job_description", "resume_text", "company"]
            missing = [k for k in required if not message.get(k)]
            if missing:
                return {"status": "error", "error": f"Missing required fields: {', '.join(missing)}"}

            payload = self.email_handler.build_recruiter_outreach_package(
                role_title=message["role_title"],
                job_description=message["job_description"],
                resume_text=message["resume_text"],
                company=message["company"],
                preferred_tone=message.get("preferred_tone", "formal"),
                candidate_name=message.get("candidate_name", ""),
                candidate_location=message.get("candidate_location", ""),
                candidate_linkedin=message.get("candidate_linkedin", ""),
                candidate_portfolio=message.get("candidate_portfolio", ""),
                additional_context=message.get("additional_context", ""),
            )
            return {"status": "ok", "intent": intent, "data": payload}

        if intent == "email_summarize":
            message_id = message.get("message_id")
            if not message_id:
                return {"status": "error", "error": "Missing required field: message_id"}
            if not self.email_handler.connect_gmail():
                return {"status": "error", "error": "Gmail authentication failed"}
            email_data = self.email_handler.get_email_details(message_id)
            summary = self.email_handler.summarize_email(email_data)
            extracted = self.email_handler.extract_info(email_data)
            return {
                "status": "ok",
                "intent": intent,
                "data": {"email": email_data, "summary": summary, "extraction": extracted},
            }

        return {"status": "unsupported", "error": f"Unknown intent: {intent}"}
