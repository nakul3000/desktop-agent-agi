import os
import os.path
import base64
import re
import json
import datetime
import requests
from typing import List, Dict, Any, Optional, Tuple
from email.message import EmailMessage
from email.utils import formatdate, make_msgid
from bs4 import BeautifulSoup
from dateutil import parser as date_parser

# Google API Client Imports
# pip install google-auth google-auth-oauthlib google-api-python-client
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

class EmailHandler:
    """
    A backend agent for handling emails using Google Gmail API and Hugging Face Llama 3.3.
    Responsibilities:
    1. Read & Parse Emails (clean HTML, handle threads)
    2. Summarize (bullets)
    3. Extract Info (deadlines, tasks, entities)
    4. Draft Replies (context-aware)
    """

    # Scopes required for reading and drafting emails
    SCOPES = [
        'https://www.googleapis.com/auth/gmail.readonly',
        'https://www.googleapis.com/auth/gmail.modify',
        'https://www.googleapis.com/auth/gmail.compose'
    ]

    # Hugging Face Model Endpoint
    HF_API_URL = "https://api-inference.huggingface.co/models/meta-llama/Llama-3.3-70B-Instruct"  # legacy
    HF_CHAT_URLS = [
        "https://router.huggingface.co/v1/chat/completions",
        "https://api-inference.huggingface.co/v1/chat/completions",
    ]
    HF_MODEL = "meta-llama/Llama-3.3-70B-Instruct"
    LINKUP_API_URL = "https://api.linkup.so/v1/search"

    def __init__(
        self,
        hf_token: Optional[str] = None,
        credentials_path: str = 'credentials.json',
        token_path: str = 'token.json',
        auto_authenticate_gmail: bool = False,
    ):
        self.hf_token = hf_token or os.getenv("HF_TOKEN", "")
        # Allow overriding model name (including provider suffix) via env var.
        # Examples:
        #   export HF_MODEL="meta-llama/Llama-3.3-70B-Instruct"
        #   export HF_MODEL="meta-llama/Llama-3.3-70B-Instruct:fireworks-ai"
        self.hf_model = os.getenv("HF_MODEL", self.HF_MODEL)
        self.linkup_api_key = os.getenv("LINKUP_API_KEY", "")
        self.creds = None
        self.service = None
        self.credentials_path = credentials_path
        self.token_path = token_path
        if auto_authenticate_gmail:
            self._authenticate_gmail(credentials_path, token_path)

    def connect_gmail(self) -> bool:
        """
        Explicitly authenticate Gmail when the orchestrator decides it's needed.
        Returns True when Gmail service is ready.
        """
        self._authenticate_gmail(self.credentials_path, self.token_path)
        return self.service is not None

    def _authenticate_gmail(self, creds_path: str, token_path: str):
        """
        Handles OAuth2 authentication with Gmail.
        Requires 'credentials.json' from Google Cloud Console.
        """
        if os.path.exists(token_path):
            self.creds = Credentials.from_authorized_user_file(token_path, self.SCOPES)
        
        # If there are no (valid) credentials available, let the user log in.
        if not self.creds or not self.creds.valid:
            if self.creds and self.creds.expired and self.creds.refresh_token:
                try:
                    self.creds.refresh(Request())
                except Exception as e:
                    print(f"Error refreshing token: {e}. Re-authenticating...")
                    self.creds = None

            if not self.creds:
                if not os.path.exists(creds_path):
                    print(f"WARNING: '{creds_path}' not found. Gmail features will not work.")
                    return
                
                flow = InstalledAppFlow.from_client_secrets_file(creds_path, self.SCOPES)
                # run_local_server tries to open a browser window for auth
                self.creds = flow.run_local_server(port=0)
            
            # Save the credentials for the next run
            with open(token_path, 'w') as token:
                token.write(self.creds.to_json())

        if self.creds:
            self.service = build('gmail', 'v1', credentials=self.creds)
            print("Gmail Service authenticated successfully.")

    def _call_llm(self, prompt: str, system_prompt: str = "You are a helpful AI assistant.") -> str:
        """
        Calls the Llama 3.3 70B model via Hugging Face.

        Prefers the newer chat-completions API (Inference Providers router).
        Falls back to legacy serverless inference endpoint if needed.
        """
        if not self.hf_token:
            return "Error: No Hugging Face Token provided."

        headers = {
            "Authorization": f"Bearer {self.hf_token}",
            "Content-Type": "application/json"
        }

        # 1) Try chat-completions (recommended in HF docs).
        chat_payload = {
            "model": self.hf_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.3,
            "max_tokens": 1024,
            "stream": False,
        }

        try:
            chat_failures: List[str] = []
            for url in self.HF_CHAT_URLS:
                try:
                    resp = requests.post(url, headers=headers, json=chat_payload, timeout=90)
                    if resp.status_code < 200 or resp.status_code >= 300:
                        chat_failures.append(
                            f"{url} -> {resp.status_code}: {self._truncate(resp.text.strip(), 400)}"
                        )
                        continue
                    out = resp.json()
                    # OpenAI-compatible response: choices[0].message.content
                    if isinstance(out, dict):
                        choices = out.get("choices") or []
                        if choices and isinstance(choices, list):
                            msg = choices[0].get("message") or {}
                            content = msg.get("content")
                            if isinstance(content, str):
                                return content.strip()
                    return str(out)
                except Exception as e:
                    chat_failures.append(f"{url} -> exception: {type(e).__name__}: {self._truncate(str(e), 200)}")

            # By default, do NOT fall back to legacy: it often returns 410 Gone and hides the real issue.
            # If you want legacy fallback, set: export HF_USE_LEGACY=1
            if os.getenv("HF_USE_LEGACY", "").strip() != "1":
                details = " | ".join(chat_failures) if chat_failures else "unknown chat failure"
                return (
                    "LLM Inference Error: chat-completions failed. "
                    "Common fixes: ensure HF token has 'Inference Providers' permission and set HF_MODEL (maybe with provider suffix).\n"
                    f"Details: {details}"
                )

            # 2) Optional fallback to legacy serverless inference API (deprecated for some models).
            formatted_prompt = (
                f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|>"
                f"<|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|>"
                f"<|start_header_id|>assistant<|end_header_id|>\n\n"
            )
            legacy_payload = {
                "inputs": formatted_prompt,
                "parameters": {"max_new_tokens": 1024, "temperature": 0.3, "return_full_text": False},
            }
            response = requests.post(self.HF_API_URL, headers=headers, json=legacy_payload, timeout=90)
            response.raise_for_status()
            output = response.json()
            if isinstance(output, list) and len(output) > 0:
                return output[0].get("generated_text", "").strip()
            if isinstance(output, dict) and "generated_text" in output:
                return output["generated_text"].strip()
            return str(output)
        except Exception as e:
            return f"LLM Inference Error: {e}"

    def _decode_b64_urlsafe(self, data: str) -> str:
        """
        Gmail message bodies are base64url encoded and may be missing padding.
        Returns decoded text (utf-8, replacement on errors).
        """
        if not data:
            return ""
        try:
            padded = data + "=" * (-len(data) % 4)
            return base64.urlsafe_b64decode(padded.encode("utf-8")).decode("utf-8", errors="replace")
        except Exception:
            return ""

    def _clean_html(self, html_content: str) -> str:
        """
        Converts HTML content to readable plain text using BeautifulSoup.
        """
        soup = BeautifulSoup(html_content, "html.parser")
        
        # Remove javascript and css
        for script in soup(["script", "style"]):
            script.extract()
            
        text = soup.get_text()
        
        # Collapse whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)
        
        return text

    def _strip_quoted_text(self, text: str) -> str:
        """
        Best-effort removal of quoted reply chains so we keep the "latest email" content.
        This is intentionally conservative to avoid deleting real content.
        """
        if not text:
            return ""

        # Common email reply separators
        separators = [
            r"^On .+ wrote:\s*$",
            r"^From:\s+.+$",
            r"^Sent:\s+.+$",
            r"^To:\s+.+$",
            r"^Subject:\s+.+$",
            r"^-----Original Message-----\s*$",
            r"^---+\s*Forwarded message\s*---+\s*$",
        ]
        sep_re = re.compile("|".join(separators), re.IGNORECASE | re.MULTILINE)

        # If we see a separator, keep only content above the first separator.
        m = sep_re.search(text)
        if m:
            text = text[: m.start()].rstrip()

        # Drop Gmail-style quoted lines starting with ">"
        lines = text.splitlines()
        kept: List[str] = []
        for line in lines:
            if line.lstrip().startswith(">"):
                continue
            kept.append(line)
        text = "\n".join(kept).strip()

        # Light cleanup of excessive blank lines
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        return text

    def _safe_json_from_llm(self, raw: str) -> Optional[Dict[str, Any]]:
        """
        Extracts a JSON object from an LLM response (handles fenced blocks and leading text).
        """
        if not raw:
            return None
        cleaned = raw.replace("```json", "").replace("```", "").strip()
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        candidate = cleaned[start : end + 1]
        try:
            return json.loads(candidate)
        except Exception:
            return None

    def _truncate(self, s: str, limit: int) -> str:
        if not s:
            return ""
        return s if len(s) <= limit else s[:limit]

    def _call_linkup_structured(self, *, query: str, schema: Dict[str, Any], depth: str = "deep") -> Dict[str, Any]:
        """
        Calls Linkup /v1/search using structured output.
        Returns parsed JSON dict when successful.
        """
        if not self.linkup_api_key:
            return {"error": "Missing LINKUP_API_KEY"}

        headers = {
            "Authorization": f"Bearer {self.linkup_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "q": query,
            "depth": depth,
            "outputType": "structured",
            "structuredOutputSchema": json.dumps(schema),
            "includeSources": False,
            "includeImages": False,
            "includeInlineCitations": False,
            "maxResults": 12,
        }
        try:
            resp = requests.post(self.LINKUP_API_URL, headers=headers, json=payload, timeout=60)
            resp.raise_for_status()
            out = resp.json()
            if isinstance(out, dict):
                return out
            return {"raw": out}
        except Exception as e:
            return {"error": str(e)}

    # Well-known company domain map — expanded for better coverage.
    _COMPANY_DOMAIN_MAP: Dict[str, str] = {
        "google": "google.com", "alphabet": "google.com",
        "meta": "meta.com", "facebook": "meta.com",
        "microsoft": "microsoft.com",
        "amazon": "amazon.com", "aws": "amazon.com",
        "apple": "apple.com",
        "netflix": "netflix.com",
        "nvidia": "nvidia.com",
        "openai": "openai.com",
        "anthropic": "anthropic.com",
        "stripe": "stripe.com",
        "airbnb": "airbnb.com",
        "uber": "uber.com",
        "lyft": "lyft.com",
        "salesforce": "salesforce.com",
        "adobe": "adobe.com",
        "oracle": "oracle.com",
        "ibm": "ibm.com",
        "intel": "intel.com",
        "tesla": "tesla.com",
        "spacex": "spacex.com",
        "palantir": "palantir.com",
        "databricks": "databricks.com",
        "snowflake": "snowflakecomputing.com",
        "coinbase": "coinbase.com",
        "robinhood": "robinhood.com",
        "doordash": "doordash.com",
        "instacart": "instacart.com",
        "figma": "figma.com",
        "notion": "makenotion.com",
        "slack": "slack.com",
        "zoom": "zoom.us",
        "twilio": "twilio.com",
        "datadog": "datadoghq.com",
        "cloudflare": "cloudflare.com",
        "confluent": "confluent.io",
        "hashicorp": "hashicorp.com",
        "elastic": "elastic.co",
        "mongodb": "mongodb.com",
        "reddit": "reddit.com",
        "pinterest": "pinterest.com",
        "snap": "snap.com", "snapchat": "snap.com",
        "twitter": "x.com", "x": "x.com",
        "linkedin": "linkedin.com",
        "spotify": "spotify.com",
        "shopify": "shopify.com",
        "squarespace": "squarespace.com",
        "tiktok": "tiktok.com", "bytedance": "bytedance.com",
        "samsung": "samsung.com",
        "jpmorgan": "jpmorgan.com", "jp morgan": "jpmorgan.com",
        "goldman sachs": "gs.com", "goldman": "gs.com",
        "deloitte": "deloitte.com",
        "mckinsey": "mckinsey.com",
    }

    _FREE_EMAIL_DOMAINS = frozenset({
        "gmail.com", "yahoo.com", "outlook.com", "hotmail.com",
        "proton.me", "protonmail.com", "aol.com", "icloud.com",
        "mail.com", "zoho.com", "yandex.com", "gmx.com",
    })

    @classmethod
    def _company_domain_hint(cls, company: str) -> str:
        name = (company or "").strip().lower()
        # Exact match first.
        if name in cls._COMPANY_DOMAIN_MAP:
            return cls._COMPANY_DOMAIN_MAP[name]
        # Partial match: check if any key is contained in the company name.
        for key, domain in cls._COMPANY_DOMAIN_MAP.items():
            if key in name:
                return domain
        return ""

    @classmethod
    def _infer_domain_from_company_name(cls, company: str) -> str:
        """
        Heuristic: when no known domain mapping exists, guess '<sanitized>.com'.
        Only used as a weak validation signal, never as the email itself.
        """
        hint = cls._company_domain_hint(company)
        if hint:
            return hint
        sanitized = re.sub(r"[^a-z0-9]", "", (company or "").strip().lower())
        if sanitized:
            return f"{sanitized}.com"
        return ""

    def _looks_like_corporate_email(self, email: str, company: str, *, discovered_domains: Optional[set] = None) -> bool:
        """
        Check whether an email looks like a corporate/work address for the given
        company. Uses the known domain map, any domains already discovered for this
        company during this search session, and a free-provider blocklist.
        """
        if not email or "@" not in email:
            return False
        domain = email.split("@", 1)[1].lower()

        # Known domain match.
        hint = self._company_domain_hint(company)
        if hint and domain.endswith(hint):
            return True

        # Match against domains already collected for this company in the same search.
        if discovered_domains and domain in discovered_domains:
            return True

        # Reject free-provider addresses.
        if domain in self._FREE_EMAIL_DOMAINS:
            return False

        # Heuristic: if the domain contains the company name (lowered, no spaces), accept.
        company_slug = re.sub(r"[^a-z0-9]", "", (company or "").strip().lower())
        if company_slug and company_slug in domain:
            return True

        # Unknown domain — accept tentatively (could be a recruiting agency).
        return True

    @staticmethod
    def _is_full_email(email: str) -> bool:
        """
        Accept only complete, non-obfuscated emails.
        Reject masked forms like d***@google.com or placeholder text.
        """
        if not email:
            return False
        e = email.strip()
        if "*" in e or "..." in e or "example.com" in e.lower():
            return False
        return re.fullmatch(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", e, flags=re.IGNORECASE) is not None

    @staticmethod
    def _is_placeholder_email(email: str) -> bool:
        """
        Reject obvious placeholder emails (e.g. z@anthropic.io, first.last@stripe.com,
        f.last@, first_last@) so they are not shown as valid contacts.
        """
        if not email or "@" not in email:
            return True
        local = email.strip().split("@", 1)[0].lower()
        if len(local) <= 1:
            return True
        # Single letter
        if re.match(r"^[a-z]$", local):
            return True
        # first.last, firstname.lastname, f.last, first_last, last.first, etc.
        if re.match(r"^first\.?last$", local) or re.match(r"^firstname\.?lastname$", local):
            return True
        if re.match(r"^f\.last$", local) or re.match(r"^first_last$", local):
            return True
        if re.match(r"^[a-z]\.last$", local):  # f.last, a.last, etc.
            return True
        if local in ("name", "email", "user", "contact", "recruiter", "hr"):
            return True
        return False

    @staticmethod
    def _is_relay_or_marketing_email(email: str) -> bool:
        """
        Reject reply relays, no-reply addresses, and known marketing/transactional
        domains (e.g. reply-xxx@reply.s12.y.mc.salesf...) so they are not shown as contacts.
        """
        if not email or "@" not in email:
            return True
        e = email.strip().lower()
        local, _, domain = e.partition("@")
        if not domain:
            return True
        # Local part: reply-*, noreply, no-reply, donotreply
        if local.startswith("reply-") or local.startswith("reply_"):
            return True
        if local in ("noreply", "no-reply", "donotreply", "do-not-reply", "mailer-daemon"):
            return True
        if ".html" in local or re.search(r"-\d{5,}-", local):  # reply-ff3013717660-321_HTML-...
            return True
        # Domain: reply.*, *.reply.*, known relay/marketing hosts
        if "reply." in domain or ".reply." in domain or domain.startswith("reply."):
            return True
        relay_domains = (
            "mc.salesf", "sendgrid", "mailchimp", "mandrill", "mailgun",
            "amazonses", "outbound.", "bounce.", "mail.", "smtp.",
        )
        for r in relay_domains:
            if r in domain:
                return True
        return False

    @staticmethod
    def _extract_all_emails_from_text(text: str) -> List[str]:
        """
        Pull all email-address-shaped strings from arbitrary text.
        Useful for scraping emails out of Linkup answer blobs.
        """
        if not text:
            return []
        return re.findall(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", text, flags=re.IGNORECASE)

    @staticmethod
    def _extract_linkedin_urls(text: str) -> List[str]:
        """Extract LinkedIn profile URLs from text."""
        if not text:
            return []
        return re.findall(r"https?://(?:www\.)?linkedin\.com/in/[A-Za-z0-9_-]+/?", text)

    @staticmethod
    def _normalize_name(name: str) -> str:
        """Normalize a person's name for dedup (lowercase, strip titles)."""
        n = (name or "").strip().lower()
        for prefix in ("mr.", "mrs.", "ms.", "dr.", "prof."):
            if n.startswith(prefix):
                n = n[len(prefix):].strip()
        return n

    # ==========================
    # 1. READ & PARSE EMAILS
    # ==========================
    
    def list_emails(self, max_results: int = 5, query: str = 'label:INBOX') -> List[Dict]:
        """Fetch a list of recent emails."""
        if not self.service:
            return []
        
        try:
            results = self.service.users().messages().list(userId='me', maxResults=max_results, q=query).execute()
            messages = results.get('messages', [])
            return messages
        except Exception as e:
            print(f"Failed to list emails: {e}")
            return []

    def get_email_details(
        self,
        message_id: str,
        *,
        include_quoted_text: bool = False,
        include_attachment_data: bool = False,
    ) -> Dict[str, Any]:
        """
        Fetches full email content, parses headers, and cleans body.
        Handles multipart messages to find text/plain or text/html.
        """
        if not self.service:
            return {}

        try:
            msg = self.service.users().messages().get(userId='me', id=message_id, format='full').execute()
            payload = msg.get('payload', {})
            headers = payload.get('headers', [])
            
            email_data = {
                "id": message_id,
                "threadId": msg.get('threadId'),
                "snippet": msg.get('snippet'),
                "subject": "No Subject",
                "sender": "Unknown",
                "date": "",
                "received_at": None,  # ISO string when parsed successfully
                "message_id_header": "",  # RFC 5322 Message-ID, if present
                "body": "",
                "body_full": "",
                "attachments": []
            }

            # Parse Headers
            for h in headers:
                name = h.get('name', '').lower()
                if name == 'subject':
                    email_data['subject'] = h.get('value')
                elif name == 'from':
                    email_data['sender'] = h.get('value')
                elif name == 'date':
                    email_data['date'] = h.get('value')
                    try:
                        dt = date_parser.parse(h.get("value"))
                        if dt:
                            email_data["received_at"] = dt.isoformat()
                    except Exception:
                        pass
                elif name == "message-id":
                    email_data["message_id_header"] = h.get("value") or ""

            # Recursive function to extract body parts
            def parse_parts(parts) -> Tuple[str, str]:
                text_content = ""
                html_content = ""
                
                for part in parts:
                    mime_type = part.get('mimeType')
                    filename = part.get('filename')
                    body = part.get('body', {})
                    data = body.get('data')
                    attachment_id = body.get("attachmentId")

                    if filename:
                        att: Dict[str, Any] = {
                            "filename": filename,
                            "mimeType": mime_type,
                            "size": body.get("size"),
                            "attachmentId": attachment_id,
                        }
                        if include_attachment_data and attachment_id:
                            try:
                                att_obj = (
                                    self.service.users()
                                    .messages()
                                    .attachments()
                                    .get(userId="me", messageId=message_id, id=attachment_id)
                                    .execute()
                                )
                                att_data = att_obj.get("data")
                                # Return base64url string to avoid huge binary blobs; caller can decode if needed.
                                att["data_b64url"] = att_data
                            except Exception as e:
                                att["error"] = str(e)
                        email_data["attachments"].append(att)
                    
                    if mime_type == 'text/plain' and data:
                        text_content += self._decode_b64_urlsafe(data)
                    elif mime_type == 'text/html' and data:
                        html_content += self._decode_b64_urlsafe(data)
                    elif 'parts' in part:
                        t, h = parse_parts(part['parts'])
                        text_content += t
                        html_content += h
                
                return text_content, html_content

            # Start parsing payload
            if 'parts' in payload:
                text_body, html_body = parse_parts(payload['parts'])
            else:
                # Single part message
                data = payload.get('body', {}).get('data')
                mime_type = payload.get('mimeType')
                text_body = ""
                html_body = ""
                if data:
                    decoded = self._decode_b64_urlsafe(data)
                    if mime_type == 'text/html':
                        html_body = decoded
                    else:
                        text_body = decoded

            # Prefer text/plain, fallback to cleaned HTML
            if text_body.strip():
                email_data["body_full"] = text_body.strip()
            elif html_body.strip():
                email_data["body_full"] = self._clean_html(html_body).strip()
            else:
                email_data["body_full"] = (email_data.get("snippet") or "").strip()  # Fallback

            if include_quoted_text:
                email_data["body"] = email_data["body_full"]
            else:
                email_data["body"] = self._strip_quoted_text(email_data["body_full"]) or email_data["body_full"]

            return email_data
        
        except Exception as e:
            print(f"Error reading email {message_id}: {e}")
            return {}

    def get_thread_details(
        self,
        thread_id: str,
        *,
        latest_only: bool = True,
        include_quoted_text: bool = False,
        include_attachment_data: bool = False,
    ) -> Dict[str, Any]:
        """
        Loads a Gmail thread and returns either the latest message or the full chain.

        Returns:
          {
            "threadId": str,
            "messages": [email_data...],   # chronological
            "latest": email_data | None,
            "combined_body": str           # helpful for summarization/extraction
          }
        """
        if not self.service:
            return {"threadId": thread_id, "messages": [], "latest": None, "combined_body": ""}

        try:
            thread = self.service.users().threads().get(userId="me", id=thread_id, format="full").execute()
            msgs = thread.get("messages", []) or []
            message_ids = [m.get("id") for m in msgs if m.get("id")]

            parsed: List[Dict[str, Any]] = []
            for mid in message_ids:
                parsed.append(
                    self.get_email_details(
                        mid,
                        include_quoted_text=include_quoted_text,
                        include_attachment_data=include_attachment_data,
                    )
                )

            # Sort by parsed received_at if present, otherwise keep Gmail order.
            def sort_key(m: Dict[str, Any]) -> Tuple[int, str]:
                iso = m.get("received_at")
                if not iso:
                    return (0, "")
                return (1, iso)

            parsed_sorted = sorted(parsed, key=sort_key)
            latest = parsed_sorted[-1] if parsed_sorted else None

            if latest_only and latest:
                combined_body = latest.get("body", "") or ""
                return {"threadId": thread_id, "messages": [latest], "latest": latest, "combined_body": combined_body}

            combined_parts: List[str] = []
            for m in parsed_sorted:
                combined_parts.append(
                    f"---\nFrom: {m.get('sender')}\nDate: {m.get('date')}\nSubject: {m.get('subject')}\n\n{m.get('body')}\n"
                )
            combined_body = "\n".join(combined_parts).strip()
            return {"threadId": thread_id, "messages": parsed_sorted, "latest": latest, "combined_body": combined_body}
        except Exception as e:
            print(f"Error reading thread {thread_id}: {e}")
            return {"threadId": thread_id, "messages": [], "latest": None, "combined_body": ""}

    # ==========================
    # 2. SUMMARIZE EMAILS
    # ==========================
    
    def summarize_email(self, email_data: Dict) -> str:
        """
        Uses LLM to produce 2-5 bullet points.
        """
        # Prefer latest-only body, but fall back to full.
        body_text = (email_data.get("body") or email_data.get("body_full") or "")[:6000]
        
        prompt = (
            f"Summarize the following email in 2-5 concise bullet points. "
            f"Focus on the most critical information (who, what, when, why).\n\n"
            f"From: {email_data.get('sender')}\n"
            f"Subject: {email_data.get('subject')}\n"
            f"Content:\n{body_text}\n\n"
            f"Output bullets only. Each bullet should start with '- '."
        )
        
        return self._call_llm(prompt, "You are a highly efficient executive assistant.")

    def summarize_thread(self, thread_data: Dict[str, Any]) -> str:
        """
        Summarizes either the latest message or the full chain (depending on thread_data shape).
        """
        combined = (thread_data.get("combined_body") or "")[:8000]
        latest = thread_data.get("latest") or {}
        prompt = (
            "Summarize the following email thread in 2-5 concise bullet points. "
            "Focus on key decisions, asks, deadlines, and next steps (not a rewrite).\n\n"
            f"Thread subject: {latest.get('subject','')}\n\n"
            f"Thread content:\n{combined}\n\n"
            "Output bullets only. Each bullet should start with '- '."
        )
        return self._call_llm(prompt, "You are a highly efficient executive assistant.")

    # ==========================
    # 3. EXTRACT ACTIONABLE INFO
    # ==========================
    
    def extract_info(self, email_data: Dict) -> Dict[str, Any]:
        """
        Extracts deadlines, tasks, and entities using a mix of Regex and LLM.
        Returns a normalized, downstream-friendly structure.
        """
        body_text = email_data.get("body") or email_data.get("body_full") or ""
        
        info = {
            "deadlines": [],
            "tasks": [],
            "entities": []
        }
        
        # A. Regex for explicit dates/deadlines
        # Matches: "Jan 15", "next Friday", "tomorrow", "by Friday", "end of month"
        date_regex = r'\b(\d{1,2}(?:st|nd|rd|th)?\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?|next\s+(?:Mon|Tues|Wed|Thurs|Fri|Sat|Sun)[a-z]*|tomorrow|end of (?:the )?month|by (?:Mon|Tues|Wed|Thurs|Fri|Sat|Sun)[a-z]*)\b'
        
        matches = re.finditer(date_regex, body_text, re.IGNORECASE)
        for m in matches:
            info["deadlines"].append({"text": m.group(0), "iso_date": None, "source": "regex"})

        # Normalize a small set of relative expressions into ISO dates (best-effort).
        # Uses the email header date if present; otherwise uses now().
        base_dt = None
        try:
            if email_data.get("date"):
                base_dt = date_parser.parse(email_data["date"])
        except Exception:
            base_dt = None
        if not base_dt:
            base_dt = datetime.datetime.now()

        def next_weekday(d: datetime.datetime, weekday: int) -> datetime.datetime:
            # weekday: Mon=0..Sun=6
            days_ahead = (weekday - d.weekday() + 7) % 7
            if days_ahead == 0:
                days_ahead = 7
            return d + datetime.timedelta(days=days_ahead)

        weekday_map = {
            "mon": 0, "monday": 0,
            "tue": 1, "tues": 1, "tuesday": 1,
            "wed": 2, "wednesday": 2,
            "thu": 3, "thurs": 3, "thursday": 3,
            "fri": 4, "friday": 4,
            "sat": 5, "saturday": 5,
            "sun": 6, "sunday": 6,
        }

        for d in info["deadlines"]:
            txt = (d.get("text") or "").strip().lower()
            try:
                if txt == "tomorrow":
                    d["iso_date"] = (base_dt + datetime.timedelta(days=1)).date().isoformat()
                elif "end of" in txt and "month" in txt:
                    # last day of the base month
                    y, mth = base_dt.year, base_dt.month
                    if mth == 12:
                        last = datetime.date(y, 12, 31)
                    else:
                        first_next = datetime.date(y, mth + 1, 1)
                        last = first_next - datetime.timedelta(days=1)
                    d["iso_date"] = last.isoformat()
                else:
                    m2 = re.search(r"(?:by|next)\s+([a-z]+)", txt)
                    if m2:
                        wd = weekday_map.get(m2.group(1)[:5], weekday_map.get(m2.group(1)))
                        if wd is not None:
                            d["iso_date"] = next_weekday(base_dt, wd).date().isoformat()
            except Exception:
                pass
            
        # B. LLM for deeper semantic extraction (Tasks & Companies)
        prompt = (
            "Analyze the email below and extract actionable information.\n"
            "Return VALID JSON ONLY with this shape:\n"
            "{\n"
            '  "tasks": [{"text": string, "due": string|null}],\n'
            '  "entities": [{"name": string, "type": "person"|"company"|"other"}],\n'
            '  "deadlines": [{"text": string, "due": string|null}]\n'
            "}\n\n"
            "Rules:\n"
            "- Keep tasks short and action-oriented (e.g. 'Review the deck', 'Send feedback').\n"
            "- If a due date is implied (e.g. 'by Friday'), put it in 'due' as best-effort ISO date (YYYY-MM-DD) or null.\n"
            "- Include people/companies mentioned (not the recipient unless named).\n\n"
            f"Email Body:\n{body_text[:3500]}\n"
        )
        
        response = self._call_llm(prompt, "You are an entity extraction system. Output JSON only.")
        
        try:
            parsed = self._safe_json_from_llm(response) or {}

            # Tasks: allow either list[str] or list[dict]
            tasks = parsed.get("tasks") or []
            if tasks and isinstance(tasks[0], str):
                info["tasks"] = [{"text": t, "due": None} for t in tasks]
            else:
                info["tasks"] = tasks

            # Entities: allow either list[str] or list[dict]
            ents = parsed.get("entities") or []
            if ents and isinstance(ents[0], str):
                info["entities"] = [{"name": e, "type": "other"} for e in ents]
            else:
                info["entities"] = ents

            # Deadlines
            llm_deadlines = parsed.get("deadlines") or []
            for d in llm_deadlines:
                if isinstance(d, str):
                    info["deadlines"].append({"text": d, "iso_date": None, "source": "llm"})
                elif isinstance(d, dict):
                    info["deadlines"].append(
                        {
                            "text": d.get("text") or "",
                            "iso_date": d.get("due"),
                            "source": "llm",
                        }
                    )
        except Exception as e:
            print(f"Extraction parsing failed: {e}")
        return info

    def extract_info_from_thread(self, thread_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Runs extraction over the combined thread body (useful for messy chains).
        """
        latest = thread_data.get("latest") or {}
        return self.extract_info({"body": thread_data.get("combined_body", ""), "date": latest.get("date", "")})

    # ==========================
    # 4. DRAFT REPLIES
    # ==========================
    
    def draft_reply(self, email_data: Dict, research_context: str = "", *, tone: str = "professional") -> str:
        """
        Drafts a professional reply incorporating optional research context.
        """
        extracted = self.extract_info(email_data)
        deadlines = extracted.get("deadlines", [])
        tasks = extracted.get("tasks", [])
        prompt = (
            f"Draft a context-aware, {tone}, neutral email reply to the message below. "
            f"Do not include the subject line. Just the email body text.\n\n"
            f"Incoming Message:\n"
            f"From: {email_data.get('sender')}\n"
            f"Subject: {email_data.get('subject')}\n"
            f"Body:\n{(email_data.get('body') or '')[:2500]}\n\n"
            f"Actionable info (for reference):\n"
            f"- Deadlines: {json.dumps(deadlines)[:800]}\n"
            f"- Tasks: {json.dumps(tasks)[:800]}\n\n"
        )
        
        if research_context:
            prompt += f"Use this additional context in your reply:\n{research_context}\n\n"
            
        prompt += (
            "Requirements:\n"
            "- Be concise.\n"
            "- If there's a clear ask, acknowledge it and propose a next step.\n"
            "- If a deadline is mentioned, confirm it.\n\n"
            "Draft Reply:"
        )
        
        return self._call_llm(prompt, "You are a professional communications assistant.")

    def create_gmail_draft(
        self,
        *,
        to: str,
        subject: str,
        body: str,
        cc: str = "",
        bcc: str = "",
        thread_id: str = "",
        in_reply_to: str = "",
        references: str = "",
    ) -> Dict[str, Any]:
        """
        Creates a Gmail draft (does NOT send).

        - `thread_id`: attach the draft to an existing thread.
        - `in_reply_to` / `references`: RFC 5322 headers (often pulled from `message_id_header`).
        """
        if not self.service:
            return {"error": "Gmail service not authenticated"}

        msg = EmailMessage()
        msg["To"] = to
        msg["Subject"] = subject
        msg["Date"] = formatdate(localtime=True)
        msg["Message-ID"] = make_msgid()
        if cc:
            msg["Cc"] = cc
        if bcc:
            msg["Bcc"] = bcc
        if in_reply_to:
            msg["In-Reply-To"] = in_reply_to
        if references:
            msg["References"] = references
        msg.set_content(body)

        raw = base64.urlsafe_b64encode(msg.as_bytes()).decode("utf-8")
        draft_body: Dict[str, Any] = {"message": {"raw": raw}}
        if thread_id:
            draft_body["message"]["threadId"] = thread_id

        try:
            draft = self.service.users().drafts().create(userId="me", body=draft_body).execute()
            return draft
        except Exception as e:
            return {"error": str(e)}

    # ==========================
    # JOB SEARCH / APPLY HELPERS
    # ==========================

    def parse_job_description(self, *, role_title: str, company: str, job_description: str) -> Dict[str, Any]:
        """
        Extract role understanding from JD:
        - role title
        - team/domain
        - required skills
        - team priorities
        """
        prompt = (
            "Extract structured understanding from this job description.\n"
            "Return VALID JSON ONLY with keys:\n"
            '{ "role_title": string, "company": string, "team_or_domain": string, '
            '"required_skills": [string], "what_the_team_cares_about": [string] }\n\n'
            "Rules:\n"
            "- Do not invent details not present.\n"
            "- team_or_domain should be concise (e.g. ML Infra, GenAI, Search, Ads).\n"
            "- required_skills should be 5-12 items.\n"
            "- what_the_team_cares_about should be 3-6 short bullets.\n\n"
            f"Company: {company}\n"
            f"Role title (provided): {role_title}\n\n"
            f"JOB DESCRIPTION:\n{self._truncate(job_description, 9000)}"
        )
        raw = self._call_llm(prompt, "You are a strict information extraction system. Output JSON only.")
        parsed = self._safe_json_from_llm(raw)
        if parsed:
            return parsed
        return {
            "role_title": role_title,
            "company": company,
            "team_or_domain": "",
            "required_skills": [],
            "what_the_team_cares_about": [],
            "raw": raw,
        }

    def parse_resume(self, *, resume_text: str) -> Dict[str, Any]:
        """
        Extract candidate understanding from resume:
        - core skills
        - relevant experiences
        - years of experience (if inferable)
        """
        prompt = (
            "Extract structured understanding from this resume.\n"
            "Return VALID JSON ONLY with keys:\n"
            '{ "core_skills": [string], "years_experience": number|null, '
            '"most_relevant_experiences": [{"title": string, "summary": string, "skills": [string]}] }\n\n'
            "Rules:\n"
            "- Be conservative; use null if years are unclear.\n"
            "- Select 2-4 most relevant experiences.\n"
            "- Do not invent employers/projects.\n\n"
            f"RESUME:\n{self._truncate(resume_text, 9000)}"
        )
        raw = self._call_llm(prompt, "You are a strict information extraction system. Output JSON only.")
        parsed = self._safe_json_from_llm(raw)
        if parsed:
            return parsed
        return {
            "core_skills": [],
            "years_experience": None,
            "most_relevant_experiences": [],
            "raw": raw,
        }

    def find_recruiter_contact(
        self,
        *,
        company: str,
        role_title: str,
        team_or_domain: str = "",
        min_emails: int = 3,
    ) -> Dict[str, Any]:
        """
        Use Linkup to find recruiter contact details for a specific company and role.

        Improvements over the original:
        - Uses a richer schema that asks for *multiple* contacts per query (name, email,
          title, linkedin_url) reducing the number of API calls needed.
        - Better-targeted queries that explicitly mention LinkedIn, career pages, etc.
        - Extracts emails and LinkedIn URLs from the raw Linkup response text as a
          fallback when structured output is incomplete.
        - Deduplicates contacts by normalized name to avoid returning the same person twice.
        - Discovers corporate email domains from early results and uses them to validate
          subsequent contacts.
        - Provides actionable fallback suggestions including LinkedIn search URLs.

        Returns:
          {
            "name": "...",
            "title": "...",
            "email": "...",
            "emails": ["..."],
            "linkedin_urls": ["..."],
            "contacts": [{"name","title","email","linkedin_url","confidence","source"}],
            "confidence": "high|medium|low",
            "source": "...",
            "fallback_suggestion": "..."
          }
        """
        # Richer schema: ask for an *array* of contacts per query (aim for multiple).
        schema = {
            "type": "object",
            "properties": {
                "contacts": {
                    "type": "array",
                    "description": "List of recruiter or hiring contacts at this company. Include as many as you can find (aim for at least 3–5 different people with their work email addresses). Each contact must have a valid email.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "description": "Recruiter full name"},
                            "title": {"type": "string", "description": "Recruiter job title (e.g., Senior Technical Recruiter)"},
                            "email": {"type": "string", "description": "Recruiter email address (full, not masked/obfuscated)"},
                            "linkedin_url": {"type": "string", "description": "LinkedIn profile URL if available"},
                            "confidence": {"type": "string", "description": "high, medium, or low"},
                            "source": {"type": "string", "description": "Where this contact info was found"},
                        },
                        "required": ["name", "email"],
                    },
                },
                "fallback_suggestion": {
                    "type": "string",
                    "description": "If no recruiter emails found, suggest how to reach the company's recruiting team",
                },
            },
            "required": ["contacts", "fallback_suggestion"],
        }

        domain = self._company_domain_hint(company)
        team_clause = f" {team_or_domain}" if team_or_domain else ""

        # Tier 1: Focused queries that are most likely to surface actual recruiter contacts.
        # Each query is crafted to hit different source types.
        role_specific_queries = [
            f"site:linkedin.com/in {company}{team_clause} recruiter OR 'talent acquisition' {role_title}",
            f"{company}{team_clause} recruiter email address {role_title} contact",
            f"{company} hiring team {role_title} recruiter name email",
        ]
        # Tier 2: Broader fallback queries when role-specific ones don't yield enough.
        broad_fallback_queries = [
            f"site:linkedin.com/in {company} technical recruiter OR 'talent acquisition partner'",
            f"{company} recruiting team email contact careers",
            f"{company} talent acquisition email recruiter contact",
        ]
        # Tier 3: When we still have too few contacts, ask explicitly for multiple recruiters.
        multi_recruiter_queries = [
            f"{company} list of recruiter emails hiring team contacts",
            f"{company} multiple recruiters talent acquisition team {role_title}",
        ]

        # Accumulator state.
        collected_contacts: List[Dict[str, str]] = []
        seen_emails: set = set()
        seen_names: set = set()
        discovered_domains: set = set()  # Corporate domains found so far.
        all_linkedin_urls: List[str] = []
        seen_linkedin: set = set()
        best_name = ""
        best_title = ""
        best_source = ""
        best_conf = "low"
        best_fallback = ""

        conf_rank = {"low": 0, "medium": 1, "high": 2}

        def _add_contact(
            name: str,
            email: str,
            *,
            title: str = "",
            linkedin_url: str = "",
            confidence: str = "medium",
            source: str = "",
        ) -> bool:
            """
            Validate and add a single contact to the collection.
            Returns True if it was added (new and valid).
            """
            nonlocal best_name, best_title, best_source, best_conf

            email = (email or "").strip()
            name = (name or "").strip()
            title = (title or "").strip()
            linkedin_url = (linkedin_url or "").strip()
            confidence = (confidence or "medium").lower().strip()
            if confidence not in conf_rank:
                confidence = "medium"

            # Track recruiter identity regardless of whether email is usable.
            if name and not best_name:
                best_name = name
            if title and not best_title:
                best_title = title
            if source and not best_source:
                best_source = source
            if conf_rank.get(confidence, 0) > conf_rank.get(best_conf, 0):
                best_conf = confidence

            # Collect LinkedIn URLs.
            if linkedin_url and linkedin_url not in seen_linkedin:
                seen_linkedin.add(linkedin_url)
                all_linkedin_urls.append(linkedin_url)

            # Validate email.
            if not email or not self._is_full_email(email):
                return False
            if self._is_placeholder_email(email):
                return False
            if self._is_relay_or_marketing_email(email):
                return False
            if not self._looks_like_corporate_email(email, company, discovered_domains=discovered_domains):
                return False

            email_l = email.lower()
            if email_l in seen_emails:
                return False

            # Deduplicate by name: if we already have a contact with the same name, skip
            # (keep the first occurrence which tends to be higher quality).
            norm = self._normalize_name(name)
            if norm and norm in seen_names:
                return False

            # Track the domain for future validation.
            email_domain = email_l.split("@", 1)[1]
            if email_domain not in self._FREE_EMAIL_DOMAINS:
                discovered_domains.add(email_domain)

            seen_emails.add(email_l)
            if norm:
                seen_names.add(norm)

            collected_contacts.append({
                "name": name,
                "title": title,
                "email": email,
                "linkedin_url": linkedin_url,
                "confidence": confidence,
                "source": source,
            })
            return True

        def _process_linkup_response(out: Dict[str, Any], *, query: str) -> None:
            """
            Process a Linkup structured response.
            Extracts contacts from the structured output AND scrapes the raw
            response text for emails/LinkedIn URLs that the model may have missed.
            """
            nonlocal best_fallback

            if out.get("error"):
                return

            # 1) Process structured contacts array.
            contacts_arr = out.get("contacts") or []
            if isinstance(contacts_arr, list):
                for c in contacts_arr:
                    if not isinstance(c, dict):
                        continue
                    _add_contact(
                        name=str(c.get("name") or ""),
                        email=str(c.get("email") or ""),
                        title=str(c.get("title") or ""),
                        linkedin_url=str(c.get("linkedin_url") or ""),
                        confidence=str(c.get("confidence") or "medium"),
                        source=str(c.get("source") or query),
                    )

            # 2) Backwards-compat: also check top-level name/email (older schema shape).
            top_email = str(out.get("email") or "").strip()
            top_name = str(out.get("name") or "").strip()
            if top_email:
                _add_contact(
                    name=top_name,
                    email=top_email,
                    title=str(out.get("title") or ""),
                    confidence=str(out.get("confidence") or "medium"),
                    source=str(out.get("source") or query),
                )

            # 3) Scrape the entire response blob for any emails/LinkedIn URLs
            #    that the structured output didn't capture.
            blob = json.dumps(out)
            scraped_emails = self._extract_all_emails_from_text(blob)
            for se in scraped_emails:
                _add_contact(name="", email=se, source=f"scraped from response to: {query[:60]}")

            scraped_linkedin = self._extract_linkedin_urls(blob)
            for url in scraped_linkedin:
                if url not in seen_linkedin:
                    seen_linkedin.add(url)
                    all_linkedin_urls.append(url)

            # 4) Capture fallback suggestions.
            fallback = str(out.get("fallback_suggestion") or "").strip()
            if fallback and not best_fallback:
                best_fallback = fallback

        def search_queries(queries: List[str], *, tier: str) -> None:
            for q in queries:
                out = self._call_linkup_structured(query=q, schema=schema, depth="deep")
                _process_linkup_response(out, query=q)

        # Execute tiered search.
        search_queries(role_specific_queries, tier="role_specific")

        if len(collected_contacts) < max(min_emails, 1):
            search_queries(broad_fallback_queries, tier="broad_fallback")

        if len(collected_contacts) < max(min_emails, 1):
            search_queries(multi_recruiter_queries, tier="multi_recruiter")

        # Sort: same-company contacts first, then by confidence (descending).
        target_domain = self._company_domain_hint(company)
        inferred = self._infer_domain_from_company_name(company)
        target_domains = {d for d in (target_domain, inferred) if d}

        def _contact_sort_key(c: Dict[str, Any]) -> Tuple[int, int]:
            email_addr = (c.get("email") or "").strip().lower()
            if not email_addr or "@" not in email_addr:
                return (1, 0)
            domain = email_addr.split("@", 1)[1]
            same_company = 0 if domain in target_domains else 1
            conf = conf_rank.get(str(c.get("confidence", "low")), 0)
            return (same_company, -conf)

        collected_contacts.sort(key=_contact_sort_key)
        top_contacts = collected_contacts[: max(min_emails, 1)]
        emails = [c["email"] for c in top_contacts]

        # Build the result.
        result: Dict[str, Any] = {
            "name": "",
            "title": "",
            "email": "",
            "emails": emails,
            "linkedin_urls": all_linkedin_urls[:5],
            "contacts": top_contacts,
            "confidence": "low",
            "source": "",
            "fallback_suggestion": "",
        }

        if top_contacts:
            primary = top_contacts[0]
            result["name"] = primary.get("name") or best_name
            result["title"] = primary.get("title") or best_title
            result["email"] = primary["email"]
            result["confidence"] = primary.get("confidence") or best_conf
            result["source"] = primary.get("source") or best_source
        else:
            # No usable emails found — build a helpful fallback.
            result["name"] = best_name
            result["title"] = best_title
            result["source"] = best_source
            result["confidence"] = "low" if not best_name else best_conf

        # Construct an actionable fallback suggestion.
        if not result["email"]:
            parts: List[str] = []
            if best_fallback:
                parts.append(best_fallback)
            if all_linkedin_urls:
                parts.append(f"LinkedIn profiles found: {', '.join(all_linkedin_urls[:3])}")
            if domain:
                parts.append(
                    f"Try common patterns: recruiting@{domain}, careers@{domain}, "
                    f"talent@{domain}"
                )
            if best_name:
                parts.append(f"Recruiter name found: {best_name}. Try LinkedIn outreach.")
            # Always include a LinkedIn search link.
            company_encoded = company.replace(" ", "+")
            role_encoded = role_title.replace(" ", "+")
            linkedin_search = (
                f"https://www.linkedin.com/search/results/people/"
                f"?keywords={company_encoded}+recruiter+{role_encoded}"
            )
            parts.append(f"LinkedIn recruiter search: {linkedin_search}")
            result["fallback_suggestion"] = " | ".join(parts) if parts else (
                f"No recruiter contact found. Search LinkedIn for {company} recruiters."
            )
        else:
            result["fallback_suggestion"] = best_fallback

        return result

    def build_job_fit_profile(
        self,
        *,
        job_description: str,
        resume_text: str,
        role_title: str = "",
        company: str = "",
        candidate_name: str = "",
    ) -> Dict[str, Any]:
        """
        Creates structured context for downstream outreach:
        - strongest matching skills/experiences
        - gaps / risks
        - suggested talking points
        - a short fit summary

        This is designed to be robust to unknown upstream schemas: caller just passes strings.
        """
        jd = self._truncate(job_description, 7000)
        resume = self._truncate(resume_text, 7000)

        prompt = (
            "You are helping draft a recruiter outreach email for a job application.\n"
            "Given a job description and a candidate resume, produce VALID JSON ONLY with this shape:\n"
            "{\n"
            '  "fit_summary": string,               \n'
            '  "top_matches": [{"evidence": string, "why_it_matters": string}],\n'
            '  "relevant_experiences": [{"project_or_role": string, "impact": string, "skills": [string]}],\n'
            '  "gaps_or_risks": [string],\n'
            '  "talking_points": [string],\n'
            '  "questions_for_recruiter": [string]\n'
            "}\n\n"
            "Rules:\n"
            "- Evidence must be specific (numbers/metrics if present) and grounded in the resume.\n"
            "- Keep it concise: 3-6 top_matches max, 2-4 relevant_experiences max.\n"
            "- If something isn't in the resume, do not invent it.\n\n"
            f"Role title: {role_title}\n"
            f"Company: {company}\n"
            f"Candidate: {candidate_name}\n\n"
            f"JOB DESCRIPTION:\n{jd}\n\n"
            f"RESUME:\n{resume}\n"
        )

        raw = self._call_llm(prompt, "You are a strict JSON extraction system. Output JSON only.")
        parsed = self._safe_json_from_llm(raw)
        if not parsed:
            return {
                "fit_summary": "",
                "top_matches": [],
                "relevant_experiences": [],
                "gaps_or_risks": [],
                "talking_points": [],
                "questions_for_recruiter": [],
                "raw": raw,
            }
        return parsed

    def draft_recruiter_outreach_email(
        self,
        *,
        role_title: str,
        company: str,
        recruiter_name: str = "",
        job_url: str = "",
        job_id: str = "",
        job_description: str = "",
        resume_text: str = "",
        candidate_name: str = "",
        candidate_location: str = "",
        candidate_linkedin: str = "",
        candidate_portfolio: str = "",
        additional_context: str = "",
        tone: str = "professional",
    ) -> Dict[str, str]:
        """
        Drafts a personalized recruiter outreach email for a selected role.

        Returns:
          {"subject": "...", "body": "..."}

        Inputs are intentionally simple strings to make integration easy with whatever
        Linkup schemas your teammate uses.
        """
        fit = self.build_job_fit_profile(
            job_description=job_description,
            resume_text=resume_text,
            role_title=role_title,
            company=company,
            candidate_name=candidate_name,
        )

        # Keep the prompt small + high-signal.
        fit_summary = self._truncate(str(fit.get("fit_summary", "")), 800)
        top_matches = self._truncate(json.dumps(fit.get("top_matches", [])), 1400)
        talking_points = self._truncate(json.dumps(fit.get("talking_points", [])), 900)

        prompt = (
            f"Draft a {tone}, concise, personalized recruiter outreach email.\n"
            "Return VALID JSON ONLY with keys: subject, body.\n\n"
            "Constraints:\n"
            "- Subject: <= 78 chars.\n"
            "- Body: 90-160 words.\n"
            "- 1 short intro line, 2-3 fit highlights (bullets allowed), 1 clear ask, polite close.\n"
            "- Write from the CANDIDATE to the recruiter (first-person as candidate).\n"
            "- Do NOT write as the recruiter.\n"
            "- Do not claim referrals unless explicitly stated.\n"
            "- Use the job URL/ID if provided.\n\n"
            f"Company: {company}\n"
            f"Role: {role_title}\n"
            f"Recruiter name: {recruiter_name}\n"
            f"Job URL: {job_url}\n"
            f"Job ID: {job_id}\n"
            f"Candidate name: {candidate_name}\n"
            f"Candidate location: {candidate_location}\n"
            f"LinkedIn: {candidate_linkedin}\n"
            f"Portfolio: {candidate_portfolio}\n"
            f"Additional context: {additional_context}\n\n"
            f"Fit summary:\n{fit_summary}\n\n"
            f"Top matches:\n{top_matches}\n\n"
            f"Talking points:\n{talking_points}\n"
        )

        raw = self._call_llm(prompt, "You write outreach emails. Output JSON only.")
        parsed = self._safe_json_from_llm(raw) or {}

        subject = (parsed.get("subject") or "").strip()
        body = (parsed.get("body") or "").strip()

        # Some models return nested JSON or fenced blocks inside "body".
        # Attempt one more pass to unwrap nested {"subject","body"} payloads.
        if body:
            nested = self._safe_json_from_llm(body)
            if nested:
                nested_subject = (nested.get("subject") or "").strip()
                nested_body = (nested.get("body") or "").strip()
                if nested_subject and not subject:
                    subject = nested_subject
                if nested_body:
                    body = nested_body

        # Strip markdown fences if any remain.
        if body.startswith("```"):
            body = body.replace("```json", "").replace("```", "").strip()

        # If body still looks like a JSON object string, try extracting "body" field robustly.
        body_candidate = body.strip()
        if body_candidate.startswith("{") and body_candidate.endswith("}"):
            try:
                obj = json.loads(body_candidate)
                if isinstance(obj, dict) and isinstance(obj.get("body"), str):
                    body = obj["body"].strip()
                    if isinstance(obj.get("subject"), str) and not subject:
                        subject = obj["subject"].strip()
            except Exception:
                # Best-effort fallback for loosely valid JSON-like strings
                m = re.search(r'"body"\s*:\s*"(.+?)"\s*(?:,|\})', body_candidate, re.DOTALL)
                if m:
                    extracted = m.group(1)
                    # Unescape common JSON escapes for readability.
                    extracted = extracted.replace('\\"', '"').replace("\\n", "\n").replace("\\t", "\t")
                    body = extracted.strip()

        # Fallback: if model didn't comply, return raw as body.
        if not subject and not body:
            return {"subject": f"Interest in {role_title} at {company}", "body": raw.strip()}
        if not subject:
            subject = f"Interest in {role_title} at {company}"
        return {"subject": subject, "body": body}

    def build_recruiter_outreach_package(
        self,
        *,
        role_title: str,
        job_description: str,
        resume_text: str,
        company: str,
        preferred_tone: str = "formal",
        candidate_name: str = "",
        candidate_location: str = "",
        candidate_linkedin: str = "",
        candidate_portfolio: str = "",
        additional_context: str = "",
    ) -> Dict[str, Any]:
        """
        End-to-end flow for this responsibility:
        1) Parse JD
        2) Parse Resume
        3) Find recruiter email via Linkup
        4) Draft personalized outreach email

        Output:
        {
          "recruiter": {"name","email","confidence"},
          "email_subject": "...",
          "email_body": "...",
          "analysis": {"jd":..., "resume":...}
        }
        """
        jd = self.parse_job_description(role_title=role_title, company=company, job_description=job_description)
        resume = self.parse_resume(resume_text=resume_text)
        recruiter = self.find_recruiter_contact(
            company=company,
            role_title=role_title,
            team_or_domain=str(jd.get("team_or_domain", "")),
        )

        tone = "professional" if preferred_tone.lower() == "formal" else "friendly"
        draft = self.draft_recruiter_outreach_email(
            role_title=role_title,
            company=company,
            recruiter_name=recruiter.get("name", ""),
            job_description=job_description,
            resume_text=resume_text,
            candidate_name=candidate_name,
            candidate_location=candidate_location,
            candidate_linkedin=candidate_linkedin,
            candidate_portfolio=candidate_portfolio,
            additional_context=additional_context,
            tone=tone,
        )

        return {
            "recruiter": {
                "name": recruiter.get("name", ""),
                "title": recruiter.get("title", ""),
                "email": recruiter.get("email", ""),
                "emails": recruiter.get("emails", []),
                "linkedin_urls": recruiter.get("linkedin_urls", []),
                "contacts": recruiter.get("contacts", []),
                "confidence": recruiter.get("confidence", "low"),
                "fallback_suggestion": recruiter.get("fallback_suggestion", ""),
            },
            "email_subject": draft.get("subject", ""),
            "email_body": draft.get("body", ""),
            "analysis": {"jd": jd, "resume": resume},
        }

    def draft_recruiter_follow_up(
        self,
        *,
        previous_email_body: str,
        days_since: int = 5,
        new_update: str = "",
        tone: str = "professional",
    ) -> Dict[str, str]:
        """
        Produces a short follow-up email (subject + body) based on your prior outreach.
        """
        prompt = (
            f"Draft a {tone}, short follow-up email.\n"
            "Return VALID JSON ONLY with keys: subject, body.\n\n"
            f"Days since last email: {days_since}\n"
            f"New update (if any): {new_update}\n\n"
            f"Previous email body:\n{self._truncate(previous_email_body, 2500)}\n\n"
            "Constraints:\n"
            "- Body: 50-90 words.\n"
            "- Polite, assumes they are busy.\n"
            "- Restate the role in one phrase.\n"
        )
        raw = self._call_llm(prompt, "You write concise follow-ups. Output JSON only.")
        parsed = self._safe_json_from_llm(raw) or {}
        subject = (parsed.get("subject") or "Following up").strip()
        body = (parsed.get("body") or raw).strip()
        return {"subject": subject, "body": body}

    def draft_reply_to_recruiter_thread(
        self,
        *,
        thread_data: Dict[str, Any],
        job_description: str = "",
        resume_text: str = "",
        additional_context: str = "",
        tone: str = "professional",
    ) -> str:
        """
        Given an ongoing recruiter thread, drafts a context-aware reply.
        Useful once the recruiter responds and you need to answer questions / schedule / share info.
        """
        combined = self._truncate(thread_data.get("combined_body", ""), 8000)
        jd = self._truncate(job_description, 4000)
        resume = self._truncate(resume_text, 4000)

        prompt = (
            f"Draft a {tone}, neutral reply to the recruiter thread below.\n"
            "Do not include the subject line. Just the body.\n\n"
            "Use these constraints:\n"
            "- If the recruiter asks for availability, propose 2-3 time windows.\n"
            "- If they ask for materials (resume/portfolio), acknowledge and offer to share/link.\n"
            "- If unclear, ask 1-2 clarifying questions.\n\n"
            f"Additional context: {additional_context}\n\n"
            f"JOB DESCRIPTION (optional):\n{jd}\n\n"
            f"RESUME (optional):\n{resume}\n\n"
            f"THREAD:\n{combined}\n\n"
            "Reply:"
        )
        return self._call_llm(prompt, "You are a professional recruiting communications assistant.")

# ==========================================
# EXAMPLE RUNNER
# ==========================================
if __name__ == "__main__":
    print("Initializing Email Handler Agent...")
    
    # 1. Configuration
    HF_TOKEN = os.getenv("HF_TOKEN")
    if not HF_TOKEN:
        # Optional fallback: read from local token.json (not the Gmail OAuth token file).
        try:
            with open("token.json", "r") as f:
                data = json.load(f) or {}
            HF_TOKEN = data.get("hf_token") or data.get("HF_TOKEN") or data.get("token")
        except Exception:
            HF_TOKEN = None

    if not HF_TOKEN:
        print("ERROR: Please set HF_TOKEN environment variable or add {\"hf_token\": \"...\"} to token.json.")
        exit(1)
        
    # Initialize
    agent = EmailHandler(hf_token=HF_TOKEN)
    
    # 2. Read Emails
    print("\nFetching recent emails...")
    messages = agent.list_emails(max_results=3)
    
    if not messages:
        print("No emails found or auth failed.")
    
    for msg in messages:
        print("\n" + "="*50)
        
        # Parse
        email = agent.get_email_details(msg['id'])
        print(f"SUBJECT: {email['subject']}")
        print(f"FROM:    {email['sender']}")
        
        # Summarize
        print("\n--- SUMMARY ---")
        summary = agent.summarize_email(email)
        print(summary)
        
        # Extract
        print("\n--- EXTRACTION ---")
        info = agent.extract_info(email)
        print(json.dumps(info, indent=2))
        
        # Draft Reply
        print("\n--- DRAFT REPLY ---")
        reply = agent.draft_reply(email, research_context="I am free next Tuesday at 2 PM.")
        print(reply)
