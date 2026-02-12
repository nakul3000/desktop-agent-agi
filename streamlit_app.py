"""
Streamlit Chat UI for JobAgent with Memory Inspector.

Run with:
    streamlit run streamlit_app.py
"""

import json
from pathlib import Path

import streamlit as st
import memory
from app import JobAgent

# ------------------------------------------------------------------ #
# Page config
# ------------------------------------------------------------------ #

st.set_page_config(
    page_title="Pathfind AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------ #
# Custom CSS for a polished dark-themed chat UI
# ------------------------------------------------------------------ #

st.markdown(
    """
    <style>
    /* Hide default Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Tighten up the top padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 1rem;
    }

    /* Style the sidebar header */
    [data-testid="stSidebar"] {
        min-width: 340px;
        max-width: 480px;
    }

    /* Memory table styling */
    .memory-table {
        font-size: 0.82rem;
        width: 100%;
    }

    /* Badge styling for roles */
    .role-user {
        background-color: #7C3AED;
        color: white;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .role-assistant {
        background-color: #1E1E2E;
        color: #FAFAFA;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
        border: 1px solid #333;
    }
    .role-tool {
        background-color: #065F46;
        color: #D1FAE5;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
    }

    /* Fact card */
    .fact-card {
        background: #1E1E2E;
        border: 1px solid #333;
        border-radius: 8px;
        padding: 10px 14px;
        margin-bottom: 8px;
    }
    .fact-key {
        color: #7C3AED;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .fact-value {
        color: #FAFAFA;
        font-size: 0.82rem;
    }
    .fact-meta {
        color: #888;
        font-size: 0.72rem;
        margin-top: 4px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------ #
# Session-state initialisation
# ------------------------------------------------------------------ #


def _init_agent() -> JobAgent:
    """Create a fresh JobAgent and persist it in session state."""
    agent = JobAgent()
    return agent


if "agent" not in st.session_state:
    st.session_state.agent = _init_agent()

if "messages" not in st.session_state:
    # Each entry: {"role": "user" | "assistant", "content": str, "files": [...]}
    st.session_state.messages = []

if "downloadable_files" not in st.session_state:
    # Tracks the latest downloadable files by type for the sidebar artifacts tab
    # e.g. {"resume_docx": {...}, "resume_pdf": {...}, "research_pdf": {...}}
    st.session_state.downloadable_files = {}


# ------------------------------------------------------------------ #
# Sidebar  —  Controls + Memory Inspector
# ------------------------------------------------------------------ #

with st.sidebar:
    st.markdown("## 🤖 Pathfind AI")
    st.caption("Intelligent Job Search Assistant")

    st.divider()

    # ---- Controls ------------------------------------------------ #
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Reset Chat", use_container_width=True):
            st.session_state.agent.reset()
            st.session_state.messages = []
            st.session_state.downloadable_files = {}
            st.rerun()
    with col2:
        refresh_memory = st.button("🔃 Refresh Memory", use_container_width=True)

    st.divider()

    # ---- Memory Inspector ---------------------------------------- #
    st.markdown("### 🧠 Memory Inspector")

    agent: JobAgent = st.session_state.agent
    session_id = agent.session_id

    tab_turns, tab_artifacts, tab_facts = st.tabs(["💬 Turns", "📦 Artifacts", "📌 Facts"])

    # -- Turns tab ------------------------------------------------- #
    with tab_turns:
        turns = memory.get_turns_by_session(session_id)
        if not turns:
            st.info("No turns recorded yet. Start chatting!")
        else:
            st.caption(f"{len(turns)} turn(s) in session")
            for t in turns:
                row = dict(t)
                role = row.get("role", "unknown")
                text = row.get("text", "")
                ts = row.get("timestamp", "")

                # Role badge
                badge_class = {
                    "user": "role-user",
                    "assistant": "role-assistant",
                    "tool": "role-tool",
                }.get(role, "role-assistant")

                tool_label = ""
                if row.get("tool_name"):
                    tool_label = f" &middot; <code>{row['tool_name']}</code>"

                st.markdown(
                    f'<span class="{badge_class}">{role}</span>{tool_label} '
                    f'<span style="color:#666;font-size:0.7rem;">{ts}</span>',
                    unsafe_allow_html=True,
                )

                # Show a truncated preview, expandable
                preview = text[:200] + ("…" if len(text) > 200 else "")
                if len(text) > 200:
                    with st.expander(preview, expanded=False):
                        st.text(text)
                else:
                    st.text(preview)

    # -- Artifacts tab --------------------------------------------- #
    with tab_artifacts:
        # ---- Downloadable files section ---- #
        dl_files = st.session_state.get("downloadable_files", {})
        if dl_files:
            st.markdown("#### 📥 Downloads")
            for ftype, finfo in dl_files.items():
                fpath = finfo.get("path", "")
                if fpath and Path(fpath).exists():
                    with open(fpath, "rb") as fp:
                        st.download_button(
                            label=finfo.get("label", "Download"),
                            data=fp.read(),
                            file_name=finfo.get("filename", Path(fpath).name),
                            mime=finfo.get("mime", "application/octet-stream"),
                            use_container_width=True,
                            key=f"sidebar_dl_{ftype}",
                        )
            st.divider()

        artifacts = memory.get_artifacts_by_session(session_id)
        if not artifacts and not dl_files:
            st.info("No artifacts stored yet.")
        elif artifacts:
            st.caption(f"{len(artifacts)} artifact(s)")
            for a in artifacts:
                row = dict(a)
                art_type = row.get("type", "unknown")
                content = row.get("content", "")
                ts = row.get("timestamp", "")
                created_by = row.get("created_by", "")

                header = f"**{art_type}**"
                if created_by:
                    header += f"  ·  _{created_by}_"

                with st.expander(header, expanded=False):
                    st.caption(ts)
                    # Try to render as JSON, fall back to plain text
                    try:
                        parsed = json.loads(content)
                        st.json(parsed)
                    except (json.JSONDecodeError, TypeError):
                        st.text(content[:2000])

    # -- Facts tab ------------------------------------------------- #
    with tab_facts:
        facts = memory.get_facts_by_session(session_id)
        if not facts:
            st.info("No facts extracted yet.")
        else:
            st.caption(f"{len(facts)} fact(s)")

            # Optional kind filter
            all_kinds = sorted(set(dict(f).get("kind", "") for f in facts))
            if len(all_kinds) > 1:
                selected_kind = st.selectbox(
                    "Filter by kind",
                    options=["all"] + all_kinds,
                    index=0,
                )
            else:
                selected_kind = "all"

            for f in facts:
                row = dict(f)
                if selected_kind != "all" and row.get("kind") != selected_kind:
                    continue

                kind = row.get("kind", "")
                key = row.get("key", "")
                value = row.get("value", "")
                confidence = row.get("confidence", 0)
                ts = row.get("timestamp", "")

                st.markdown(
                    f"""<div class="fact-card">
                        <div class="fact-key">{kind}: {key}</div>
                        <div class="fact-value">{value}</div>
                        <div class="fact-meta">confidence: {confidence:.0%} &middot; {ts}</div>
                    </div>""",
                    unsafe_allow_html=True,
                )


# ------------------------------------------------------------------ #
# Helper — render download buttons for a list of file dicts
# ------------------------------------------------------------------ #

def _render_download_buttons(files: list[dict], key_prefix: str = "dl"):
    """Render download buttons for generated files inside a chat message."""
    valid_files = [f for f in files if f.get("path") and Path(f["path"]).exists()]
    if not valid_files:
        return
    st.markdown("---")
    st.markdown("**📥 Downloads:**")
    cols = st.columns(min(len(valid_files), 3))
    for i, finfo in enumerate(valid_files):
        with cols[i % len(cols)]:
            with open(finfo["path"], "rb") as fp:
                st.download_button(
                    label=finfo.get("label", "Download"),
                    data=fp.read(),
                    file_name=finfo.get("filename", Path(finfo["path"]).name),
                    mime=finfo.get("mime", "application/octet-stream"),
                    use_container_width=True,
                    key=f"{key_prefix}_{i}",
                )


# ------------------------------------------------------------------ #
# Main area  —  Chat interface
# ------------------------------------------------------------------ #

st.markdown("# 🤖 Pathfind AI")
st.caption(
    "Search for jobs · Research companies · Tailor your resume · Draft cover letters & emails"
)

# Render conversation history
for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # Render download buttons for this message if it has files
        if msg.get("files"):
            _render_download_buttons(msg["files"], key_prefix=f"hist_{idx}")

# Chat input
if prompt := st.chat_input("Ask me anything about your job search…"):
    # Append & display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get agent response with a spinner
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            response = st.session_state.agent.chat(prompt)
        st.markdown(response)

        # Check for generated files from the agent
        generated_files = list(st.session_state.agent.last_generated_files)
        if generated_files:
            _render_download_buttons(generated_files, key_prefix="new")
            # Update the sidebar downloadable files registry
            for finfo in generated_files:
                st.session_state.downloadable_files[finfo["type"]] = finfo

    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
        "files": generated_files if generated_files else [],
    })

    # Rerun to refresh the memory sidebar automatically
    st.rerun()
