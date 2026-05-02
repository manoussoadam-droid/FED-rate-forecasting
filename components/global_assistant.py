"""Global AI assistant chatbot — always accessible via sidebar."""

from __future__ import annotations

import streamlit as st
import streamlit.components.v1 as components

from core.agentic_fed import (
    AgentUnavailableError,
    portkey_setup_status,
    run_fed_agent,
)


_ASSISTANT_NAME = "Alex"
_ASSISTANT_AVATAR = "🤖"
_ASSISTANT_GREETING = f"Hey! I'm {_ASSISTANT_NAME}, your Fed AI analyst"


def _inject_custom_sidebar_toggle() -> None:
    """Inject custom animated robot button to replace default sidebar toggle."""
    components.html(
        """
<script>
// Wait for Streamlit to load
const waitForSidebar = setInterval(() => {
    const toggleButton = document.querySelector('button[kind="header"]');
    if (toggleButton) {
        clearInterval(waitForSidebar);
        
        // Hide original button
        toggleButton.style.display = 'none';
        
        // Create custom robot button
        const robotBtn = document.createElement('div');
        robotBtn.className = 'custom-robot-toggle';
        robotBtn.innerHTML = '🤖';
        robotBtn.setAttribute('data-tooltip', "Hey! I'm Alex, your Fed AI analyst");
        
        // Insert before sidebar
        const sidebar = document.querySelector('[data-testid="stSidebar"]');
        if (sidebar && sidebar.parentNode) {
            sidebar.parentNode.insertBefore(robotBtn, sidebar);
        }
        
        // Click handler - trigger original button
        robotBtn.addEventListener('click', () => {
            toggleButton.click();
        });
    }
}, 100);

// Stop after 5 seconds if not found
setTimeout(() => clearInterval(waitForSidebar), 5000);
</script>

<style>
/* Custom robot toggle button */
.custom-robot-toggle {
    position: fixed;
    top: 1rem;
    left: 1rem;
    z-index: 999999;
    width: 60px;
    height: 60px;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 2rem;
    cursor: pointer;
    box-shadow: 0 4px 16px rgba(102, 126, 234, 0.4);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    animation: robot-blink 4s infinite;
    user-select: none;
}

.custom-robot-toggle:hover {
    transform: scale(1.15);
    box-shadow: 0 6px 24px rgba(102, 126, 234, 0.6);
}

.custom-robot-toggle:active {
    animation: robot-pulse 0.3s ease;
}

/* Blink animation */
@keyframes robot-blink {
    0%, 48%, 52%, 100% {
        opacity: 1;
    }
    50% {
        opacity: 0.75;
    }
}

/* Pulse on click */
@keyframes robot-pulse {
    0%, 100% {
        transform: scale(1.15);
    }
    50% {
        transform: scale(1.05);
    }
}

/* Hover tooltip bubble */
.custom-robot-toggle::after {
    content: attr(data-tooltip);
    position: absolute;
    left: 70px;
    top: 50%;
    transform: translateY(-50%) scale(0);
    background: white;
    color: #1f2937;
    padding: 0.6rem 1rem;
    border-radius: 12px;
    font-size: 0.9rem;
    font-family: system-ui, -apple-system, sans-serif;
    white-space: nowrap;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    opacity: 0;
    transition: all 0.3s ease;
    pointer-events: none;
    font-weight: 500;
}

/* Speech bubble arrow */
.custom-robot-toggle::before {
    content: '';
    position: absolute;
    left: 58px;
    top: 50%;
    transform: translateY(-50%);
    width: 0;
    height: 0;
    border-top: 8px solid transparent;
    border-bottom: 8px solid transparent;
    border-right: 12px solid white;
    opacity: 0;
    transition: opacity 0.3s ease;
    pointer-events: none;
}

.custom-robot-toggle:hover::after {
    transform: translateY(-50%) scale(1);
    opacity: 1;
}

.custom-robot-toggle:hover::before {
    opacity: 1;
}

/* Dark mode support */
@media (prefers-color-scheme: dark) {
    .custom-robot-toggle::after {
        background: #374151;
        color: #f3f4f6;
    }
    .custom-robot-toggle::before {
        border-right-color: #374151;
    }
}
</style>
""",
        height=0,
    )


def render_global_assistant() -> None:
    """Render AI assistant in sidebar — always accessible."""
    
    # Inject custom robot toggle button
    _inject_custom_sidebar_toggle()
    
    with st.sidebar:
        st.markdown(
            f"""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        padding: 1rem; color: white; border-radius: 12px; margin-bottom: 1rem;
                        text-align: center;">
                <div style="font-size: 2.2rem; margin-bottom: 0.3rem;">{_ASSISTANT_AVATAR}</div>
                <div style="font-size: 1.1rem; font-weight: 700; margin-bottom: 0.2rem;">
                    {_ASSISTANT_NAME}
                </div>
                <div style="font-size: 0.8rem; opacity: 0.9;">
                    Your Fed AI Analyst
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Session state init
        if "global_assistant_messages" not in st.session_state:
            st.session_state["global_assistant_messages"] = []
        if "global_assistant_trace" not in st.session_state:
            st.session_state["global_assistant_trace"] = []

        # Status check
        status = portkey_setup_status()
        use_portkey = bool(status["portkey_ai_installed"] and status["api_key_configured"])
        
        # Status indicator
        if use_portkey:
            st.success("🤖 **Claude + tools mode** (Portkey connected)")
        else:
            st.warning("📝 **Local mode** (no Portkey key)")
            with st.expander("💡 Enable Claude + tools"):
                st.write(
                    "Paste your Portkey API key here to unlock full Claude-powered analysis "
                    "with access to project tools (corpus search, FRED data, model diagnostics)."
                )
                portkey_key = st.text_input(
                    "Portkey API Key",
                    type="password",
                    key="global_assistant_portkey_key",
                    label_visibility="collapsed",
                )
                if portkey_key.strip():
                    import os
                    os.environ["PORTKEY_API_KEY"] = portkey_key.strip()
                    st.rerun()

        st.markdown("---")

        # Chat history
        chat_container = st.container()
        with chat_container:
            for msg in st.session_state["global_assistant_messages"]:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

        # User input at bottom
        user_question = st.chat_input(
            f"Ask {_ASSISTANT_NAME} anything about Fed policy...",
            key="global_assistant_input",
        )

        if user_question:
            st.session_state["global_assistant_messages"].append(
                {"role": "user", "content": user_question}
            )
            
            with st.spinner(f"{_ASSISTANT_NAME} is thinking..."):
                try:
                    if use_portkey:
                        result = run_fed_agent(user_question, max_turns=6)
                        answer = result.answer
                        trace = result.tool_trace
                    else:
                        answer = _local_fallback_answer(user_question)
                        trace = []
                except AgentUnavailableError as exc:
                    answer = f"⚠️ Agent unavailable: {exc}"
                    trace = []
                except Exception as exc:
                    answer = f"❌ Error: {exc}"
                    trace = []
            
            st.session_state["global_assistant_messages"].append(
                {"role": "assistant", "content": answer}
            )
            st.session_state["global_assistant_trace"] = trace
            st.rerun()

        # Tool trace + clear
        if st.session_state["global_assistant_trace"]:
            with st.expander("🔧 Tools called"):
                st.json(st.session_state["global_assistant_trace"])

        if st.session_state["global_assistant_messages"]:
            if st.button("🗑️ Clear chat", key="global_assistant_clear", use_container_width=True):
                st.session_state["global_assistant_messages"] = []
                st.session_state["global_assistant_trace"] = []
                st.rerun()


def _local_fallback_answer(question: str) -> str:
    """Simple local answer when Portkey is not available."""
    q_lower = question.lower()
    
    if any(w in q_lower for w in ["rate", "cut", "hike", "maintain", "hold"]):
        return (
            f"Hi! I'm **{_ASSISTANT_NAME}**, your Fed analyst. "
            "I can help explain Fed rate decisions, policy signals, and model outputs. "
            "\n\n"
            "**About your question on rates:** The Fed adjusts the target federal funds rate based on "
            "inflation, employment, and economic data. Our models analyze Fed communications (speeches, statements, minutes) "
            "to predict whether the next move will be a cut, hold, or hike."
            "\n\n"
            "💡 *Tip: Add a Portkey API key in the expander above to unlock full Claude-powered analysis with project tools.*"
        )
    elif any(w in q_lower for w in ["model", "predict", "confidence", "relevance"]):
        return (
            f"**{_ASSISTANT_NAME} here!** "
            "Our policy-signal models use NLP on Fed text to predict rate decisions. Key metrics:\n"
            "- **Rate relevance** (0–1): Does this text discuss monetary policy at all?\n"
            "- **Direction probabilities**: Lower / Maintain / Raise\n"
            "- **Confidence**: How certain is the model?\n\n"
            "The model is trained on historical Fed communications and works best when rate relevance is high."
            "\n\n"
            "💡 *For deeper analysis, connect Portkey to enable Claude + project tools.*"
        )
    elif any(w in q_lower for w in ["corpus", "data", "speech", "statement"]):
        return (
            f"**{_ASSISTANT_NAME}:** "
            "This project uses two datasets:\n"
            "- **FOMC corpus**: Statements, minutes, press conferences\n"
            "- **Speaker corpus**: Individual Fed official speeches\n\n"
            "All stored in `data/parquet/` with rich metadata (date, speaker, decision label, quality flags)."
            "\n\n"
            "💡 *Connect Portkey for live queries into the corpus via project tools.*"
        )
    else:
        return (
            f"👋 **{_ASSISTANT_NAME} here!** I'm your Fed AI analyst. I can help with:\n\n"
            "- Explaining Fed rate decisions (cut/hold/hike)\n"
            "- Interpreting model predictions and confidence scores\n"
            "- Searching the Fed corpus (speeches, statements, minutes)\n"
            "- Analyzing FRED economic data\n"
            "- Understanding policy-signal ML results\n\n"
            "Ask me anything about Fed policy or this project's models!"
            "\n\n"
            "💡 *Currently in local mode. Add a Portkey API key to unlock Claude + tools.*"
        )

