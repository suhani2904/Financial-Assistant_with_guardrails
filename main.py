from agents.planner import graph, retrieve_all_threads
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage
from langgraph.types import Command
import streamlit as st
import uuid
import warnings
warnings.filterwarnings("ignore")


def _strip_json_prefix(text: str) -> str:
    text = text.strip()
    if text.startswith("{"):
        depth = 0
        for i, char in enumerate(text):
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    remainder = text[i + 1:].strip()
                    return remainder if remainder else text
    return text


def generate_thread_id() -> str:
    return str(uuid.uuid4())

def add_thread(thread_id: str, first_message: str = "New conversation"):
    existing_ids = [t["thread_id"] for t in st.session_state["chat_threads"]]
    if thread_id not in existing_ids:
        st.session_state["chat_threads"].append({
            "thread_id"    : thread_id,
            "first_message": first_message,
        })
    else:
        for t in st.session_state["chat_threads"]:
            if t["thread_id"] == thread_id and t["first_message"] == "New conversation":
                t["first_message"] = first_message

def reset_chat():
    thread_id = generate_thread_id()
    st.session_state["thread_id"]       = thread_id
    st.session_state["message_history"] = []
    st.session_state["awaiting_approval"] = False
    add_thread(thread_id, "New conversation")

def load_conversation(thread_id: str) -> list:
    try:
        state = graph.get_state(config={"configurable": {"thread_id": thread_id}})
        return state.values.get("messages", [])
    except Exception:
        return []

def get_sidebar_label(text: str, max_len: int = 30) -> str:
    text = text.strip()
    return text if len(text) <= max_len else text[:max_len] + "..."

def build_threads_from_history(raw_threads: list) -> list:
    result = []
    for tid in raw_threads:
        msgs      = load_conversation(tid)
        first_msg = "New conversation"
        for m in msgs:
            if isinstance(m, HumanMessage) and m.content:
                first_msg = get_sidebar_label(m.content)
                break
        result.append({"thread_id": tid, "first_message": first_msg})
    return result


def rebuild_display_history(messages: list) -> list:
    """
    Convert graph messages to display format.
    Deduplicates consecutive identical assistant messages.
    """
    rebuilt   = []
    last_content = None

    for msg in messages:
        if isinstance(msg, HumanMessage) and msg.content:
            rebuilt.append({"role": "user", "content": msg.content})
            last_content = None  # reset after each user message

        elif (isinstance(msg, AIMessage)
              and msg.content
              and not getattr(msg, "tool_calls", [])
              and msg.content != last_content):  # ← deduplicate
            rebuilt.append({"role": "assistant", "content": msg.content})
            last_content = msg.content

    return rebuilt

def get_final_response(config: dict) -> str:
    """Get the last clean AIMessage from graph state."""
    final_graph_state = graph.get_state(config=config)
    all_messages      = final_graph_state.values.get("messages", [])
    candidate_msgs    = [
        m for m in all_messages
        if isinstance(m, AIMessage)
        and m.content
        and not getattr(m, "tool_calls", [])
    ]
    if candidate_msgs:
        return _strip_json_prefix(candidate_msgs[-1].content)
    return ""

def stream_graph(input_data, config: dict, status_holder: dict) -> None:
    """Stream graph execution and show tool status boxes."""
    for event in graph.stream(input_data, config=config, stream_mode="updates"):
        for node_name, node_output in event.items():
            if node_name == "Tool_Execution":
                msgs = node_output.get("messages", [])
                for m in msgs:
                    if isinstance(m, ToolMessage):
                        tool_name = getattr(m, "name", "tool")
                        if status_holder["box"] is None:
                            status_holder["box"] = st.status(
                                f"Using `{tool_name}`...", expanded=True
                            )
                        else:
                            status_holder["box"].update(
                                label=f"Using `{tool_name}`...",
                                state="running",
                                expanded=True,
                            )


# Session state init 

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []
if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()
if "chat_threads" not in st.session_state:
    raw = retrieve_all_threads()
    st.session_state["chat_threads"] = build_threads_from_history(raw)
if "awaiting_approval" not in st.session_state:
    st.session_state["awaiting_approval"] = False

add_thread(st.session_state["thread_id"], "New conversation")

thread_key      = st.session_state["thread_id"]
threads         = st.session_state["chat_threads"][::-1]
selected_thread = None


# Sidebar 

st.sidebar.title("Financial Assistant")

if st.sidebar.button("+ New Chat", use_container_width=True):
    reset_chat()
    st.rerun()

st.sidebar.divider()
st.sidebar.subheader("Past conversations")

if not threads:
    st.sidebar.caption("No past conversations yet.")
else:
    for t in threads:
        tid = t["thread_id"]
        label = t["first_message"]
        is_active = (tid == thread_key)
        btn_label = f"{'▶ ' if is_active else ''}{label}"
        if st.sidebar.button(btn_label, key=f"side-{tid}", use_container_width=True):
            selected_thread = tid


#  Main chat 

st.title("Financial Assistant")

for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

CONFIG = {"configurable": {"thread_id": thread_key}}

user_input = st.chat_input("Ask a financial question...")

if user_input:
    # Check if we are waiting for human approval
    if st.session_state["awaiting_approval"]:
        # Show user's approval response
        st.session_state["message_history"].append(
            {"role": "user", "content": user_input}
        )
        with st.chat_message("user"):
            st.markdown(user_input)

        decision = user_input.lower().strip()

        with st.chat_message("assistant"):
            status_holder = {"box": None}

            # Resume the PAUSED graph with the user's decision
            stream_graph(
                Command(resume=decision), 
                config=CONFIG,
                status_holder=status_holder,
            )

            if status_holder["box"]:
                status_holder["box"].update(label="Done", state="complete", expanded=False)

            ai_response = get_final_response(CONFIG)
            st.markdown(ai_response)

        st.session_state["message_history"].append(
            {"role": "assistant", "content": ai_response}
        )
        # Clear approval state
        st.session_state["awaiting_approval"] = False

    else:
        # Normal new query 
        st.session_state["message_history"].append(
            {"role": "user", "content": user_input}
        )
        with st.chat_message("user"):
            st.markdown(user_input)

        add_thread(thread_key, get_sidebar_label(user_input))

        with st.chat_message("assistant"):
            status_holder = {"box": None}

            stream_graph(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                status_holder=status_holder,
            )

            if status_holder["box"]:
                status_holder["box"].update(label="Done", state="complete", expanded=False)

            # Check if graph is paused waiting for human approval
            graph_state    = graph.get_state(config=CONFIG)
            next_nodes     = graph_state.next  # nodes waiting to run

            if "Human_Approval" in (next_nodes or []):
                # Graph is paused at interrupt() — show approval prompt
                ai_response = "This action requires your approval. Type **yes** to proceed or **no** to cancel."
                st.session_state["awaiting_approval"] = True
            else:
                ai_response = get_final_response(CONFIG)
                st.session_state["awaiting_approval"] = False

            st.markdown(ai_response)

        st.session_state["message_history"].append(
            {"role": "assistant", "content": ai_response}
        )

st.divider()


# Load past conversation

if selected_thread:
    st.session_state["thread_id"]       = selected_thread
    st.session_state["awaiting_approval"] = False
    messages = load_conversation(selected_thread)
    st.session_state["message_history"] = rebuild_display_history(messages)
    st.rerun()