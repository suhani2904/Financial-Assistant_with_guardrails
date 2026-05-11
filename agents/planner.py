# agents/planner.py

from tools.calculator import calculator
from tools.demo_live_market_data import get_live_market_data
from tools.knowledge_base import query_rag
from tools.take_action import take_action
from Guardrails.input_guardrails import run_input_guardrails
from Guardrails.output_guardrails import run_output_guardrails
from Guardrails.action_guardrails import run_action_guardrails

from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from pydantic import BaseModel, Field
from typing import TypedDict, List, Any, Literal, Annotated, Dict
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver
import json
from langgraph.types import interrupt

load_dotenv()

class KnowledgeBaseInput(BaseModel):
    ticker: str = Field(description="Stock ticker symbol e.g. NVDA, AAPL, MSFT, TSLA")
    query:  str = Field(description="Question about company financials, earnings, risks, or products")

class TradingInput(BaseModel):
    ticker:      str = Field(description="Stock ticker e.g. NVDA, AAPL")
    shares:      int = Field(description="Number of shares to buy or sell")
    action_type: Literal["buy", "sell"] = Field(description="Trading action: buy or sell")

class calculator_input(BaseModel):
    expression: str = Field(description="Mathematical expression to evaluate, e.g. '10000 * 1.07 ** 10 , english words allowed'")



@tool(args_schema=calculator_input)
def calculator_tool(expression: str) -> str:
    """Evaluate a math expression for financial calculations. Example: '10000 * 1.07 ** 10'"""
    return calculator(expression)

@tool
def market_data_tool(ticker: str) -> str:
    """Get real-time stock price and latest news for a ticker like NVDA or AAPL."""
    return get_live_market_data(ticker)

@tool(args_schema=KnowledgeBaseInput)
def query_knowledge_base_tool(ticker: str, query: str) -> str:
    """Search the SEC 10-K annual report knowledge base for a specific company."""
    return str(query_rag(ticker, query))

@tool(args_schema=TradingInput)
def trading_action_tool(ticker: str, shares: int, action_type: Literal["buy", "sell"]) -> str:
    """Execute a simulated buy or sell trade for a stock."""
    return take_action(ticker, shares, action_type)

# Tool registry — maps tool name → callable
TOOL_MAP = {
    "calculator_tool"           : calculator_tool,
    "market_data_tool"          : market_data_tool,
    "query_knowledge_base_tool" : query_knowledge_base_tool,
    "trading_action_tool"       : trading_action_tool,
}


# Use llama-3.1-8b for plan generation 
plan_llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    max_tokens=1024,
)

# Use 70b only for final response generation and rejections
response_llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    max_tokens=1024,
)

rejection_llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    max_tokens=300,
)

tools = [calculator_tool, market_data_tool, query_knowledge_base_tool, trading_action_tool]



class AgentState(TypedDict):
    messages : Annotated[List[Any], add_messages]
    is_allowed : bool
    output_blocked: bool
    action_plan : List[Dict]          # generated plan steps
    action_approved : bool                # did guardrails approve?
    action_block_reason  : str                 # why blocked
    tool_results  : List[Dict]          # results from tool execution
    human_approval_required : bool       # does this plan require human approval?   



PLANNING_PROMPT = """You are a financial planning assistant with memory of the full conversation.

Given the conversation history and the latest user request, create a step-by-step 
plan of tool calls needed.

IMPORTANT: Use context from previous messages.
- If the user says "what about Apple?" after discussing NVDA, plan for AAPL.
- If the user says "sell those shares", use the ticker and quantity from history.
- If the user asks a follow-up question, use prior context to answer.

Available tools:
- calculator_tool: for math. args: {"expression": "..."}
- market_data_tool: for stock data. args: {"ticker": "NVDA"}
- query_knowledge_base_tool: for SEC filings. args: {"ticker": "NVDA", "query": "..."}
- trading_action_tool: for trades. args: {"ticker": "NVDA", "shares": 10, "action_type": "buy"}

Respond ONLY with valid JSON:
{"plan": [{"tool_name": "...", "tool_args": {...}, "reasoning": "why this step"}]}

If no tools are needed, respond with: {"plan": []}"""


RESPONSE_PROMPT = """You are a helpful financial assistant with full memory of this conversation.

You remember:
- Everything the user has asked before
- All data and prices you have fetched
- Any decisions or context from earlier in the conversation

Use this memory to give coherent, contextual responses.
Never ask for information the user already provided.
If the user refers to "it" or "that stock", resolve it from the conversation history."""

def _build_history_text(messages: list, max_messages: int = 12) -> str:
    """Build readable conversation history from recent messages."""
    relevant = [
        m for m in messages
        if isinstance(m, (HumanMessage, AIMessage))
        and m.content
        and not getattr(m, "tool_calls", [])
    ][-max_messages:]

    lines = []
    for m in relevant:
        if isinstance(m, HumanMessage):
            lines.append(f"User: {m.content}")
        elif isinstance(m, AIMessage):
            # Trim disclaimer from history to save tokens
            content = m.content
            if "---" in content:
                content = content[:content.rfind("---")].strip()
            lines.append(f"Assistant: {content[:300]}")

    return "\n".join(lines) if lines else "No prior conversation."



def input_guardrail_node(state: AgentState) -> Dict:
    """Layer 1: Validate the user query before anything else runs."""
    prompt  = state["messages"][-1].content
    results = run_input_guardrails(prompt)

    is_allowed        = True
    rejection_reasons = []

    topic = results["topic_check"].get("topic", "UNKNOWN")
    if topic not in ["FINANCE_INVESTING"]:
        is_allowed = False
        rejection_reasons.append(f"Off-topic query. Topic detected: {topic}")

    if not results["threat_check"].get("is_safe", True):
        is_allowed = False
        violations = results["threat_check"].get("policy_violations", [])
        rejection_reasons.append(f"Threat detected: {violations}")

    if results["sensitive_data_check"].get("pii_found"):
        is_allowed = False
        pii_types  = results["sensitive_data_check"].get("pii_types", [])
        rejection_reasons.append(f"PII detected: {pii_types}")

    if results["sensitive_data_check"].get("mnpi_risk"):
        is_allowed = False
        rejection_reasons.append("MNPI risk detected")

    if not is_allowed:
        rejection_text = "\n".join(f"- {r}" for r in rejection_reasons)
        safe_msg = rejection_llm.invoke([
            SystemMessage(content=(
                "You are a polite financial safety assistant. "
                "Explain why the request was blocked in 2-3 plain English bullet points. "
                "Do NOT output JSON. Start each bullet with a dash (-)."
            )),
            HumanMessage(content=f"Blocked reasons:\n{rejection_text}\n\nQuery: {prompt}")
        ])
        print(f"INPUT GUARDRAIL: REJECTED")
        return {
            "is_allowed" : False,
            "output_blocked": False,
            "action_approved": False,
            "tool_results"  : [],
            "messages": [AIMessage(content=safe_msg.content.strip())],
        }

    print("INPUT GUARDRAIL: ALLOWED")
    return {
        "is_allowed"    : True,
        "output_blocked": False,
        "action_approved": False,
        "tool_results"  : [],
    }


def planning_node(state: AgentState) -> Dict:
    print("\n--- PLANNING NODE ---")

    messages     = state["messages"]
    user_request = ""
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            user_request = m.content
            break

    history_text = _build_history_text(messages)

    try:
        response = plan_llm.invoke([
            SystemMessage(content=PLANNING_PROMPT),
            HumanMessage(content=(
                f"Conversation history:\n{history_text}\n\n"
                f"Latest request: {user_request}"
            )),
        ])
        content = response.content.strip().replace("```json","").replace("```","").strip()
        parsed  = json.loads(content)
        plan    = parsed.get("plan", [])
        print(f"  Plan: {len(plan)} step(s)")
        for i, s in enumerate(plan):
            print(f"  Step {i+1}: {s.get('tool_name')} — {s.get('reasoning','')[:60]}")
        return {"action_plan": plan}
    except Exception as e:
        print(f"  [Planning] ERROR: {e}")
        return {"action_plan": []}

def action_guardrails_node(state: AgentState) -> Dict:
    """
    Layer 2: Run groundedness + human escalation + policy on the plan.
    If plan is empty (no tools needed) → auto-approve.
    """
    plan     = state.get("action_plan", [])
    messages = state["messages"]

    # No tools needed — skip guardrails
    if not plan:
        print("  No tools in plan — skipping action guardrails")
        return {"action_approved": True, "action_block_reason": ""}

    print("\n--- ACTION GUARDRAILS NODE ---")
    is_approved, block_reason , human_approval= run_action_guardrails(
        plan=plan,
        conversation_history=messages,
    )

    if human_approval:
        print("ACTION PLAN: REQUIRES HUMAN APPROVAL")
        if human_approval:
            print("ACTION PLAN: REQUIRES HUMAN APPROVAL")
            return {
                "action_approved": False,
                "human_approval_required": True,
                "action_block_reason": "This action requires your approval (Yes/No).",
                "messages": [
                    AIMessage(content="This action may be risky. Do you want to proceed? (yes/no)")
                ]
            }


    elif not is_approved:
        block_msg = "Your request was reviewed and could not be executed.\n\n" + block_reason
        print("ACTION PLAN: BLOCKED")
        return {
            "human_approval_required": False,
            "action_approved" : False,
            "action_block_reason": block_reason,
            "messages" : [AIMessage(content=block_msg)],
        }

    print("ACTION PLAN: APPROVED")
    return {
        "human_approval_required": False,
        "action_approved": True,
        "action_block_reason": "",
    }


def human_approval_node(state: AgentState) -> Dict:
    """
    Pause graph and wait for human input via interrupt().
    The graph freezes here until Command(resume=decision) is sent.
    """
    print("\n--- HUMAN APPROVAL NODE: Waiting for human input ---")

    # interrupt() pauses the graph and returns the resume value
    # when Command(resume=value) is called from Streamlit
    decision = interrupt("Approve this trading action? Reply yes or no.")

    print(f"Human decision received: {decision}")
    decision = str(decision).lower().strip()

    if decision in ["yes", "y"]:
        print("User approved action")
        return {
            "action_approved"        : True,
            "human_approval_required": False,
        }
    else:
        print("User denied action")
        return {
            "action_approved": False,
            "human_approval_required": False,
            "messages" : [AIMessage(content="Action cancelled as per your decision. No trade was executed.")],
        }

def tool_execution_node(state: AgentState) -> Dict:
    """
    Execute each tool in the approved plan directly.
    No LLM involved here — pure deterministic execution.
    Results stored in tool_results for Response_Generation to use.
    """
    print("\n--- TOOL EXECUTION NODE ---")
    plan  = state.get("action_plan", [])
    tool_results = []

    for step in plan:
        tool_name = step.get("tool_name", "")
        tool_args = step.get("tool_args", {})
        reasoning = step.get("reasoning", "")

        print(f"Executing: {tool_name}({tool_args})")

        if tool_name not in TOOL_MAP:
            result = f"Error: Unknown tool '{tool_name}'"
            print(f"  Result: {result}")
        else:
            try:
                if tool_name == "calculator_tool":
                    expr = tool_args.get("expression", "")

                    # Replace placeholder with actual price
                    for prev in tool_results:
                        if prev["tool_name"] == "market_data_tool":
                            data = json.loads(prev["result"])
                            price = data.get("price")

                            if "current_price" in expr:
                                expr = expr.replace("current_price", str(price))

                    tool_args["expression"] = expr
                result = TOOL_MAP[tool_name].invoke(tool_args)
                print(f"  Result: {str(result)[:120]}...")
            except Exception as e:
                result = f"Tool error: {str(e)}"
                print(f"  Error: {result}")

        tool_results.append({
            "tool_name": tool_name,
            "tool_args": tool_args,
            "reasoning": reasoning,
            "result"   : str(result),
        })

        # Add as ToolMessage so it's visible in message history
        messages_to_add = [
            ToolMessage(
                content = str(result),
                tool_call_id = f"{tool_name}_{len(tool_results)}",
                name = tool_name,
            )
        ]

    return {
        "tool_results": tool_results,
        "messages": messages_to_add if tool_results else [],
    }

def response_generation_node(state: AgentState) -> Dict:
    print("\n--- RESPONSE GENERATION NODE ---")

    messages     = state["messages"]
    tool_results = state.get("tool_results", [])
    history_text = _build_history_text(messages)

    user_request = ""
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            user_request = m.content
            break

    if tool_results:
        results_text = "\n\n".join([
            f"Tool: {r['tool_name']}\nArgs: {r['tool_args']}\nResult: {r['result']}"
            for r in tool_results
        ])
        response = response_llm.invoke([
            SystemMessage(content=RESPONSE_PROMPT),
            HumanMessage(content=(
                f"Conversation history:\n{history_text}\n\n"
                f"Latest request: {user_request}\n\n"
                f"Tool results:\n{results_text}\n\n"
                f"Write a clear, contextual response using the conversation history."
            ))
        ])
    else:
        response = response_llm.invoke([
            SystemMessage(content=RESPONSE_PROMPT),
            HumanMessage(content=(
                f"Conversation history:\n{history_text}\n\n"
                f"Latest request: {user_request}\n\n"
                f"Answer using the full conversation context."
            ))
        ])

    print("  Response generated.")
    return {"messages": [AIMessage(content=response.content.strip())]}



def output_guardrail_node(state: AgentState) -> Dict:
    """Layer 3: Validate agent's final response before showing to user."""
    last_ai_msg = next(
        (m for m in reversed(state["messages"])
         if isinstance(m, AIMessage) and m.content
         and not getattr(m, "tool_calls", [])),
        None
    )

    if not last_ai_msg:
        return {"output_blocked": False}

    is_safe, final = run_output_guardrails(last_ai_msg.content)
    print(f"OUTPUT GUARDRAIL: {'SAFE' if is_safe else 'BLOCKED'}")

    if is_safe:
        state["messages"][-1] = AIMessage(content=final) 
        return {"output_blocked": False}

    return {
        "output_blocked": not is_safe,
        "messages" : [AIMessage(content=final)],
    }


def route_after_input(state: AgentState) -> Literal["Planning", "__end__"]:
    if state["is_allowed"]:
        return "Planning"
    return "__end__"


def route_after_action_guard(state: AgentState) -> Literal["Tool_Execution", "Human_Approval", "__end__"]:
    if state.get("human_approval_required", False):
        return "Human_Approval"

    if state.get("action_approved", False):
        return "Tool_Execution"

    return "__end__"

def route_after_human(state: AgentState) -> Literal["Tool_Execution", "__end__"]:
    if state.get("action_approved", False):
        return "Tool_Execution"
    return "__end__"

conn = sqlite3.connect("chatbot.db", check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)



def _build():
    wf = StateGraph(AgentState)

   
    wf.add_node("Input_Guardrails", input_guardrail_node)
    wf.add_node("Planning", planning_node)
    wf.add_node("Action_Guardrails", action_guardrails_node)
    wf.add_node("Tool_Execution", tool_execution_node)
    wf.add_node("Response_Generation", response_generation_node)
    wf.add_node("Output_Guardrails", output_guardrail_node)
    wf.add_node("Human_Approval", human_approval_node)

   
    wf.add_edge(START,"Input_Guardrails")

    wf.add_conditional_edges("Input_Guardrails",route_after_input,
        {"Planning": "Planning", "__end__": END})

    

    wf.add_conditional_edges("Action_Guardrails",route_after_action_guard,
        {"Human_Approval": "Human_Approval",
        "Tool_Execution": "Tool_Execution",
        "__end__": END})

    wf.add_conditional_edges("Human_Approval",route_after_human,
        {"Tool_Execution": "Tool_Execution",
            "__end__": END})
    
    wf.add_edge("Planning","Action_Guardrails")

    wf.add_edge("Tool_Execution","Response_Generation")
    wf.add_edge("Response_Generation", "Output_Guardrails")
    wf.add_edge("Output_Guardrails",END)

    return wf.compile(checkpointer=checkpointer)

graph = _build()

def retrieve_all_threads() -> list:
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])
    return list(all_threads)