from typing import Dict, Any, List, Tuple
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnableParallel, RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
import json
from tools.demo_live_market_data import get_live_market_data

guard_llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    max_tokens=200,
)

MAJOR_EXCHANGES = {
    "NVDA","AAPL","MSFT","GOOGL","GOOG","AMZN","META","TSLA",
    "NFLX","AMD","INTC","QCOM","ADBE","CRM","JPM","BAC","GS",
    "MS","WFC","SPY","QQQ","DIA","IWM",
    "RELIANCE.NS","TCS.NS","INFY.NS","HDFCBANK.NS","WIPRO.NS",
}

# ── Guard A: Groundedness 

GROUNDEDNESS_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a financial reasoning auditor.
Evaluate whether each step in the action plan is logically grounded
in the conversation history provided.

Flag a step as HALLUCINATED if:
- It references facts not mentioned in the conversation
- It assumes information the agent has not gathered yet
- The reasoning is circular or fabricated

Respond ONLY with valid JSON, no trailing commas:
{{"overall_grounded": true, "confidence": "HIGH/MEDIUM/LOW", "issues": []}}"""),
    ("human", "Conversation:\n{conversation}\n\nPlan:\n{plan}")
])

def check_groundedness(inputs: Dict) -> Dict[str, Any]:
    print("  [Groundedness Guard] Checking...")
    plan = inputs.get("plan", [])
    conversation = inputs.get("conversation", "")
    try:
        chain   = GROUNDEDNESS_PROMPT | guard_llm
        result  = chain.invoke({"conversation": conversation, "plan": json.dumps(plan, indent=2)})
        content = result.content.strip().replace("```json","").replace("```","").strip()
        parsed  = json.loads(content)
        overall_grounded = parsed.get("overall_grounded", True)
        confidence = parsed.get("confidence", "LOW")
        issues= parsed.get("issues", [])
        print(f"  [Groundedness Guard] Grounded:{overall_grounded} | Confidence:{confidence}")
        return {"overall_grounded": overall_grounded, "confidence": confidence, "issues": issues}
    except Exception as e:
        print(f"  [Groundedness Guard] ERROR:{e} — defaulting grounded")
        return {"overall_grounded": True, "confidence": "LOW", "issues": []}


# ── Guard B: Human Escalation 

def check_human_escalation(inputs: Dict) -> Dict[str, Any]:
    print("  [Human Escalation Guard] Checking...")
    plan = inputs.get("plan", [])


    for step in plan:
        if step.get("tool_name") == "trading_action_tool":

            tool_args = step.get("tool_args", {})
            ticker = str(tool_args.get("ticker", "")).upper().strip()
            shares = tool_args.get("shares", 0)
            data = json.loads(get_live_market_data(ticker))
            current_price = data.get("price")
            trade_value = current_price * shares

            if current_price and trade_value > 5000:
                return {"requires_human_approval": True, "triggers_fired": ["Trading action detected"]}
        
    return {"requires_human_approval": False, "triggers_fired": []}

# ── Guard C: Policy Compliance 

POLICY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """
You are a trading compliance officer responsible for detecting HIGH-LEVEL policy risks.

IMPORTANT:
- Do NOT re-check numerical or deterministic rules (e.g., trade value, % change, ticker validity).
- These are already validated by the system.

Your job is ONLY to detect:
1. Suspicious or manipulative intent
2. Attempts to bypass trading policies
3. Panic-driven or emotionally risky decisions
4. Requests involving insider information or unethical trading
5. Ambiguous or unclear trading instructions
6. If the company is fake company not listed anywhere  

Analyze the user's intent and the action plan.

Return STRICT JSON:
{{
  "compliant": true/false,
  "violations": [
    {{
      "policy_violated": "<short label>",
      "reason": "<clear explanation>"
    }}
  ]
}}

If no issues → compliant = true
"""),

    ("human", """Action Plan: {plan} 
Conversation History: {conversation}""")
])
def check_policy_compliance(inputs: Dict) -> Dict[str, Any]:
    print("[Policy Guard] Checking compliance...")

    plan = inputs.get("plan", [])
    violations = []
    compliant = True

    policy_3 = False

    for step in plan:
        if step.get("tool_name") != "trading_action_tool":
            continue

        tool_args = step.get("tool_args", {})

        ticker = str(tool_args.get("ticker", "")).upper().strip()
        shares = tool_args.get("shares", 0)
        action_type = str(tool_args.get("action_type", "")).lower()

        try:
            shares = int(shares)
        except:
            shares = 0


        current_price = None
        if ticker:
            try:
                data = json.loads(get_live_market_data(ticker))
                current_price = data.get("price")
            except Exception as e:
                print(f"  [Policy Guard] Market data fetch failed: {e}")

    
        if current_price and shares > 0:
            trade_value = current_price * shares

            if trade_value > 10000:
                compliant = False
                violations.append({
                    "policy_violated": "Policy 1 — Trade value limit ($10,000)",
                    "reason": f"{shares} × ${current_price:.2f} = ${trade_value:,.2f} exceeds limit"
                })

        # Policy 2 — SELL on >5% drop
        if action_type == "sell" and ticker:
            try:
                data = json.loads(get_live_market_data(ticker))
                change_percent = data.get("change_percent")

                if change_percent is not None and change_percent <= -5.0:
                    compliant = False
                    violations.append({
                        "policy_violated": "Policy 2 — No SELL on >5% intraday drop",
                        "reason": (
                            f"{ticker} is down {change_percent:.2f}% in current session. "
                            f"SELL orders are blocked under policy."
                        ),
                    })

            except Exception as e:
                print(f"  [Policy Guard] Market data fetch failed: {e}")

        # Policy 3 — Major exchange only
        if ticker and ticker not in MAJOR_EXCHANGES:
            compliant = False
            policy_3 = True

    # LLM semantic check — catches what regex misses
    
    try:
        conversation = inputs.get("conversation", "")
        chain  = POLICY_PROMPT | guard_llm
        result = chain.invoke({
            "plan" : plan,
            "conversation": conversation
        })
        content = result.content.strip().replace("```json","").replace("```","").strip()
        parsed  = json.loads(content)
        if not parsed.get("compliant", True):
            compliant = False
            violations.extend(parsed.get("violations", []))
    except Exception as e:
        print(f"  [Policy Guard] LLM check error: {e}")

    print(f"[Policy Guard] Compliant:{compliant} | Violations:{len(violations)}")
    for v in violations:
        print(f"{v['policy_violated']}: {v['reason']}")

    return {"compliant": compliant, "violations": violations , "policy_3_violation": policy_3}


# ── RunnableParallel

action_guardrail_chain = RunnableParallel(
    groundedness_check = RunnableLambda(check_groundedness),
    escalation_check   = RunnableLambda(check_human_escalation),
    policy_check = RunnableLambda(check_policy_compliance),
)


def run_action_guardrails(
    plan: List[Dict],
    conversation_history: List,
) -> Tuple[bool, str]:
    """
    Run all 3 action guardrails in parallel.
    Returns: (is_approved: bool, block_reason: str)
    """
    print("\n>>> ACTION GUARDRAILS (IN PARALLEL) <<<")

    # Build conversation text
    conv_text = ""
    for m in conversation_history:
        if isinstance(m, HumanMessage):
            conv_text += f"User: {m.content}\n"
        elif isinstance(m, AIMessage) and m.content:
            conv_text += f"Assistant: {m.content}\n"

    inputs  = {"plan": plan, "conversation": conv_text}
    results = action_guardrail_chain.invoke(inputs)

    print(">>> ACTION GUARDRAILS COMPLETE <<<")

    is_approved  = True
    block_parts  = []
    human_approval = False

    # Check A — Groundedness
    ground = results["groundedness_check"]
    if not ground.get("overall_grounded") and ground.get("confidence") == "HIGH":
        is_approved = False
        issues  = ground.get("issues", [])
        block_parts.append(
            "**Hallucinated reasoning detected:**\n"
            + "\n".join(f"- {i}" for i in issues)
        )


    # Check B — Human escalation
    escalation = results["escalation_check"]
    if escalation.get("requires_human_approval"):
        is_approved = False
        human_approval = True
        triggers = escalation.get("triggers_fired", [])
        block_parts.append(
            "**Human approval required before proceeding:**\n"
            + "\n".join(f"- {t}" for t in triggers))

    # Check C — Policy

    policy = results["policy_check"]
    if policy.get("policy_3_violation"):
        is_approved = False
        block_parts.append(
            "**Policy 3 violation:** Trade involves ticker not listed on major exchanges. "
            "Please verify ticker and ensure it is listed on a major exchange."
        )
        human_approval = False

    if not policy.get("compliant"):
        is_approved = False
        violations  = policy.get("violations", [])
        block_parts.append(
            "**Policy violation detected:**\n"
            + "\n".join(
                f"- {v.get('policy_violated','')}: {v.get('reason','')}"
                for v in violations
            )
        )

    block_reason = "\n\n".join(block_parts)

    return is_approved, block_reason , human_approval