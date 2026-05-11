from typing import Dict, Any, Tuple
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnableParallel, RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
import time
import json
import re

output_guard_llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    max_tokens=100,
)

DISCLAIMER = (
    "\n\n---\n"
    "**Disclaimer:** This is for informational purposes only and does not "
    "constitute financial advice. Please consult a SEBI-registered advisor "
    "before making investment decisions."
)

# ── Check 1: Hallucination 

hallucination_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a financial response quality checker.
Evaluate if the response contains hallucinations or unverified claims.
Check for made-up stock prices, false statistics, fabricated company events,
or overconfident claims without evidence.

Respond ONLY with valid JSON:
{{"hallucination_detected": true/false, "confidence": "HIGH/MEDIUM/LOW", "reason": "brief explanation"}}"""),
    ("human", "Financial response to check:\n{response}")
])

def check_hallucination(inputs: Dict) -> Dict[str, Any]:
    response = inputs["response"]
    print("  [Hallucination Guard] Checking...")
    start = time.time()
    try:
        chain   = hallucination_prompt | output_guard_llm
        result  = chain.invoke({"response": response})
        content = result.content.strip().replace("```json", "").replace("```", "").strip()
        parsed  = json.loads(content)
        detected   = parsed.get("hallucination_detected", False)
        confidence = parsed.get("confidence", "LOW")
        reason     = parsed.get("reason", "")
        latency    = time.time() - start
        print(f"  [Hallucination Guard] Detected:{detected} Confidence:{confidence} | {latency:.2f}s")
        return {"hallucination_detected": detected, "confidence": confidence, "reason": reason, "latency": latency}
    except Exception as e:
        print(f"  [Hallucination Guard] ERROR:{e} — defaulting safe")
        return {"hallucination_detected": False, "confidence": "LOW", "reason": str(e), "latency": time.time() - start}


# ── Check 2: Unsafe Claims 

UNSAFE_PATTERNS = [
    (r"guaranteed\s+(returns?|profits?|\d+%)","guaranteed_returns"),
    (r"risk.?free\s+investment", "risk_free_claim"),
    (r"cannot\s+lose", "cannot_lose_claim"),
    (r"always\s+(go|goes|rise|rises|increase)","always_rises_claim"),
    (r"100%\s+(safe|secure|guaranteed)","100_percent_safe"),
    (r"you\s+will\s+definitely\s+(make|earn|profit)",  "profit_guarantee"),
]

unsafe_output_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a financial compliance checker.
Detect if the response contains misleading or dangerous financial claims.
Flag UNSAFE if: guaranteed returns, risk-free claims, cannot lose, always goes up,
promotion of illegal schemes, or absolute profit promises.

Respond ONLY with valid JSON:
{{"is_safe": true/false, "violation_type": "NONE or brief description", "severity": "NONE/LOW/MEDIUM/HIGH"}}"""),
    ("human", "Financial response to check:\n{response}")
])

def check_unsafe_claims(inputs: Dict) -> Dict[str, Any]:
    response = inputs["response"]
    print("  [Unsafe Claims Guard] Checking...")
    start = time.time()

    # Fast regex first
    for pattern, violation in UNSAFE_PATTERNS:
        if re.search(pattern, response, re.IGNORECASE):
            latency = time.time() - start
            print(f"  [Unsafe Claims Guard] REGEX caught: {violation} | {latency:.4f}s")
            return {"is_safe": False, "violation_type": violation, "severity": "HIGH", "method": "regex", "latency": latency}

    # LLM semantic check
    try:
        chain   = unsafe_output_prompt | output_guard_llm
        result  = chain.invoke({"response": response})
        content = result.content.strip().replace("```json", "").replace("```", "").strip()
        parsed  = json.loads(content)
        is_safe        = parsed.get("is_safe", True)
        violation_type = parsed.get("violation_type", "NONE")
        severity       = parsed.get("severity", "NONE")
        latency        = time.time() - start
        print(f"  [Unsafe Claims Guard] Safe:{is_safe} | {latency:.2f}s")
        return {"is_safe": is_safe, "violation_type": violation_type, "severity": severity, "method": "llm", "latency": latency}
    except Exception as e:
        print(f"  [Unsafe Claims Guard] ERROR:{e} — defaulting safe")
        return {"is_safe": True, "violation_type": "NONE", "severity": "NONE", "method": "error_fallback", "latency": time.time() - start}


# ── Check 3: Output PII 

def check_output_pii(inputs: Dict) -> Dict[str, Any]:
    response = inputs["response"]
    print("  [Output PII Guard] Scanning...")
    start = time.time()

    PII_PATTERNS = {
        "account_number": r'\b(ACCT|ACCOUNT)[- ]?(\d{3}[- ]?){2}\d{4}\b',
        "credit_card"   : r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b',
        "phone_number"  : r'\b(\+91|0)?[6-9]\d{9}\b',
        "email"         : r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        "pan_number"    : r'\b[A-Z]{5}[0-9]{4}[A-Z]{1}\b',
        "aadhar"        : r'\b\d{4}\s\d{4}\s\d{4}\b',
        "ssn"           : r'\b\d{3}-\d{2}-\d{4}\b',
    }

    pii_found = False
    pii_types = []
    redacted  = response

    for pii_type, pattern in PII_PATTERNS.items():
        if re.search(pattern, response, re.IGNORECASE):
            redacted  = re.sub(pattern, f"[REDACTED_{pii_type.upper()}]", redacted, flags=re.IGNORECASE)
            pii_found = True
            pii_types.append(pii_type)

    latency = time.time() - start
    print(f"  [Output PII Guard] PII:{pii_found} {pii_types} | {latency:.4f}s")
    return {"pii_found": pii_found, "pii_types": pii_types, "redacted_response": redacted, "latency": latency}


# ── RunnableParallel

output_guardrail_chain = RunnableParallel(
    hallucination_check = RunnableLambda(check_hallucination),
    unsafe_claims_check = RunnableLambda(check_unsafe_claims),
    pii_leak_check      = RunnableLambda(check_output_pii),
)


def run_output_guardrails(response: str) -> Tuple[bool, str]:
    """
    Run all 3 output guardrails IN PARALLEL using RunnableParallel.

    Returns:
        (is_allowed: bool, final_response: str)
    """
    print("\nOUTPUT GUARDRAILS (IN PARALLEL) ")
    start   = time.time()
    results = output_guardrail_chain.invoke({"response": response})
    latency = time.time() - start
    print(f"OUTPUT GUARDRAILS COMPLETE. Latency: {latency:.2f}s")

    is_allowed        = True
    rejection_reasons = []

    # Check 1 — Hallucination (only block HIGH confidence)
    hall = results["hallucination_check"]
    if hall.get("hallucination_detected") and hall.get("confidence") == "HIGH":
        is_allowed = False
        rejection_reasons.append(f"Hallucination detected: {hall.get('reason', '')}")

    # Check 2 — Unsafe claims
    unsafe = results["unsafe_claims_check"]
    if not unsafe.get("is_safe"):
        is_allowed = False
        rejection_reasons.append(
            f"Unsafe claim [{unsafe.get('severity')}]: {unsafe.get('violation_type')}"
        )

    # Check 3 — PII (redact but don't block)
    pii = results["pii_leak_check"]
    if pii.get("pii_found"):
        response = pii.get("redacted_response", response)
        print(f"  PII redacted from response: {pii.get('pii_types')}")

    if is_allowed:
        final = response + DISCLAIMER
        print("VERDICT: RESPONSE ALLOWED — Disclaimer injected")
    else:
        final = (
            "Response blocked by output safety guardrails.\n\n"
            + "\n".join(f"- {r}" for r in rejection_reasons)
        )
        print("VERDICT: RESPONSE BLOCKED")

    return is_allowed, final