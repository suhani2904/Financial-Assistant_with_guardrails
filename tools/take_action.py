import json
from typing import Literal

def take_action(ticker: str, shares: int, action_type: Literal["buy", "sell"]) -> str:
    """
    Simulates executing a stock trade.
    In production this would call a brokerage API.
    """
    print(f"tool call : take_action(ticker={ticker}, shares={shares}, action_type={action_type})")
    return json.dumps({
        "ticker"    : ticker.upper(),
        "shares"    : shares,
        "action"    : action_type,
        "status"    : "success",
        "message"   : f"Simulated {action_type} of {shares} shares of {ticker.upper()}.",
    })