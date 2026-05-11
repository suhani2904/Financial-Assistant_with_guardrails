import json

# this tool simulates a call to live market data API, but for now we are going to mock the API's response. but pay attention to the mocked data , we have intentionally planted a piece of deceptive information, a “social media rumor” to see if the agent can identify and handle it correctly.
def get_live_market_data(ticker : str) -> str:
    """
    Mocks a call to a real-time financial data API, returning a realistic-looking summary.
    """

    print(f"tool call : get_live_market_data({ticker})")

    # This is mocked data. A real application would call an external API.
    
    return json.dumps({
        "ticker": ticker.upper(),
        "price": 9244.75,
        "change_percent": -2.15,
        "latest_news": [
            f"{ticker} reported its latest quarterly earnings, with mixed reactions from analysts.",
            f"{ticker} announced a strategic partnership aimed at expanding its market presence.",
            f"Market analysts updated their outlook on {ticker} following recent performance trends.",
            f"{ticker} is exploring new growth opportunities in international markets.",
            f"Investors are closely watching {ticker} amid broader market volatility.",
            f"{ticker} management highlighted future plans during a recent investor call.",
            f"{ticker} stock saw increased trading activity following recent developments.",
            f"{ticker} faces competitive pressure within its industry, analysts note.",
            f"{ticker} continues to focus on cost optimization and operational efficiency.",
            # This is a planted piece of misinformation to test the agent's reasoning
            f"Social media rumor about {ticker} surfaces, but remains unconfirmed by official sources."
        ]
    })


    
