import streamlit as st
import requests
import json
from openai import OpenAI

# Load API keys from secrets.toml
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
NEWS_API_KEY = st.secrets["NEWS_API_KEY"]

client = OpenAI(api_key=OPENAI_API_KEY)

# -----------------------------------------------------------
# 🧠 AGENT DEFINITIONS (Helper Functions)
# -----------------------------------------------------------

# AGENT 1: Claim Extractor
def extract_claim_agent(text):
    """Extract the main factual claim from the input text."""
    prompt = f"Extract the main factual claim from this text: '{text}'. Return only the main claim, no explanation. Be concise."
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

# AGENT 2: Historical Detector
def detect_historical_agent(claim):
    """Decide if a claim is historical or current."""
    prompt = f"""
Determine if the following claim refers to a historical or past event (something that already happened and is widely known in history)
or if it's about a current or recent event needing up-to-date news.

Claim: "{claim}"

Respond only with one word: "historical" or "current".
"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    answer = response.choices[0].message.content.strip().lower()
    return "historical" in answer

# AGENT 3: Historical Verifier (Used when current news is not needed)
def verify_historical_agent(claim):
    """Verify historically known claims directly with GPT and demand an explanation."""
    prompt = f"""
Fact-check the following historical claim. You must provide a brief explanation
and a final verdict.

Claim: '{claim}'

Format your response strictly as a JSON object with two keys:
1. 'verdict': (True, False, or Uncertain)
2. 'explanation': (A short, factual summary based on your knowledge)
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        data = json.loads(response.choices[0].message.content.strip())
        return data

    except Exception as e:
        print(f"Historical verification error: {e}")
        return {"verdict": "Uncertain", "explanation": "Failed to get structured historical verification."}


# AGENT 4: News Searcher (Uses NewsAPI)
def search_news_agent(claim):
    """Search relevant news articles using NewsAPI."""
    # Search for the claim directly for highest relevance
    url = f"https://newsapi.org/v2/everything?q={claim}&language=en&sortBy=relevancy&apiKey={NEWS_API_KEY}"
    response = requests.get(url)
    data = response.json()
    articles = data.get("articles", [])
    # Limit to top 5 articles
    return articles[:5]


# AGENT 5: Tone and Source Credibility Scorer (Rule-Based + LLM)
def score_credibility_agent(text, articles):
    """Evaluate source domains and text tone."""
    trusted_sources = [
        "bbc.com", "reuters.com", "apnews.com", "nytimes.com", "theguardian.com",
        "indiatimes.com", "ndtv.com", "hindustantimes.com", "cnn.com", "aljazeera.com"
    ]

    # --- A. Source Score (Rule-Based) ---
    source_score = 0
    for article in articles:
        url = article.get("url", "")
        if any(src in url for src in trusted_sources):
            source_score += 20
        else:
            # Gives a small score for general articles if they are not known clickbait
            source_score += 5
    source_score = min(source_score, 100)


    # --- B. Tone Score (LLM-based) ---
    prompt = f"Rate the sensationalism of this original user statement (1=neutral/objective, 100=hyperbolic/sensational): '{text}'. Give only a number."
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    try:
        sensationalism_score = int(response.choices[0].message.content.strip())
        tone_score = 100 - sensationalism_score  # Higher = more trustworthy tone
    except:
        tone_score = 50 # Default if model gives weird output

    return source_score, tone_score


# AGENT 6: Evidence Synthesis Agent
def evidence_synthesis_agent(claim, articles):
    """
    Analyzes article snippets against the claim to find support or contradiction.
    This creates a deeper understanding than just checking domains.
    """
    # Create a concise summary of the evidence
    evidence_list = []
    for i, article in enumerate(articles):
        evidence_list.append(f"Source {i+1} ({article.get('source', {}).get('name', 'Unknown')}): Title='{article.get('title')}', Description='{article.get('description')}'")

    evidence_str = "\n---\n".join(evidence_list)

    prompt = f"""
The original claim is: "{claim}"

Analyze the following evidence snippets from various news sources:
---
{evidence_str}
---

Determine if the evidence:
1. **Strongly supports** the claim.
2. **Weakly supports** the claim (only circumstantial or limited evidence).
3. **Contradicts** the claim.
4. Is **Insufficient** to make a judgment.

Respond only with one of the four options: Strongly Supports, Weakly Supports, Contradicts, or Insufficient.
"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()


# AGENT 7: Final Reasoner Agent
def final_reasoner_agent(claim, synthesis_result, source_score, tone_score, articles):
    """The final decision-maker. Formulates the final verdict and explanation."""

    # Map synthesis to a weight
    synthesis_weight = 0
    if "strongly supports" in synthesis_result.lower():
        synthesis_weight = 100
    elif "weakly supports" in synthesis_result.lower():
        synthesis_weight = 70
    elif "insufficient" in synthesis_result.lower():
        synthesis_weight = 40
    elif "contradicts" in synthesis_result.lower():
        synthesis_weight = 10

    # Combine all factors (Source Credibility (40%) + Tone (20%) + Evidence Synthesis (40%))
    truth_score = round(
        (0.4 * source_score) +
        (0.2 * tone_score) +
        (0.4 * synthesis_weight)
    )

    if truth_score > 85 and synthesis_result == "Strongly Supports":
        verdict = "✅ Verified (High Confidence)"
    elif truth_score > 60 and "supports" in synthesis_result.lower():
        verdict = "⚠️ Possibly True / Needs Further Review"
    elif synthesis_result == "Contradicts":
        verdict = "❌ Likely Fake (Contradictory Evidence Found)"
    else:
        verdict = "❓ Unverified (Insufficient Evidence or Low Trust)"

    # Generate a final explanatory summary
    prompt = f"""
Based on the following analysis, write a concise 2-3 sentence summary explaining the final verdict:

- Claim: {claim}
- Source Credibility Score (out of 100): {source_score}
- Original Text Tone Score (out of 100, higher is better): {tone_score}
- Evidence Synthesis Result: {synthesis_result}
- Final Truth Score: {truth_score}

Start the summary with the final verdict.
"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    explanation = response.choices[0].message.content.strip()

    return {
        "verdict": verdict,
        "score": truth_score,
        "explanation": explanation,
        "synthesis": synthesis_result,
        "articles": articles
    }

# AGENT 8: No-Evidence Reasoner Agent
def no_evidence_reasoner_agent(claim):
    """Provides potential reasons why no news articles were found for the claim."""
    prompt = f"""
No news articles could be found via search for the following claim: '{claim}'.
Based on the content of the claim, provide a brief (1-2 sentence) explanation suggesting the most likely reason why this might be the case.
Consider if the claim is too niche, too local, too old (but not historical), or possibly fabricated.
Start the explanation with a phrase like 'This claim could not be verified because...'
"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()


# -----------------------------------------------------------
# 🛠️ ORCHESTRATOR FUNCTION
# -----------------------------------------------------------

def verify_claim_orchestrator(text):
    """Orchestrates the entire multi-agent verification process."""
    st.session_state.messages.append({"role": "system", "content": "Starting verification..."})

    # --- STEP 1: Extraction & Classification ---
    claim = extract_claim_agent(text)
    is_historical = detect_historical_agent(claim)

    # --- STEP 2: Branching for Historical Claims ---
    if is_historical:
        st.session_state.messages.append({"role": "system", "content": "Classified as historical. Using internal verification agent."})
        hist_result = verify_historical_agent(claim)
        
        verdict = "✅ Verified (High Confidence)" if hist_result['verdict'] == 'True' else "❌ Likely Fake" if hist_result['verdict'] == 'False' else "⚠️ Uncertain"
        score = 100 if hist_result['verdict'] == 'True' else 0 if hist_result['verdict'] == 'False' else 50
        
        return {
            "verdict": verdict,
            "score": score,
            "claim": claim,
            "explanation": f"Historical Verdict: {hist_result.get('explanation', 'No explanation provided.')}",
            "sources": []
        }

    # --- STEP 3: Current Claim Research and Scoring ---
    st.session_state.messages.append({"role": "system", "content": "Classified as current. Starting news research."})
    articles = search_news_agent(claim)

    if not articles:
        # Use the specialized agent to explain why no news was found
        no_evidence_explanation = no_evidence_reasoner_agent(claim)
        
        return {
            "verdict": "⚠️ No News Found",
            "score": 0,
            "claim": claim,
            "explanation": no_evidence_explanation,
            "sources": []
        }

    source_score, tone_score = score_credibility_agent(text, articles)
    st.session_state.messages.append({"role": "system", "content": f"Initial Scoring: Source Credibility={source_score}, Tone Trust={tone_score}"})

    # --- STEP 4: Evidence Synthesis ---
    synthesis_result = evidence_synthesis_agent(claim, articles)
    st.session_state.messages.append({"role": "system", "content": f"Evidence Synthesis Result: {synthesis_result}"})

    # --- STEP 5: Final Reasoning ---
    final_result = final_reasoner_agent(claim, synthesis_result, source_score, tone_score, articles)

    return {
        "verdict": final_result["verdict"],
        "score": final_result["score"],
        "claim": claim,
        "explanation": final_result["explanation"],
        "sources": [a["url"] for a in articles]
    }


# -----------------------------------------------------------
# 🧩 STREAMLIT UI
# -----------------------------------------------------------

st.set_page_config(page_title="Multi-Agent Fact-Check System", page_icon="🧠", layout="centered")
st.title("🧠 Misgo-ai")
st.markdown("A team of 8 specialized AI agents collaborates to verify your claims using LLM reasoning and the News API.")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display past messages
for msg in st.session_state.messages:
    if msg["role"] == "system": # Don't show system messages
        continue
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
user_input = st.chat_input("Enter any news or information to check:")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Multi-Agent System is running..."):
            result = verify_claim_orchestrator(user_input)

            if result["sources"]:
                sources_text = "\n".join([f"- [{url.split('/')[2]}... ]({url})" for url in result["sources"]])
            else:
                sources_text = "_No external news sources were used for this claim._"

            response_text = f"""
**🧠 Main Claim:** {result['claim']}

**🎯 Final Verdict:** **{result['verdict']}** **📊 Truth Score:** {result['score']} / 100  

---
##### 💬 Reasoning
{result['explanation']}
---

**📰 Evidence:** {sources_text}
"""
            st.markdown(response_text)
            st.session_state.messages.append({"role": "assistant", "content": response_text})
            
            # Remove system messages after displaying the result to keep chat cleaner
            st.session_state.messages = [m for m in st.session_state.messages if m["role"] != "system"]
            st.rerun() # Rerun to remove system messages from display
