from openai import OpenAI

# Configure the Gemini client (OpenAI-compatible)
client = OpenAI(
    api_key="AIzaSyCoDA-zItxAAzQmAN_-kdKCEKc2lNDoTb8",  # Replace with your actual Gemini API key
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# Reranking function
def rerank_with_gemini(query, candidates):
    candidate_text = "\n".join([f"{i+1}. {c}" for i, c in enumerate(candidates)])
    
    prompt = f"""
You are a smart food search assistant.

Given a user query and a list of candidate food items, rank them from most to least relevant to the query.

Query: "{query}"

Candidate food items:
{candidate_text}

Output ONLY the numbers in a comma-separated format, NO additional text.
Example: 2, 1, 3

NO additional text, just the numbers.
"""

    response = client.chat.completions.create(
        model="gemini-2.0-flash",
        messages=[

            {"role": "user", "content": prompt}
        ],
        temperature=0.0
    )

    return response.choices[0].message.content.strip()


def evaluate_ranking(query, candidates):
    """
    Evaluate the ranking of candidates based on the query and expected ranking.
    
    Args:
        query (str): The user query.
        candidates (list): List of candidate food items.
        expected_ranking (list): Expected ranking of candidates as a list of indices.
        
    Returns:
        dict: Evaluation results including reranked items and their relevance scores.
    """
    reranked_items = rerank_with_gemini(query, candidates)
    reranked_indices = [int(i) - 1 for i in reranked_items.split(",")]
    
    evaluation_results = {
        "query": query,
        "reranked_items": [candidates[i] for i in reranked_indices],
        "expected_ranking": [candidates[i] for i in expected_ranking]
    }
    
    return evaluation_results