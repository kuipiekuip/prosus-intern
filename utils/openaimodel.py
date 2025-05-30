from openai import OpenAI

client = OpenAI(api_key="XXXXXXX")

def rerank_with_openai(query, candidates):
    candidate_text = "\n".join([f"{i+1}. {c}" for i, c in enumerate(candidates)])
    
    prompt = f"""
You are a smart food search assistant.

Given a user query and a list of candidate food items, rank them from most to least relevant to the query.

Query: "{query}"

Candidate food items:
{candidate_text}

Output ONLY the numbers in a comma-separated format, no additional text.
Example: 2, 1, 3
"""

    response = client.responses.create(
        model="gpt-4.1",
        input=prompt
    )

    return response.output[0].content[0].text.strip()