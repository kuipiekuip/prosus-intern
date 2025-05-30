import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import json
from sklearn.metrics.pairwise import cosine_similarity
from utils.gemini import rerank_with_gemini
from utils.openaimodel import rerank_with_openai

# Load data
items = pd.read_csv("cleaned_items_with_metadata.csv")
item_texts = items["full_text"].tolist()
queries = pd.read_csv("./data/queries.csv")["search_term_pt"].tolist()
item_embeddings = np.load("./embeddings/text_embeddings_openai_small.npy")
query_embeddings = np.load("./embeddings/query_embeddings_openai_small.npy")
 # Your precomputed 768-dim vectors
parsed_meta = [json.loads(item) for item in items["itemMetadata"]]
# model = SentenceTransformer("PORTULAN/serafim-100m-portuguese-pt-sentence-encoder")

st.title("🍽️ Semantic Food Search Evaluation Demo")

query_idx = st.slider("Select query index", 0, len(queries) - 1)
query = queries[query_idx]
st.subheader(f"🔎 Query: {query}")

similarity_matrix = cosine_similarity(query_embeddings, item_embeddings)
top_k = 5
top_k_indices = np.argsort(-similarity_matrix, axis=1)[:, :top_k]

results = []

for i, item_idxs in enumerate(top_k_indices):
    query_text = queries[i]
    matched_items = [item_texts[j] for j in item_idxs]
    results.append({
        "query": query_text,
        "top_k_results": matched_items
    })
    
if st.button("🔍 Search"):
    matched_items = results[query_idx]["top_k_results"]
    print(f"Matched items: {matched_items}")
    # Gemini reranking
    # new_rank = rerank_with_openai(query, matched_items)
    # print("new rank: ", new_rank)  # returns something like "2, 1, 3, 5, 4"
    # reranked_items = [matched_items[int(i)-1] for i in new_rank.split(",")]
    # print(f"Reranked items: {reranked_items}")
    # Display results with metadata and images
    st.subheader("🔝 Top Matches (without rerank)")

    for text in matched_items[:3]:
        # Find the item in original DB by full_text match
        match_row = items[items["full_text"] == text]
        if match_row.empty:
            continue
        
        row = match_row.iloc[0]
        meta = json.loads(row["itemMetadata"])
        
        name = meta.get("name", "No name")
        desc = meta.get("description", "")
        cat = meta.get("category_name", "")
        images = meta.get("images", [])
        img = images[0] if images else None
  # first image
        st.markdown(f"### 🥘 {name}")
        if img:
            image_url = f"https://static.ifood-static.com.br/image/upload/t_low/pratos/{img}"
            st.image(image_url, use_column_width=True)
        else:
            st.markdown("⚠️ No image available.")


        st.markdown(f"**Category:** {cat}")
        st.markdown(f"**Description:** {desc}")


# Button to export evaluation CSV later

