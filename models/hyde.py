import os
from groq import Groq
import streamlit as st


# HYDE QUERY EXPANSION : this function generates a hypothetical policy excerpt based on the user's query to improve retrieval alignment.
#  It uses the same LLM to create a concise, relevant policy snippet that can be embedded and indexed alongside actual documents,
#   helping to surface more relevant context during retrieval.

def hyde_query_expansion(query: str,api_key) -> str:
    client = Groq(api_key=api_key)
    """Generate a hypothetical policy excerpt to improve retrieval alignment."""
    try:
        expansion_prompt = (
            f"Write a concise one-paragraph HR policy excerpt that would directly answer: '{query}'. "
            "Write it as if it's from an actual HR policy document."
        )
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": expansion_prompt}],
            max_tokens=150,
            temperature=0.1
        )
        return response.choices[0].message.content
    except Exception:
        return query  # Fallbacks to original query on failure
