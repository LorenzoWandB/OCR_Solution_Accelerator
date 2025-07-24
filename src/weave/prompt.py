import weave

prompt = weave.StringPrompt("""
You are a helpful assistant. Answer the user's query based on the
provided context. If the context does not contain the answer, say so.

CONTEXT:
{context}

QUERY:
{query}
""")