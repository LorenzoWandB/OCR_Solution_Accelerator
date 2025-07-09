import weave

extractor_prompt = """
Extract all readable text from this image. Format the extracted entities as a valid JSON.
Do not return any extra text, just the JSON. Do not include ```json```
"""
system_prompt = weave.StringPrompt(extractor_prompt)
weave.publish(system_prompt, name="Extractor-Prompt")