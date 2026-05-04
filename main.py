from groq import Groq
from dotenv import load_dotenv
import os
import json

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def analyze_text(user_input):
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": """You are a text analysis assistant. 
                Analyze the user's text and respond ONLY with a JSON object in this exact format:
                {
                    "summary": "one sentence summary",
                    "sentiment": "positive, negative, or neutral",
                    "key_topics": ["topic1", "topic2", "topic3"],
                    "word_count": number
                }
                Return only the JSON, no extra text.""",
            },
            {"role": "user", "content": user_input},
        ],
    )

    raw = response.choices[0].message.content
    result = json.loads(raw)
    return result


while True:
    user_input = input("\nEnter text to analyze (or 'quit' to exit): ")

    if user_input.lower() == "quit":
        print("Goodbye!")
        break

    try:
        result = analyze_text(user_input)
        print("\n--- Analysis Result ---")
        print(f"Summary:    {result['summary']}")
        print(f"Sentiment:  {result['sentiment']}")
        print(f"Topics:     {', '.join(result['key_topics'])}")
        print(f"Word count: {result['word_count']}")
    except Exception as e:
        print(f"Something went wrong: {e}")
