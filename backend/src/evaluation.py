import json
import os
import sys
from dotenv import load_dotenv

load_dotenv()

# --------------------------------------------------
# CONFIG
# --------------------------------------------------

JUDGE_MODEL = "llama-3.3-70b-versatile"

API_KEY = os.getenv("GROQ_API_KEY")
if not API_KEY:
    raise RuntimeError("GROQ_API_KEY not set")

try:
    from groq import Groq
    client = Groq(api_key=API_KEY)
except ImportError:
    print("❌ Missing dependency: pip install groq")
    sys.exit(1)


# --------------------------------------------------
# JUDGE FUNCTION
# --------------------------------------------------

def grade_response(question: str, student_answer: str, expert_answer: str) -> dict:
    """
    Grades a model response against an expert answer.
    Returns: { "score": int, "explanation": str }
    """

    prompt = f"""
You are an impartial legal evaluator.

Evaluate the STUDENT answer strictly against the EXPERT answer.
Focus on legal accuracy, completeness, and correctness.
Do NOT reward style, confidence, or verbosity.

Question:
{question}

Student Answer:
{student_answer}

Expert Answer:
{expert_answer}

Scoring Rubric:
5 = Fully correct, legally precise, no material omissions
4 = Mostly correct, minor legal imprecision
3 = Partially correct, missing key legal elements
2 = Largely incorrect, minimal correct content
1 = Completely incorrect or irrelevant

Return ONLY valid JSON.
Format:
{{ "score": int, "explanation": "string" }}
"""

    try:
        completion = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.0,
        )

        result = json.loads(completion.choices[0].message.content)

        # Defensive validation
        if not isinstance(result.get("score"), int):
            raise ValueError("Score is not an integer")

        return result

    except Exception as e:
        return {
            "score": 0,
            "explanation": f"Judge failure: {str(e)}"
        }


# --------------------------------------------------
# TEST HARNESS
# --------------------------------------------------

test_cases = [
    {
        "q": "What is the punishment for theft?",
        "gold": "Theft is punishable under Section 379 IPC with imprisonment up to three years, or with fine, or with both.",
        "student": "Theft is bad and you go to jail for 3 years under section 379."
    },
    {
        "q": "What is Article 21 of the Constitution of India?",
        "gold": "Article 21 guarantees the protection of life and personal liberty except according to procedure established by law.",
        "student": "Article 21 is about freedom of speech."
    }
]

print(f"\n--- 👨‍⚖️ AI Judge Evaluation ({JUDGE_MODEL}) ---\n")

for case in test_cases:
    result = grade_response(case["q"], case["student"], case["gold"])

    print(f"Question: {case['q']}")
    print(f"Student Answer: {case['student']}")
    print(f"Score: {result['score']}/5")
    print(f"Explanation: {result['explanation']}")
    print("-" * 60)
