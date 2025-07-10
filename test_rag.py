# query_data and langchain imports would be here
from query_data import query_rag
from langchain_google_genai import ChatGoogleGenerativeAI

MODEL = "gemini-1.5-flash-latest"
EVAL_PROMPT = """
Expected Response: {expected_response}
Actual Response: {actual_response}
---
(Answer with 'true' or 'false') Does the actual response contain the same core information as the expected response?
"""

# ... your test functions remain the same ...

def test_joseon_founding_year():
    assert query_and_validate(
        question="What year was the Joseon dynasty founded? (Answer with the year only)",
        expected_response="1392",
    )

def test_korean_war_start_year():
    assert query_and_validate(
        question="What year did the Korean War start? (Answer with the year only)",
        expected_response="1950",
    )

def test_hangul_creator():
    assert query_and_validate(
        question="Who created the Korean alphabet, Hangul? (Answer with the name only)",
        expected_response="Sejong the Great",
    )

# --- CORRECTED FUNCTION ---
def query_and_validate(question: str, expected_response: str) -> bool:
    """
    Queries the RAG system and uses a Gemini LLM to validate the response.
    """
    # FIX: Clean the RAG output immediately to remove whitespace/newlines.
    # query_rag returns a string when return_sources=False (default)
    rag_response = query_rag(question, return_sources=False)
    response_text = str(rag_response).strip()
    
    prompt = EVAL_PROMPT.format(
        expected_response=expected_response, actual_response=response_text
    )

    # Use the Google Gemini model for evaluation.
    model = ChatGoogleGenerativeAI(model=MODEL, temperature=0.0)
    evaluation_result = model.invoke(prompt)
    evaluation_results_str = str(evaluation_result.content).strip().lower()

    print(f"\n--- Test Validation ---\nQuestion: {question}")
    print(f"Expected: {expected_response}")
    print(f"Actual: {response_text}") # This will now print the cleaned text

    if "true" in evaluation_results_str:
        print("\033[92m" + "Result: CORRECT" + "\033[0m")
        return True
    elif "false" in evaluation_results_str:
        print("\033[91m" + "Result: INCORRECT" + "\033[0m")
        return False
    else:
        print("\033[93m" + f"Result: UNCERTAIN (LLM response: {evaluation_results_str})" + "\033[0m")
        return False

if __name__ == "__main__":
    test_joseon_founding_year()
    test_korean_war_start_year()
    test_hangul_creator()
