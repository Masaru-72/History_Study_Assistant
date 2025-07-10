import argparse
import requests
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from get_embedding_function import get_embedding_function
import markdown2
from PIL import Image
import io
from langchain_core.messages import HumanMessage
import asyncio
from typing import Optional, AsyncIterable


# === CONFIGURATION ===
MODEL = "gemini-1.5-pro"
TEMP = 0.4
CHROMA_PATH = "chroma"
GOOGLE_API_KEY = "your_google_api_key"
SEARCH_ENGINE_ID = "your_search_engine_id"

# === MEMORY ===
memory = ConversationBufferMemory(memory_key="history", return_messages=False)

# === PROMPT TEMPLATES ===
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template("""
Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.

Chat History:
{history}
Follow Up Input: {question}
Standalone question:
""")

SUMMARY_PROMPT_TEMPLATE = {
    "en": PromptTemplate.from_template("""
You are an expert summarizer. Based on the following text, create a concise, three-bullet-point summary. Each bullet point should be a complete sentence.

Text:
{text}

Three-bullet-point summary:
"""),
    "kor": PromptTemplate.from_template("""
당신은 전문 요약가입니다. 다음 텍스트를 기반으로 간결한 세 문장의 요약을 작성하십시오. 각 항목은 완전한 문장이어야 합니다.

텍스트:
{text}

세 문장 요약:
""")
}


PROMPT_TEMPLATE = {
    "en": """
You are a knowledgeable, adaptive Korean history expert. Answer the following question as thoroughly as possible.

Only discuss Joseon history when the user's question is clearly about the Joseon Dynasty or its context. For all other topics, do not mention Joseon and history at all even if it Korea related like "What is Korea?". Provide a concise, general answer based strictly on verified knowledge.
For the image inputs, check if it is related to the question. If it is, use it to answer the question. If it is not, ignore it by saying the picture is not related. 

Use only the provided context when it contains relevant information. If the context lacks key details, draw carefully from your own verified historical knowledge—but do not invent facts or speculate beyond established history. Never respond with phrases like \"no further details provided\" or \"the context only mentions them by name.\" Instead, expand on what is known and provide meaningful historical insight, even if it requires background information.

Match the tone and style of the user's phrasing—answer casually if asked casually, formally if asked formally, or in a specific style (like Yakuza, cat, King Sejong, historical figure style) if requested.

Make your explanation clear, logically structured, and informative. Adjust the length to match the complexity of the question and the depth of historical content—never under-explain or over-explain (2-3 sentences would be okay). When possible, include relevant events, figures, motivations, or consequences.

If the context contains conflicting information, rely on your historical expertise to select the more credible source and present a clear, supported answer.

Context:
{context}

Chat History:
{history}

---

Question: {question}

Answer:
""",
    "ko": """
당신은 지식이 풍부한 한국사 전문가입니다. 아래 질문에 대해 최대한 자세히 한국어로 답변해 주세요.

사용자의 질문이 조선 왕조 또는 그 맥락에 대한 것이 분명한 경우에만 조선 역사에 대해 논의하세요. 다른 주제에 대해서는 "한국이란 무엇인가?", "What is Korea?"처럼 한국과 관련된 내용일지라도 조선과 역사를 전혀 언급하지 마세요. 검증된 지식을 바탕으로 간결하고 일반적인 답변을 제공하세요.

이미지 입력이 질문과 관련이 있다면 그것을 활용하여 답변하고, 관련이 없다면 '이미지와 관련 없음'이라고 답해주세요.

제공된 문맥에 관련 정보가 있다면 그것만 사용하세요. 부족한 경우에는 검증된 역사 지식에 근거해 내용을 보완하되, 사실을 조작하거나 추측하지 마세요.

사용자의 말투에 맞춰 답변하세요—캐주얼하게 묻는다면 캐주얼하게, 포멀하게 묻는다면 포멀하게, 특정 스타일(예: 야쿠자 스타일)이 있다면 그것에 맞춰주세요.

답변은 명확하고 논리적으로 구성되어야 하며, 질문의 복잡성과 역사적 내용의 깊이에 따라 길이를 조절하세요. 가능하면 사건, 인물, 동기, 결과 등을 포함해 주세요.

문맥에 모순되는 정보가 있다면 더 신뢰할 수 있는 것을 선택해 명확하게 설명하세요.

문맥:
{context}

대화 기록:
{history}

---

질문: {question}

답변:
"""
}

# === GOOGLE SEARCH FALLBACK ===
def google_search_fallback(query: str, num_results: int = 5) -> str:
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "q": query,
        "key": GOOGLE_API_KEY,
        "cx": SEARCH_ENGINE_ID,
        "num": num_results
    }
    res = requests.get(url, params=params)
    items = res.json().get("items", [])
    return "\n\n".join(f"{i+1}. {item['title']}\n{item['snippet']}" for i, item in enumerate(items))

# === STANDALONE QUESTION ===
def _get_standalone_question(question: str, history: str) -> str:
    llm = ChatGoogleGenerativeAI(model=MODEL, temperature=0.0)
    prompt = CONDENSE_QUESTION_PROMPT.format(history=history, question=question)
    response = llm.invoke(prompt)
    content = response.content[0] if isinstance(response.content, list) else response.content
    return str(content) if isinstance(content, dict) else content

# === IMAGE RELEVANCE CHECK ===
def is_image_relevant_to_query(image_bytes: bytes, query_text: str) -> bool:
    try:
        vision_model = ChatGoogleGenerativeAI(model="gemini-1.5-pro-vision", temperature=0.1)
        image = Image.open(io.BytesIO(image_bytes))
        prompt = HumanMessage(content=[
            {"type": "text", "text": (
                f"Given this question:\n'{query_text}'\n"
                "Is the uploaded image relevant to answering it? "
                "Answer only 'yes' or 'no'."
            )},
            {"type": "image", "image": image}
        ])
        response = vision_model.invoke([prompt])
        content = response.content[0] if isinstance(response.content, list) else response.content
        return str(content).strip().lower().startswith("yes")
    except Exception:
        return False

def _generate_summary(text_to_summarize: str, llm: ChatGoogleGenerativeAI, lang: str = "en") -> str:
    """
    Generates a longer answer first, then a summary, and returns both formatted in HTML.
    """
    if lang == "ko":
        expansion_prompt = (
            "다음 내용을 읽고 역사적 배경과 맥락을 충분히 설명하는 방식으로 자세히 풀어 써 주세요. "
            "정확하고 구체적인 설명을 포함하고, 원래 텍스트보다 약 2배 길이로 작성해 주세요. 요약하지 말고, 단락 구성을 명확히 해 주세요.\n\n"
            f"텍스트:\n{text_to_summarize}"
        )
    else:
        expansion_prompt = (
            "Expand and elaborate on the following text in a clear and informative way, "
            "providing rich historical detail and contextual background. "
            "Make the explanation approximately twice as long as the original input, but stay accurate and focused. "
            "Do not summarize — just expand the content first.\n\n"
            f"Text:\n{text_to_summarize}"
        )

    # Step 1: Expand original answer
    expansion_response = llm.invoke(expansion_prompt)
    expanded_text = (
        expansion_response.content[0] if isinstance(expansion_response.content, list)
        else expansion_response.content
    )
    expanded_text = str(expanded_text)

    if lang == "ko":
        concise_prompt = (
            "다음 텍스트를 읽고 **간결한 요약 제목**과 **3개의 핵심 문장 요약**을 작성해 주세요.\n"
            "각 항목은 완전한 문장이어야 하며, 중요한 내용을 중심으로 정리해 주세요.\n\n"
            "**제목**\n- 요약 1\n- 요약 2\n- 요약 3\n\n"
            f"텍스트:\n{expanded_text}"
        )
    else:
        concise_prompt = (
            "Give a short title and summarize the following in **3 concise bullet points**. "
            "Keep the summary brief and focused only on the key points. "
            "Format:\n\n"
            "**Title**\n- Bullet 1\n- Bullet 2\n- Bullet 3\n\n"
            f"Text:\n{expanded_text}"
        )
    summary_response = llm.invoke(concise_prompt)
    summary_text = (
        summary_response.content[0] if isinstance(summary_response.content, list)
        else summary_response.content
    )
    summary_text = str(summary_text)

    # Parse summary into HTML
    lines = summary_text.strip().splitlines()
    title = lines[0].strip(" *") if lines else ""
    bullets = [line.strip(" -") for line in lines[1:] if line.strip()] if len(lines) > 1 else []
    bullet_html = "<br>".join(f"• {b}" for b in bullets)
    bold_title = " ".join(f"<b>{word}</b>" for word in title.split())
    generated_summary_html = f"{bold_title}<br>{bullet_html}"

    # Format expanded text into HTML
    original_text_html = expanded_text.replace('\n', '<br>')
    separator = "<br><br><br>"
    return f"{generated_summary_html}{separator}{original_text_html}"


def should_summarize_llm(text: str) -> bool:
    check_prompt = (
        "Given the following text, answer with only 'yes' or 'no'. "
        "Say 'yes' if the text is more than a few sentences, contains historical details, "
        "or has multiple ideas that would benefit from a short summary. "
        "Only say 'no' if the text is extremely short or self-explanatory.\n\n"
        f"Text:\n{text}"
    )
    
    checker = ChatGoogleGenerativeAI(model=MODEL, temperature=0.0)
    response = checker.invoke(check_prompt)
    content = response.content[0] if isinstance(response.content, list) else response.content
    return str(content).strip().lower().startswith("yes")

# === FINAL STREAMING FUNCTION ===
async def query_rag_stream(
    query_text: str,
    memory: ConversationBufferMemory,
    image_bytes: Optional[bytes] = None,
    speed: float = 1.0,
    lang: str = "en"
) -> AsyncIterable[str]:
    print("[QUERY RECEIVED]:", query_text)
    chat_history = memory.load_memory_variables({}).get("history", "")
    standalone_question = _get_standalone_question(query_text, chat_history) if chat_history else query_text

    # === VECTOR SEARCH (Chroma) ===
    embedding_function = get_embedding_function()
    try:
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
        results = db.similarity_search_with_score(standalone_question, k=5)
        print(f"[CHROMA RESULTS]: {len(results)} docs")

        if not results or all(score > 0.9 for _, score in results):
            context_text = google_search_fallback(standalone_question)
        else:
            context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in results])
    except Exception as e:
        print(f"[CHROMA ERROR]: {e}")
        context_text = "No relevant context found. Falling back to expert knowledge."

    # === IMAGE CHECK ===
    if image_bytes and is_image_relevant_to_query(image_bytes, standalone_question):
        context_text += "\n\n---\n\n[Relevant image uploaded by user was used.]"
    elif image_bytes:
        context_text += "\n\n---\n\n[Image uploaded, but not related to the question — ignored.]"

    prompt_template = PROMPT_TEMPLATE.get(lang, PROMPT_TEMPLATE["en"])
    prompt = ChatPromptTemplate.from_template(prompt_template).format(
             context=context_text,
             history=chat_history or "No history yet.",
             question=query_text,
            )

    llm = ChatGoogleGenerativeAI(model=MODEL, temperature=TEMP)

    print("[START STREAMING WORD-BY-WORD]")
    full_response = ""
    try:
        # First gather full content
        full_text = ""
        for chunk in llm.stream(prompt):
            token = chunk.content
            if token:
                if isinstance(token, list):
                    token_str = "".join(str(item) for item in token)
                else:
                    token_str = str(token)
                full_text += token_str

        # Add summary before streaming
        if should_summarize_llm(full_text):
            full_text = _generate_summary(full_text, llm, lang)

        # Stream word by word
        words = full_text.split()
        base_delay = 0.05
        delay = base_delay / speed 
        for i, word in enumerate(words):
            word_with_space = word + (" " if i < len(words) - 1 else "")
            full_response += word_with_space
            yield word_with_space
            await asyncio.sleep(delay)  # Adjust for pacing

    except Exception as e:
        print(f"[STREAMING ERROR]: {e}")
        error_msg = f"[ERROR] {str(e)}"
        yield error_msg
        full_response = error_msg

    memory.save_context({"input": query_text}, {"output": full_response})

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()

    response_text = query_rag_stream(args.query_text, memory)
    print(f"\nResponse: {response_text}")

if __name__ == "__main__":
    main()
