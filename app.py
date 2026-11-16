import chainlit as cl
import csv
import os
import time
from datetime import datetime
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

embedding_model = FastEmbedEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embedding_model
)

llm = ChatGroq(model="openai/gpt-oss-120b")

rag_prompt = ChatPromptTemplate.from_template("""
You are a helpful customer support assistant for a telecom company.
Answer the user's question using ONLY the context provided below.
If the answer isn't in the context, say you don't have that information, in the same language as the question.

IMPORTANT: Detect the language of the QUESTION below (English or Arabic) and respond in that exact language, regardless of what language the context happens to be in.

Context:
{context}

Question: {question}

Answer:
""")

chitchat_prompt = ChatPromptTemplate.from_template("""
You are a friendly customer support assistant for a telecom company.
The user sent a casual/conversational message (not a product question).
Reply briefly and warmly in the SAME language as their message, and gently invite them to ask about mobile plans, billing, roaming, internet, or account issues.

User message: {message}

Reply:
""")

classify_prompt = ChatPromptTemplate.from_template("""
Classify the following user message into exactly one category:
- "chitchat" — greetings, small talk, thanks, how are you, casual conversation, anything NOT asking about telecom services
- "product" — a real question about mobile plans, billing, roaming, internet, troubleshooting, account, devices, or promotions

Respond with ONLY one word: chitchat or product

Message: {message}
""")

MAX_INPUT_LENGTH = 500
RATE_LIMIT_COUNT = 5
RATE_LIMIT_WINDOW = 30
FEEDBACK_FILE = "feedback.csv"


def classify_message(message: str) -> str:
    chain = classify_prompt | llm
    result = chain.invoke({"message": message}).content.strip().lower()
    return "chitchat" if "chitchat" in result else "product"


def rag_answer(question, k=5):
    retrieved_docs = vectorstore.similarity_search(question, k=k)
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    chain = rag_prompt | llm
    response = chain.invoke({"context": context, "question": question})
    return response.content


def chitchat_answer(message):
    chain = chitchat_prompt | llm
    response = chain.invoke({"message": message})
    return response.content


def save_feedback(rating, category, question, answer):
    file_exists = os.path.isfile(FEEDBACK_FILE)
    # utf-8-sig so Arabic text displays correctly when opened in Excel
    with open(FEEDBACK_FILE, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "rating", "category", "question", "answer"])
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            rating,
            category,
            question,
            answer,
        ])


@cl.on_message
async def main(message: cl.Message):
    user_question = message.content

    if not user_question.strip():
        await cl.Message(content="It looks like your message was empty — could you type your question?").send()
        return

    if len(user_question) > MAX_INPUT_LENGTH:
        await cl.Message(
            content=f"Your message is a bit long (over {MAX_INPUT_LENGTH} characters). Could you shorten your question?"
        ).send()
        return

    now = time.time()
    timestamps = cl.user_session.get("timestamps", [])
    timestamps = [t for t in timestamps if now - t < RATE_LIMIT_WINDOW]

    if len(timestamps) >= RATE_LIMIT_COUNT:
        await cl.Message(
            content="You're sending messages a bit too quickly. Please wait a few seconds and try again."
        ).send()
        return

    timestamps.append(now)
    cl.user_session.set("timestamps", timestamps)

    # Classify before deciding how to respond
    category = classify_message(user_question)

    if category == "chitchat":
        answer = chitchat_answer(user_question)
    else:
        answer = rag_answer(user_question)

    # Create the reply with rating buttons attached
    reply = cl.Message(content=answer)
    reply.actions = [
        cl.Action(name="rate_up", payload={"msg_id": reply.id}, label="👍"),
        cl.Action(name="rate_down", payload={"msg_id": reply.id}, label="👎"),
    ]

    # Remember the question/answer for this reply so the rating can be logged
    pending = cl.user_session.get("pending_ratings", {})
    pending[reply.id] = {
        "message": reply,
        "category": category,
        "question": user_question,
        "answer": answer,
    }
    cl.user_session.set("pending_ratings", pending)

    await reply.send()


async def handle_rating(action: cl.Action, rating: str):
    pending = cl.user_session.get("pending_ratings", {})
    entry = pending.pop(action.payload.get("msg_id"), None)
    cl.user_session.set("pending_ratings", pending)

    # Already rated (or session restarted) — ignore duplicate clicks
    if entry is None:
        return

    save_feedback(rating, entry["category"], entry["question"], entry["answer"])

    # Remove both buttons and show a short confirmation under the answer
    reply = entry["message"]
    for a in reply.actions:
        await a.remove()
    reply.actions = []
    reply.content = f"{entry['answer']}\n\n*Thanks for your feedback {'👍' if rating == 'up' else '👎'}*"
    await reply.update()


@cl.action_callback("rate_up")
async def on_rate_up(action: cl.Action):
    await handle_rating(action, "up")


@cl.action_callback("rate_down")
async def on_rate_down(action: cl.Action):
    await handle_rating(action, "down")