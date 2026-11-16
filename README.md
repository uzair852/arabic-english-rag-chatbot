\# Arabic/English RAG Customer Support Chatbot



A multilingual retrieval-augmented generation (RAG) chatbot for telecom customer support. It answers questions about mobile plans, billing, roaming, internet and account issues, and replies in the same language the user writes in (English or Arabic). It only answers from a curated knowledge base, which reduces made-up answers.



Built during my AI Engineering internship at e\& (Etisalat), June–August 2026.



\## Features



\- \*\*Bilingual:\*\* detects whether the question is in English or Arabic and answers in that language, even when the retrieved context is in the other language

\- \*\*Grounded answers:\*\* responds only from retrieved knowledge base content, and says so when the answer isn't available

\- \*\*Intent routing:\*\* an LLM classifier separates casual chat from real product questions, so greetings get a friendly reply instead of a document search

\- \*\*Input guardrails:\*\* blocks empty messages, limits input to 500 characters, and rate-limits each session (5 messages per 30 seconds)

\- \*\*Containerised deployment:\*\* Dockerised and deployed on Railway



\## Architecture



```

User message

&#x20;    │

&#x20;    ▼

Input checks (empty / length / rate limit)

&#x20;    │

&#x20;    ▼

Intent classifier (Llama 3.3 70B)

&#x20;    │

&#x20;    ├── chitchat ──► Friendly reply in the user's language

&#x20;    │

&#x20;    └── product ──► Embed query (FastEmbed, multilingual MiniLM)

&#x20;                         │

&#x20;                         ▼

&#x20;                    Retrieve top 5 chunks (ChromaDB)

&#x20;                         │

&#x20;                         ▼

&#x20;                    Generate grounded answer (Llama 3.3 70B via Groq)

```



\## Tech Stack



| Component | Technology |

|---|---|

| LLM | Llama 3.3 70B (Groq) |

| Orchestration | LangChain |

| Embeddings | FastEmbed – `paraphrase-multilingual-MiniLM-L12-v2` |

| Vector database | ChromaDB |

| Interface | Chainlit |

| Deployment | Docker, Railway |



\## Project Structure



```

├── app.py                           # Chainlit app: routing, retrieval and generation

├── fastembed.ipynb                  # Builds the ChromaDB vector store from the knowledge base

├── telecom\_faq\_knowledge\_base.md    # Source knowledge base

├── chroma\_db/                       # Prebuilt vector store

├── Dockerfile                       # Container setup

├── requirements.txt

└── RAG\_Chatbot\_Solution\_Document.pdf  # Full solution write-up

```



\## Running Locally



1\. Clone the repo:

&#x20;  ```

&#x20;  git clone https://github.com/uzair852/arabic-english-rag-chatbot.git

&#x20;  cd arabic-english-rag-chatbot

&#x20;  ```

2\. Install dependencies:

&#x20;  ```

&#x20;  pip install -r requirements.txt

&#x20;  ```

3\. Create a `.env` file with your Groq API key:

&#x20;  ```

&#x20;  GROQ\_API\_KEY=your\_key\_here

&#x20;  ```

4\. Start the app:

&#x20;  ```

&#x20;  chainlit run app.py

&#x20;  ```



The vector store is already included. To rebuild it after changing the knowledge base, run `fastembed.ipynb`.



\## Documentation



See \[RAG\_Chatbot\_Solution\_Document.pdf](RAG\_Chatbot\_Solution\_Document.pdf) for the full design and evaluation write-up.

