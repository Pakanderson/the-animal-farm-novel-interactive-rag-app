import os
from dotenv import load_dotenv
import streamlit as st

# Load local environment variables (if any exist locally)
load_dotenv()

# --- 1. Secure API Token Configuration ---
# Look for the Groq API key in Streamlit Secrets first, otherwise fall back to local environment
if "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
elif not os.getenv("GROQ_API_KEY"):
    st.warning(
        "Groq API Key not found. Please set it in your local environment or Streamlit Secrets."
    )

# Look for the Hugging Face Token in Streamlit Secrets first, otherwise fall back to local environment
if "HF_TOKEN" in st.secrets:
    os.environ["HF_TOKEN"] = st.secrets["HF_TOKEN"]

# --- 2. Framework & Library Imports ---
# Official Hugging Face Hub client
from huggingface_hub import InferenceClient

# LlamaIndex Core & Interface Setup Elements
from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.llms.groq import Groq
from llama_index.core.chat_engine import ContextChatEngine
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.base.llms.types import ChatMessage, MessageRole

# Custom Native Base Embedding Types
from llama_index.core.embeddings import BaseEmbedding
from pydantic import Field
from typing import Any, List

# --- Page Configuration ---
st.set_page_config(
    page_title="Animal Farm Novel Analyzer",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --- Cloud API Embedding Wrapper using InferenceClient ---
class LightAPIEmbedding(BaseEmbedding):
    model_name: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")

    def _get_query_embedding(self, query: str) -> List[float]:
        # Securely fetch token dynamically at runtime
        hf_token = os.getenv("HF_TOKEN") or (
            st.secrets["HF_TOKEN"] if "HF_TOKEN" in st.secrets else ""
        )
        client = InferenceClient(token=hf_token)
        embedding = client.feature_extraction(text=query, model=self.model_name)
        if hasattr(embedding, "tolist"):
            embedding = embedding.tolist()
        if (
            isinstance(embedding, list)
            and len(embedding) > 0
            and isinstance(embedding[0], list)
        ):
            embedding = embedding[0]
        return [float(x) for x in embedding]

    def _get_text_embedding(self, text: str) -> List[float]:
        hf_token = os.getenv("HF_TOKEN") or (
            st.secrets["HF_TOKEN"] if "HF_TOKEN" in st.secrets else ""
        )
        client = InferenceClient(token=hf_token)
        embedding = client.feature_extraction(text=text, model=self.model_name)
        if hasattr(embedding, "tolist"):
            embedding = embedding.tolist()
        if (
            isinstance(embedding, list)
            and len(embedding) > 0
            and isinstance(embedding[0], list)
        ):
            embedding = embedding[0]
        return [float(x) for x in embedding]

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)


# --- 3. Initialize RAG Backend (Cached for Performance) ---
@st.cache_resource
def initialize_rag_system():
    model = "llama-3.3-70b-versatile"

    # Secure dynamic API token acquisition
    groq_key = os.getenv("GROQ_API_KEY")
    llm = Groq(model=model, token=groq_key)

    # Instantiate our stable Hub Client embedded wrapper
    embed_model = LightAPIEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Assign components to LlamaIndex Settings global configurations
    Settings.llm = llm
    Settings.embed_model = embed_model

    # Load local vector data registry maps safely
    storage_context = StorageContext.from_defaults(persist_dir="vector_index")
    vector_index = load_index_from_storage(storage_context)

    return vector_index


# Run the backend system initialization
try:
    vector_index = initialize_rag_system()
except Exception as e:
    st.error(
        f"Failed to load vector index. Ensure the 'vector_index' folder exists. Error: {e}"
    )
    st.stop()


# --- 4. Streamlit Layout & Interface Configuration ---
st.title("The Animal Farm Novel: Political Satire Analyzer")
st.markdown(
    "*An AI-powered RAG system designed to parse, retrieve, and analyze the deep allegorical layers of George Orwell's classic novel.*"
)

# Sidebar Structure for RAG Controls
st.sidebar.header("RAG Settings")
top_k_slider = st.sidebar.slider(
    label="Number of Documents (Top K)",
    min_value=1,
    max_value=5,
    value=2,
    step=1,
    help="Controls how many relevant chunks of the novel are retrieved to answer your question.",
)

# Reset Button to clear conversations cleanly
if st.sidebar.button("Reset Conversation", use_container_width=True):
    st.session_state.chat_history = []
    st.rerun()


# --- 5. Streamlit Session State Chat History ---
if "chat_history" not in st.session_state or len(st.session_state.chat_history) == 0:
    st.session_state.chat_history = [
        {
            "role": "assistant",
            "content": "Welcome! This system analyzes George Orwell's 'Animal Farm'—a novel using farm animals to satirize social and political revolutions. Ask me about the characters (Napoleon, Snowball, Boxer) or how the laws of the farm change!",
        }
    ]

# Keep dialogue visually persistent across app updates
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# --- 6. Interactive Chat Input Logic ---
if user_prompt := st.chat_input(
    "Ask something like: 'What is the real-world meaning behind Napoleon and Snowball's conflict?'"
):

    # Render and save the user's incoming message instantly
    with st.chat_message("user"):
        st.markdown(user_prompt)
    st.session_state.chat_history.append({"role": "user", "content": user_prompt})

    # Assemble the active LlamaIndex query components based on layout variables
    retriever = vector_index.as_retriever(similarity_top_k=int(top_k_slider))

    prefix_messages = [
        ChatMessage(
            role=MessageRole.SYSTEM,
            content="You are a nice chatbot having a conversation with a human.",
        ),
        ChatMessage(
            role=MessageRole.SYSTEM,
            content="Answer the question based only on the following context and previous conversation.",
        ),
        ChatMessage(
            role=MessageRole.SYSTEM, content="Keep your answers short and succinct."
        ),
    ]

    # Convert past messages cleanly into formal LlamaIndex Message Schema
    bot_history = []
    for m in st.session_state.chat_history[:-1]:  # Exclude current prompt
        role = MessageRole.USER if m["role"] == "user" else MessageRole.ASSISTANT
        bot_history.append(ChatMessage(role=role, content=m["content"]))

    # Initialize runtime context chat instance
    memory = ChatMemoryBuffer.from_defaults()
    rag_bot = ContextChatEngine(
        llm=Settings.llm,
        retriever=retriever,
        memory=memory,
        prefix_messages=prefix_messages,
    )

    # Trigger model inference behind Streamlit loading spinners
    with st.chat_message("assistant"):
        with st.spinner("Analyzing Orwellian structures..."):
            response = rag_bot.chat(user_prompt, chat_history=bot_history)
            st.markdown(response.response)

    # Save the generated response text back into state memory
    st.session_state.chat_history.append(
        {"role": "assistant", "content": response.response}
    )
