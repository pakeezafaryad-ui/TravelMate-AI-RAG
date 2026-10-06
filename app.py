
import streamlit as st
import chromadb
from groq import Groq
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="TravelMate AI",
    page_icon="✈️",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main {
    padding-top: 1rem;
}

.hero {
    padding: 25px;
    border-radius: 18px;
    margin-bottom: 25px;
    background: linear-gradient(
        135deg,
        #16213E,
        #0F3460
    );
}

.hero h1 {
    color: white;
    font-size: 42px;
    margin-bottom: 5px;
}

.hero p {
    color: #DDE7F5;
    font-size: 18px;
}

.info-card {
    padding: 18px;
    border-radius: 15px;
    background: #F5F7FA;
    margin-bottom: 15px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# HEADER
# ============================================================

st.markdown("""
<div class="hero">

<h1>✈️ TravelMate AI</h1>

<p>
Your intelligent travel knowledge assistant
</p>

</div>
""", unsafe_allow_html=True)


# ============================================================
# GROQ CLIENT
# ============================================================

try:

    groq_api_key = st.secrets["GROQ_API_KEY"]

except Exception:

    st.error(
        "GROQ_API_KEY is not configured. "
        "Add it to Streamlit Secrets."
    )

    st.stop()


client = Groq(
    api_key=groq_api_key
)


# ============================================================
# EMBEDDING MODEL
# ============================================================

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )


embedding_model = load_embedding_model()


# ============================================================
# CHROMADB
# ============================================================

@st.cache_resource
def load_vector_database():

    chroma_client = chromadb.Client()

    collection = chroma_client.get_or_create_collection(
        name="travelmate_documents"
    )

    return collection


collection = load_vector_database()


# ============================================================
# TEXT SPLITTER
# ============================================================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("📄 Travel Knowledge Base")

    st.write(
        "Upload travel documents and TravelMate "
        "will use them as its knowledge source."
    )

    uploaded_files = st.file_uploader(
        "Upload PDF documents",
        type=["pdf"],
        accept_multiple_files=True
    )

    st.divider()

    st.subheader("📊 Knowledge Base")

    st.metric(
        "Stored chunks",
        collection.count()
    )

    st.divider()

    st.caption(
        "Powered by Groq + Hugging Face + ChromaDB"
    )


# ============================================================
# PROCESS UPLOADED PDFS
# ============================================================

if uploaded_files:

    for uploaded_file in uploaded_files:

        file_name = uploaded_file.name

        # Check whether this file has already been processed
        existing = collection.get(
            where={"source": file_name}
        )

        if existing["ids"]:
            continue

        try:

            reader = PdfReader(uploaded_file)

            text = ""

            for page in reader.pages:

                page_text = page.extract_text()

                if page_text:
                    text += page_text + "\n"

            if not text.strip():

                st.warning(
                    f"Could not extract text from {file_name}."
                )

                continue

            # Split document into chunks
            chunks = text_splitter.split_text(text)

            # Create embeddings
            embeddings = embedding_model.encode(
                chunks,
                show_progress_bar=False
            ).tolist()

            # Unique IDs
            ids = [
                f"{file_name}_{i}"
                for i in range(len(chunks))
            ]

            # Metadata
            metadatas = [
                {
                    "source": file_name,
                    "chunk": i
                }
                for i in range(len(chunks))
            ]

            # Store in ChromaDB
            collection.upsert(
                ids=ids,
                documents=chunks,
                embeddings=embeddings,
                metadatas=metadatas
            )

            st.sidebar.success(
                f"✅ {file_name} processed"
            )

        except Exception as e:

            st.sidebar.error(
                f"Error processing {file_name}: {e}"
            )


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


# ============================================================
# WELCOME MESSAGE
# ============================================================

if not st.session_state.messages:

    st.markdown("""
    <div class="info-card">

    <h3>👋 Welcome to TravelMate AI!</h3>

    <p>
    Upload your travel PDFs from the sidebar and ask
    questions about flights, hotels, destinations,
    packages, policies and more.
    </p>

    </div>
    """, unsafe_allow_html=True)


# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if message.get("sources"):

            with st.expander("📚 Sources"):

                for source in message["sources"]:

                    st.caption(f"📄 {source}")


# ============================================================
# CHAT INPUT
# ============================================================

question = st.chat_input(
    "Ask TravelMate something..."
)


if question:

    # --------------------------------------------------------
    # USER MESSAGE
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):

        st.markdown(question)


    # --------------------------------------------------------
    # CHECK KNOWLEDGE BASE
    # --------------------------------------------------------

    if collection.count() == 0:

        answer = (
            "📄 Please upload a travel PDF first. "
            "I need a document to use as my knowledge source."
        )

        sources = []

    else:

        # ----------------------------------------------------
        # EMBED USER QUESTION
        # ----------------------------------------------------

        query_embedding = embedding_model.encode(
            question
        ).tolist()


        # ----------------------------------------------------
        # RETRIEVE RELEVANT CHUNKS
        # ----------------------------------------------------

        number_of_results = min(
            4,
            collection.count()
        )

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=number_of_results
        )


        documents = results["documents"][0]

        metadatas = results["metadatas"][0]


        # ----------------------------------------------------
        # BUILD CONTEXT
        # ----------------------------------------------------

        context = "\n\n---\n\n".join(
            documents
        )


        # ----------------------------------------------------
        # SOURCE NAMES
        # ----------------------------------------------------

        sources = sorted(
            set(
                metadata["source"]
                for metadata in metadatas
            )
        )


        # ----------------------------------------------------
        # RAG PROMPT
        # ----------------------------------------------------

        prompt = f"""
You are TravelMate AI, a helpful travel information
assistant.

Your job is to answer the user's question using ONLY
the information contained in the provided context.

IMPORTANT RULES:

1. Do not invent facts.
2. Do not make up flight schedules.
3. Do not make up hotel prices.
4. Do not make up travel policies.
5. If the answer is not contained in the context,
   clearly say that the information is not available
   in the uploaded travel documents.
6. Keep the answer clear and useful.
7. If several relevant options exist, organize them
   using bullet points or a small table.

DOCUMENT CONTEXT:

{context}

USER QUESTION:

{question}
"""


        # ----------------------------------------------------
        # GROQ
        # ----------------------------------------------------

        try:

            response = client.chat.completions.create(

                model="openai/gpt-oss-20b",

                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are TravelMate AI. "
                            "Answer only from the supplied context."
                        )
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],

                temperature=0.2,

                max_tokens=800
            )

            answer = response.choices[0].message.content

        except Exception as e:

            answer = (
                f"⚠️ I couldn't generate an answer right now.\n\n"
                f"Error: `{e}`"
            )

            sources = []


    # --------------------------------------------------------
    # DISPLAY ASSISTANT RESPONSE
    # --------------------------------------------------------

    with st.chat_message("assistant"):

        st.markdown(answer)

        if sources:

            with st.expander("📚 Sources used"):

                for source in sources:

                    st.caption(
                        f"📄 {source}"
                    )


    # --------------------------------------------------------
    # SAVE ASSISTANT MESSAGE
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources
        }
    )
