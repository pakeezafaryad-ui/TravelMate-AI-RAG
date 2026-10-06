import streamlit as st
import chromadb
import requests
from groq import Groq
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

st.set_page_config(
    page_title="TravelMate AI",
    page_icon="✈️",
    layout="wide"
)

# -----------------------------
# CUSTOM CSS
# -----------------------------
st.markdown("""
<style>
.main {
    padding-top: 1rem;
}

.hero {
    padding: 25px;
    border-radius: 18px;
    margin-bottom: 25px;
    background: linear-gradient(135deg, #16213E, #0F3460);
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

.flight-card {
    padding: 18px;
    border-radius: 15px;
    background: #F5F7FA;
    margin-bottom: 15px;
    border: 1px solid #D9E2EC;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# HEADER
# -----------------------------
st.markdown("""
<div class="hero">
<h1>✈️ TravelMate AI</h1>
<p>Your intelligent travel knowledge assistant</p>
</div>
""", unsafe_allow_html=True)

# -----------------------------
# API KEYS
# -----------------------------
try:
    groq_api_key = st.secrets["GROQ_API_KEY"]
    aviation_api_key = st.secrets["AVIATIONSTACK_API_KEY"]
except Exception:
    st.error(
        "API keys are not configured correctly. "
        "Please check Streamlit Secrets."
    )
    st.stop()

groq_client = Groq(api_key=groq_api_key)

# -----------------------------
# EMBEDDING MODEL
# -----------------------------
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

embedding_model = load_embedding_model()

# -----------------------------
# CHROMA DATABASE
# -----------------------------
@st.cache_resource
def load_vector_database():
    chroma_client = chromadb.Client()

    collection = chroma_client.get_or_create_collection(
        name="travelmate_documents"
    )

    return collection


collection = load_vector_database()

# -----------------------------
# TEXT SPLITTER
# -----------------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

# -----------------------------
# LIVE FLIGHT FUNCTION
# -----------------------------
def get_live_flights():

    url = "https://api.aviationstack.com/v1/flights"

    params = {
        "access_key": aviation_api_key,
        "dep_iata": "LHE",
        "arr_iata": "DXB"
    }

    try:

        response = requests.get(
            url,
            params=params,
            timeout=20
        )

        data = response.json()

        if "error" in data:
            return {
                "error": data["error"].get(
                    "message",
                    "Aviation API returned an error."
                )
            }

        return data

    except requests.RequestException as e:

        return {
            "error": f"Could not connect to flight API: {e}"
        }

# -----------------------------
# DISPLAY LIVE FLIGHTS
# -----------------------------
def show_live_flights():

    st.subheader("✈️ Live Lahore → Dubai Flights")

    with st.spinner("Checking live flight data..."):

        data = get_live_flights()

    if "error" in data:

        st.error(data["error"])

        return

    flights = data.get("data", [])

    if not flights:

        st.info(
            "No current Lahore → Dubai flights were returned "
            "by the live flight API."
        )

        return

    st.caption(
        "Live flight information provided by Aviationstack."
    )

    for flight in flights:

        airline = flight.get("airline", {})
        flight_info = flight.get("flight", {})
        departure = flight.get("departure", {})
        arrival = flight.get("arrival", {})
        status = flight.get("flight_status", "Unknown")

        airline_name = airline.get(
            "name",
            "Unknown airline"
        )

        flight_number = flight_info.get(
            "iata",
            flight_info.get("number", "Unknown")
        )

        departure_airport = departure.get(
            "airport",
            "Allama Iqbal International Airport"
        )

        arrival_airport = arrival.get(
            "airport",
            "Dubai International Airport"
        )

        departure_time = departure.get(
            "scheduled",
            "Not available"
        )

        arrival_time = arrival.get(
            "scheduled",
            "Not available"
        )

        departure_terminal = departure.get(
            "terminal",
            "Not available"
        )

        departure_gate = departure.get(
            "gate",
            "Not available"
        )

        arrival_terminal = arrival.get(
            "terminal",
            "Not available"
        )

        arrival_gate = arrival.get(
            "gate",
            "Not available"
        )

        st.markdown(
            f"""
            <div class="flight-card">

            <h3>✈️ {airline_name} — Flight {flight_number}</h3>

            <p><b>Status:</b> {status}</p>

            <hr>

            <p>
            🛫 <b>Departure:</b> {departure_airport}<br>
            📍 LHE — Lahore
            </p>

            <p>
            🕐 <b>Scheduled departure:</b>
            {departure_time}
            </p>

            <p>
            🚪 <b>Terminal:</b>
            {departure_terminal}
            &nbsp;&nbsp;
            <b>Gate:</b>
            {departure_gate}
            </p>

            <hr>

            <p>
            🛬 <b>Arrival:</b> {arrival_airport}<br>
            📍 DXB — Dubai
            </p>

            <p>
            🕐 <b>Scheduled arrival:</b>
            {arrival_time}
            </p>

            <p>
            🚪 <b>Terminal:</b>
            {arrival_terminal}
            &nbsp;&nbsp;
            <b>Gate:</b>
            {arrival_gate}
            </p>

            </div>
            """,
            unsafe_allow_html=True
        )

# -----------------------------
# SIDEBAR
# -----------------------------
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

    st.subheader("✈️ Live Flights")

    if st.button(
        "Check Lahore → Dubai Flights",
        use_container_width=True
    ):
        show_live_flights()

    st.divider()

    st.caption(
        "Powered by Groq + Hugging Face + "
        "ChromaDB + Aviationstack"
    )

# -----------------------------
# PROCESS PDF
# -----------------------------
if uploaded_files:

    for uploaded_file in uploaded_files:

        file_name = uploaded_file.name

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

            chunks = text_splitter.split_text(text)

            embeddings = embedding_model.encode(
                chunks,
                show_progress_bar=False
            ).tolist()

            ids = [
                f"{file_name}_{i}"
                for i in range(len(chunks))
            ]

            metadatas = [
                {
                    "source": file_name,
                    "chunk": i
                }
                for i in range(len(chunks))
            ]

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

# -----------------------------
# CHAT HISTORY
# -----------------------------
if "messages" not in st.session_state:

    st.session_state.messages = []

# -----------------------------
# WELCOME CARD
# -----------------------------
if not st.session_state.messages:

    st.markdown("""
    <div class="info-card">

    <h3>👋 Welcome to TravelMate AI!</h3>

    <p>
    Ask about travel information from your uploaded
    documents or check live Lahore → Dubai flights
    using the flight API.
    </p>

    </div>
    """, unsafe_allow_html=True)

# -----------------------------
# DISPLAY CHAT
# -----------------------------
for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )

        if message.get("sources"):

            with st.expander(
                "📚 Sources"
            ):

                for source in message["sources"]:

                    st.caption(
                        f"📄 {source}"
                    )

# -----------------------------
# CHAT INPUT
# -----------------------------
question = st.chat_input(
    "Ask TravelMate something..."
)

if question:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):

        st.markdown(question)

    # Detect live flight request
    lower_question = question.lower()

    live_flight_request = (
        ("flight" in lower_question or
         "flights" in lower_question)
        and
        (
            "lahore" in lower_question
            or "lhe" in lower_question
        )
        and
        (
            "dubai" in lower_question
            or "dxb" in lower_question
        )
    )

    if live_flight_request:

        with st.chat_message("assistant"):

            show_live_flights()

        answer = (
            "I displayed the current Lahore → Dubai "
            "flight information from the live Aviationstack API."
        )

        sources = ["Aviationstack Live Flight API"]

    elif collection.count() == 0:

        answer = (
            "📄 Please upload a travel PDF first. "
            "I need a document to use as my knowledge source."
        )

        sources = []

        with st.chat_message("assistant"):

            st.markdown(answer)

    else:

        query_embedding = (
            embedding_model
            .encode(question)
            .tolist()
        )

        number_of_results = min(
            4,
            collection.count()
        )

        results = collection.query(
            query_embeddings=[
                query_embedding
            ],
            n_results=number_of_results
        )

        documents = results["documents"][0]

        metadatas = results["metadatas"][0]

        context = "\n\n---\n\n".join(
            documents
        )

        sources = sorted(
            set(
                metadata["source"]
                for metadata in metadatas
            )
        )

        prompt = f"""
You are TravelMate AI, a helpful travel information assistant.

Answer the user's question using ONLY the information
contained in the provided document context.

IMPORTANT:

1. Do not invent facts.
2. Do not invent flight schedules.
3. Do not invent prices.
4. Do not invent hotel information.
5. Do not invent travel policies.
6. If the answer is not contained in the context,
   say that the information is not available in
   the uploaded travel documents.
7. Keep the answer clear and useful.

DOCUMENT CONTEXT:

{context}

USER QUESTION:

{question}
"""

        try:

            response = groq_client.chat.completions.create(

                model="openai/gpt-oss-20b",

                messages=[
                    {
                        "role": "system",
                        "content":
                        "You are TravelMate AI. "
                        "Answer only from the supplied context."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],

                temperature=0.2,

                max_tokens=800
            )

            answer = (
                response
                .choices[0]
                .message
                .content
            )

        except Exception as e:

            answer = (
                "⚠️ I couldn't generate an answer right now.\n\n"
                f"Error: `{e}`"
            )

            sources = []

        with st.chat_message("assistant"):

            st.markdown(answer)

            if sources:

                with st.expander(
                    "📚 Sources used"
                ):

                    for source in sources:

                        st.caption(
                            f"📄 {source}"
                        )

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources
        }
    )
