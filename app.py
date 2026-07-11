import streamlit as st
from transformers import pipeline
import re
import yt_dlp
import requests
import json
import os
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

st.set_page_config(page_title="YouTube Transcript + RAG", layout="wide")

st.title("🎬 YouTube Transcript + RAG")

# Initialize models
@st.cache_resource
def load_embedding_model():
    """Load sentence transformer model for embeddings"""
    return SentenceTransformer('all-MiniLM-L6-v2')

# Initialize OpenAI client
def get_openai_client():
    """Get OpenAI client with API key from environment or session state"""
    # First try environment variable
    api_key = os.getenv('OPENAI_API_KEY')

    # If not in environment, try session state
    if not api_key and 'openai_api_key' in st.session_state:
        api_key = st.session_state.openai_api_key

    if not api_key:
        return None
    return OpenAI(api_key=api_key)

def create_chunks(text, chunk_size=500, overlap=50):
    """Split text into overlapping chunks"""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    
    return chunks

def create_vector_index(chunks, embedding_model):
    """Create FAISS index from text chunks"""
    # Generate embeddings
    embeddings = embedding_model.encode(chunks)
    
    # Create FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
    
    # Normalize embeddings for cosine similarity
    faiss.normalize_L2(embeddings)
    index.add(embeddings.astype('float32'))
    
    return index, embeddings

def retrieve_relevant_chunks(question, chunks, index, embedding_model, top_k=3):
    """Retrieve most relevant chunks for a question"""
    # Encode question
    question_embedding = embedding_model.encode([question])
    faiss.normalize_L2(question_embedding)
    
    # Search for similar chunks
    scores, indices = index.search(question_embedding.astype('float32'), top_k)
    
    # Get relevant chunks
    relevant_chunks = [chunks[i] for i in indices[0] if i < len(chunks)]
    
    return relevant_chunks, scores[0]

def ask_openai_question(client, question, relevant_chunks):
    """Ask question using OpenAI LLM with only relevant chunks"""
    try:
        # Combine relevant chunks
        context = "\n\n".join(relevant_chunks)

        prompt = f"""Based on the following relevant parts of a YouTube video transcript, please answer the question accurately and concisely.

RELEVANT TRANSCRIPT PARTS:
{context}

QUESTION: {question}

Please provide a clear, accurate answer based only on the information in the provided transcript parts. If the answer cannot be found in these parts, say so."""

        response = client.chat.completions.create(
            model="gpt-4o-mini",  # OpenAI model
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions based on provided transcript parts accurately and concisely."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=300
        )

        return response.choices[0].message.content, 0.95  # High confidence for OpenAI
    except Exception as e:
        st.error(f"Error with OpenAI API: {e}")
        return None, 0.0

def parse_json_captions(json_text):
    """Parse JSON-formatted captions"""
    try:
        data = json.loads(json_text)
        transcript_parts = []
        
        if 'events' in data:
            for event in data['events']:
                if 'segs' in event:
                    for seg in event['segs']:
                        if 'utf8' in seg:
                            transcript_parts.append(seg['utf8'])
        
        return ' '.join(transcript_parts)
    except:
        return None

def parse_srt_captions(srt_text):
    """Parse SRT-formatted captions"""
    lines = srt_text.split('\n')
    transcript_parts = []
    
    for line in lines:
        line = line.strip()
        # Skip empty lines, numbers, and timestamp lines
        if line and not line.isdigit() and '-->' not in line and not re.match(r'^\d{2}:\d{2}:\d{2}', line):
            transcript_parts.append(line)
    
    return ' '.join(transcript_parts)

def get_best_english_key(captions_dict):
    """Return the best available English caption key from a captions dict."""
    if not captions_dict:
        return None
    # Prefer exact 'en', then any key starting with 'en'
    if 'en' in captions_dict:
        return 'en'
    for key in captions_dict:
        if key.startswith('en'):
            return key
    return None

def get_transcript_with_ytdlp(video_url):
    """Get transcript using yt-dlp method"""
    try:
        ydl_opts = {
            'writesubtitles': True,
            'writeautomaticsub': True,
            'skip_download': True,
            'quiet': True,
            'no_warnings': True
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=False)

            # Try manual subtitles first
            manual_key = get_best_english_key(info.get('subtitles', {}))
            if manual_key:
                subtitle_url = info['subtitles'][manual_key][0]['url']
                response = requests.get(subtitle_url)
                if response.status_code == 200:
                    subtitle_text = response.text

                    json_result = parse_json_captions(subtitle_text)
                    if json_result:
                        return json_result, "Manual captions (JSON)"

                    srt_result = parse_srt_captions(subtitle_text)
                    if srt_result:
                        return srt_result, "Manual captions (SRT)"

            # Try automatic subtitles if manual ones aren't available
            auto_key = get_best_english_key(info.get('automatic_captions', {}))
            if auto_key:
                subtitle_url = info['automatic_captions'][auto_key][0]['url']
                response = requests.get(subtitle_url)
                if response.status_code == 200:
                    subtitle_text = response.text

                    json_result = parse_json_captions(subtitle_text)
                    if json_result:
                        return json_result, f"Auto-generated captions (JSON, lang: {auto_key})"

                    srt_result = parse_srt_captions(subtitle_text)
                    if srt_result:
                        return srt_result, f"Auto-generated captions (SRT, lang: {auto_key})"

            return None, "No captions found"

    except Exception as e:
        st.error(f"Error extracting transcript: {e}")
        return None, "Error"

# Add OpenAI API setup in sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("🔑 OpenAI API Setup")

# Check if API key is already set
api_key = os.getenv('OPENAI_API_KEY')

if api_key:
    st.sidebar.success("✅ OPENAI_API_KEY from environment!")
    st.sidebar.info(f"Key: {api_key[:10]}...")
else:
    st.sidebar.warning("⚠️ OPENAI_API_KEY not found in environment!")

    # Allow manual input
    manual_api_key = st.sidebar.text_input(
        "Enter your OpenAI API Key:",
        type="password",
        help="Get your API key from platform.openai.com"
    )

    if manual_api_key:
        st.session_state.openai_api_key = manual_api_key
        st.sidebar.success("✅ API Key saved!")
        st.sidebar.info("You can now use OpenAI for better answers!")

    st.sidebar.markdown("""
    **To get your API key:**
    1. Go to [platform.openai.com](https://platform.openai.com/api-keys)
    2. Sign in to your account
    3. Create a new secret key

    **Model used:** gpt-4o-mini
    """)

# Step 1: Input YouTube URL
video_url = st.text_input("Enter YouTube Video URL:")

if st.button("Get Transcript"):
    if not video_url:
        st.error("Please enter a valid YouTube URL.")
    else:
        with st.spinner("Extracting transcript..."):
            try:
                # Extract video ID from URL
                video_id_match = re.search(r'(?:v=|\/)([0-9A-Za-z_-]{11}).*', video_url)
                if not video_id_match:
                    st.error("Invalid YouTube URL. Please provide a valid YouTube video URL.")
                    st.stop()
                
                video_id = video_id_match.group(1)
                st.info(f"Video ID: {video_id}")
                
                # Get transcript using yt-dlp
                transcript_text, caption_type = get_transcript_with_ytdlp(video_url)
                
                if not transcript_text:
                    st.error("Could not retrieve transcript.")
                    st.info("This video may not have available captions. Common reasons:")
                    st.info("• Video doesn't have captions/transcripts enabled")
                    st.info("• Video is private or restricted")
                    st.info("• Video is too new or too old")
                    st.info("• Region restrictions")
                    st.info("\n💡 Try a different YouTube video that has captions enabled.")
                    st.stop()
                
                if not transcript_text.strip():
                    st.error("Transcript is empty. This video may not have proper captions.")
                    st.stop()
                
                st.success(f"Transcript extracted successfully using {caption_type}!")
                st.text_area("Transcript:", transcript_text, height=300)
                
                # Create chunks and vector index for RAG
                with st.spinner("Creating RAG index..."):
                    embedding_model = load_embedding_model()
                    chunks = create_chunks(transcript_text)
                    index, embeddings = create_vector_index(chunks, embedding_model)
                    
                    # Store in session state
                    st.session_state.transcript = transcript_text
                    st.session_state.chunks = chunks
                    st.session_state.index = index
                    st.session_state.embedding_model = embedding_model
                
                st.success(f"✅ RAG index created with {len(chunks)} chunks!")
                
            except Exception as e:
                st.error(f"Error: {e}")
                st.info("Try a different YouTube video that has captions enabled.")

# Step 2: Q&A Section (only show if transcript is available)
if 'transcript' in st.session_state and st.session_state.transcript:
    st.subheader("Ask Questions about the Video")
    
    # Check if OpenAI is available
    openai_client = get_openai_client()

    if openai_client:
        st.success("✅ OpenAI connected - Efficient RAG answers available!")
        question = st.text_input("Enter your question:")

        if question:
            with st.spinner("Retrieving relevant chunks and generating answer..."):
                try:
                    # Retrieve relevant chunks
                    relevant_chunks, scores = retrieve_relevant_chunks(
                        question,
                        st.session_state.chunks,
                        st.session_state.index,
                        st.session_state.embedding_model
                    )

                    # Show relevant chunks (for transparency)
                    with st.expander("🔍 Relevant transcript parts used:"):
                        for i, (chunk, score) in enumerate(zip(relevant_chunks, scores)):
                            st.write(f"**Chunk {i+1} (Relevance: {score:.2f}):**")
                            st.write(chunk)
                            st.write("---")

                    # Generate answer with only relevant chunks
                    answer, confidence = ask_openai_question(openai_client, question, relevant_chunks)
                    if answer:
                        st.write("**Answer:**", answer)
                        st.write(f"**Confidence:** {confidence:.2%}")
                        st.info(f"💡 Used {len(relevant_chunks)} relevant chunks instead of full transcript (much more efficient!)")
                    else:
                        st.error("Failed to get answer from OpenAI.")
                except Exception as e:
                    st.error(f"Error generating answer: {e}")
    else:
        st.warning("⚠️ OpenAI API key not found. Using fallback Hugging Face model.")
        question = st.text_input("Enter your question:")
        
        if question:
            with st.spinner("Generating answer with Hugging Face model..."):
                try:
                    qa_model = pipeline(
                        "question-answering",
                        model="distilbert-base-uncased-distilled-squad"
                    )
                    answer = qa_model(question=question, context=st.session_state.transcript)
                    st.write("**Answer:**", answer["answer"])
                    st.write(f"**Confidence:** {answer['score']:.2%}")
                except Exception as e:
                    st.error(f"Error generating answer: {e}")

# Add a test section with a known working video
st.sidebar.markdown("---")
st.sidebar.subheader("🧪 Test with Known Working Video")
if st.sidebar.button("Test with TED Talk"):
    st.session_state.test_url = "https://www.youtube.com/watch?v=8jPQjjsBbIc"
    st.rerun()

# Display test URL if available
if 'test_url' in st.session_state:
    st.info(f"Test URL loaded: {st.session_state.test_url}")
    # Clear the test URL after displaying
    del st.session_state.test_url


