import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound

st.set_page_config(page_title="YouTube Transcript Fetcher", layout="wide")
st.title("🎬 YouTube Transcript Fetcher")

video_url = st.text_input("Enter YouTube Video URL:", "")
language = st.text_input("Transcript language (e.g., en, hi):", "en")

if st.button("Load Transcript"):
    try:
        # Extract video ID
        if "v=" in video_url:
            video_id = video_url.split("v=")[1].split("&")[0]
        else:
            st.error("Invalid YouTube URL format.")
            st.stop()

        transcript = None

        try:
            # Try normal transcript
            transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=[language])
        except (TranscriptsDisabled, NoTranscriptFound):
            # Try auto-generated transcript
            st.warning("No manual transcript found. Trying auto-generated subtitles...")
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            for t in transcript_list:
                if t.is_generated and language in t.language_code:
                    transcript = t.fetch()
                    break

        if not transcript:
            st.error("No transcript available in the requested language (manual or auto).")
        else:
            full_text = " ".join([entry["text"] for entry in transcript])
            st.subheader("Transcript:")
            st.write(full_text)

    except Exception as e:
        st.error(f"Error fetching transcript: {e}")

