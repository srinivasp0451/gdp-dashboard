import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import NoTranscriptFound, TranscriptsDisabled
from urllib.parse import urlparse, parse_qs
from googletrans import Translator

# Function to extract the video ID from a YouTube URL
def extract_video_id(youtube_url):
    parsed_url = urlparse(youtube_url)
    video_id = parse_qs(parsed_url.query).get("v")
    return video_id[0] if video_id else parsed_url.path.split('/')[-1]

# Function to get transcript from a YouTube video, allowing language specification
def get_youtube_transcript(video_id, language_code='en'):
    try:
        # Fetch the available transcripts for the video
        available_transcripts = YouTubeTranscriptApi.list_transcripts(video_id)
        
        try:
            # Try to fetch the transcript in the specified language (e.g., English)
            transcript = available_transcripts.find_transcript([language_code])
        except NoTranscriptFound:
            # If the transcript in the specified language isn't available, fetch any available transcript
            st.warning(f"No transcript found for the language '{language_code}'. Fetching auto-generated transcript.")
            transcript = available_transcripts.find_generated_transcript([language_code, 'hi'])  # Fallback to Hindi
        
        return transcript.fetch()
    except NoTranscriptFound:
        st.error("No transcript is available for this video.")
        return None
    except TranscriptsDisabled:
        st.error("Transcripts are disabled for this video.")
        return None
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None

# Function to translate transcript using Google Translator
def translate_transcript(transcript, src_language='hi', dest_language='en'):
    translator = Translator()
    translated_text = []
    for entry in transcript:
        translated_text.append(translator.translate(entry['text'], src=src_language, dest=dest_language).text)
    return translated_text

# Streamlit App
st.title("YouTube Video Transcription and Translation")

# Input for YouTube URL
youtube_url = st.text_input("Enter YouTube URL:", placeholder="https://www.youtube.com/watch?v=example")

if youtube_url:
    video_id = extract_video_id(youtube_url)
    st.write(f"Extracted Video ID: {video_id}")

    # Get Transcript
    if st.button("Get Transcript"):
        transcript = get_youtube_transcript(video_id, language_code='hi')  # Try to get Hindi transcript
        if transcript:
            st.write("Original Transcript in Hindi:")
            st.write("\n".join([entry['text'] for entry in transcript]))

            # Translate Transcript to English
            st.write("Translating to English...")
            translated_transcript = translate_transcript(transcript, src_language='hi', dest_language='en')
            st.write("Translated Transcript in English:")
            st.write("\n".join(translated_transcript))
