import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse, parse_qs

# Function to extract the video ID from a YouTube URL
def extract_video_id(youtube_url):
    parsed_url = urlparse(youtube_url)
    video_id = parse_qs(parsed_url.query).get("v")
    return video_id[0] if video_id else parsed_url.path.split('/')[-1]

# Function to get transcript from a YouTube video, allowing language specification
def get_youtube_transcript(youtube_url, language_code='en'):
    try:
        video_id = extract_video_id(youtube_url)

        # Fetch the available transcripts for the video
        available_transcripts = YouTubeTranscriptApi.list_transcripts(video_id)

        try:
            # Try to fetch the transcript in the specified language
            transcript = available_transcripts.find_transcript([language_code])
        except:
            # If the transcript in the specified language isn't available, fetch any available transcript
            st.write(f"No transcript found for the language '{language_code}'. Fetching available transcript.")
            transcript = available_transcripts.find_generated_transcript([language_code, 'hi'])  # Fallback to Hindi or other auto-generated language

        # Get the translated transcript (if needed) or the original one
        transcript_text = "\n".join([entry['text'] for entry in transcript.fetch()])
        return transcript_text

    except Exception as e:
        return f"An error occurred: {e}"

# Streamlit app
st.title("YouTube Transcript Fetcher")

# Input for YouTube video URL
youtube_url = st.text_input("Enter YouTube URL:")

# Language selection (defaults to English 'en')
language_code = st.text_input("Enter Language Code (e.g., 'en', 'hi'):", value='en')

# Fetch transcript when button is clicked
if st.button("Fetch Transcript"):
    if youtube_url:
        transcript = get_youtube_transcript(youtube_url, language_code)
        if transcript:
            st.text_area("Transcript:", transcript, height=400)
    else:
        st.write("Please enter a valid YouTube URL.")
