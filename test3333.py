import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse, parse_qs

# Function to extract the video ID from a YouTube URL
def extract_video_id(youtube_url):
    parsed_url = urlparse(youtube_url)
    video_id = parse_qs(parsed_url.query).get("v")
    return video_id[0] if video_id else parsed_url.path.split('/')[-1]

# Function to get transcript from a YouTube video, fixed to English ('en')
def get_youtube_transcript(youtube_url):
    try:
        video_id = extract_video_id(youtube_url)

        # Fetch the available transcripts for the video
        available_transcripts = YouTubeTranscriptApi.list_transcripts(video_id)

        try:
            # Try to fetch the transcript in English
            transcript = available_transcripts.find_transcript(['en'])
        except:
            # If the English transcript isn't available, fetch auto-generated transcripts
            st.write(f"No English transcript found. Fetching available auto-generated transcript.")
            transcript = available_transcripts.find_generated_transcript(['en', 'hi'])  # Fallback to auto-generated transcript

        # Get the transcript and return it
        transcript_text = "\n".join([entry['text'] for entry in transcript.fetch()])
        return transcript_text

    except Exception as e:
        return f"An error occurred: {e}"

# Streamlit app
st.title("YouTube Transcript Fetcher")

# Input for YouTube video URL
youtube_url = st.text_input("Enter YouTube URL:")

# Fetch transcript when button is clicked
if st.button("Fetch Transcript"):
    if youtube_url:
        transcript = get_youtube_transcript(youtube_url)
        if transcript:
            st.text_area("Transcript:", transcript, height=400)
    else:
        st.write("Please enter a valid YouTube URL.")
