from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse, parse_qs
import streamlit as st

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
            # Try to fetch the transcript in the specified language (e.g., English)
            transcript = available_transcripts.find_transcript([language_code])
        except:
            # If the transcript in the specified language isn't available, fetch any available transcript
            #print(f"No transcript found for the language '{language_code}'. Fetching available transcript.")
            transcript = available_transcripts.find_generated_transcript([language_code, 'hi'])  # Fallback to Hindi or other auto-generated language

        # Get the translated transcript (if needed) or the original one
        transcript_text = "\n".join([entry['text'] for entry in transcript.fetch()])
        #print(transcript_text)
        # Save the transcript to a file
        #with open(f"{video_id}_transcript.txt", "w", encoding="utf-8") as file:
            #file.write(transcript_text)

        print("Transcript saved successfully!")
        return transcript_text
    except Exception as e:
        st.write("error occurred: {e}")
st.title("Youtube transcript")
youtube_url = st.text_input("Enter Url")
#youtube_url = "https://youtu.be/uYwmKK2J3L8?si=UhIKQWZI1BgCE9O9"
text = get_youtube_transcript(youtube_url, language_code='en')
st.write(text)
