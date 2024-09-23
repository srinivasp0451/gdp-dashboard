import streamlit as st
import os
import speech_recognition as sr
from pytube import YouTube
from pydub import AudioSegment
from pydub.utils import which


import os

# Check if ffmpeg is installed
if os.system("ffmpeg -version") != 0:
    os.system("apt-get update && apt-get install -y ffmpeg")




# Ensure ffmpeg is configured correctly
AudioSegment.converter = which("ffmpeg")

# Function to download audio from YouTube video
def download_audio_from_youtube(url, output_path='audio.mp4'):
    yt = YouTube(url)
    audio_stream = yt.streams.filter(only_audio=True).first()
    audio_stream.download(filename=output_path)
    return output_path

# Function to convert audio to WAV format using pydub
def convert_audio_to_wav(input_path, output_path='audio.wav'):
    audio = AudioSegment.from_file(input_path)
    audio.export(output_path, format='wav')
    return output_path

# Function to convert audio to text using SpeechRecognition
def audio_to_text(audio_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio = recognizer.record(source)
    try:
        text = recognizer.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        return "Google Speech Recognition could not understand the audio"
    except sr.RequestError as e:
        return f"Could not request results from Google Speech Recognition service; {e}"

# Streamlit app
def main():
    st.title("YouTube Audio to Text Converter")
    st.write("Enter the URL of a YouTube video and convert its audio to text.")

    # Input for YouTube URL
    youtube_url = st.text_input("YouTube Video URL", "")

    if st.button("Convert to Text"):
        if youtube_url:
            with st.spinner("Downloading audio from YouTube..."):
                try:
                    audio_file = download_audio_from_youtube(youtube_url, 'audio.mp4')
                    st.success("Audio downloaded successfully.")
                    
                    # Convert to WAV
                    with st.spinner("Converting audio to WAV format..."):
                        wav_file = convert_audio_to_wav(audio_file, 'audio.wav')
                        st.success("Audio converted to WAV format.")

                    # Convert to text
                    with st.spinner("Converting audio to text..."):
                        transcribed_text = audio_to_text(wav_file)
                        st.success("Audio converted to text successfully.")

                    # Display the transcribed text
                    st.subheader("Transcribed Text")
                    st.text_area("Text", transcribed_text, height=200)

                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
        else:
            st.warning("Please enter a valid YouTube URL.")

if __name__ == "__main__":
    main()
