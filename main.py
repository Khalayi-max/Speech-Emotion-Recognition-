import streamlit as st
from PIL import Image
from pydub import AudioSegment
import numpy as np
import soundfile
import pickle
import librosa
import sounddevice as sd
from scipy.io.wavfile import write

st.write("""
# PYTHON PROJECT
## SPEECH EMOTION RECOGNITION
""")
image = Image.open("title.jpg")
st.image(image, use_column_width=True)
st.write("""
### What is Speech Emotion Recognition?

Speech Emotion Recognition, abbreviated as SER, is the act of attempting to recognize human emotion and affective states from speech. This is capitalizing on the fact that voice often reflects underlying emotion through tone and pitch. This is also the phenomenon that animals like dogs and horses employ to be able to understand human emotion.
SER is tough because emotions are subjective and annotating audio is challenging.

In this Python project, we recognize emotions from speech. The model delivers an accuracy of 73.96%. That’s good enough for us !!.
""")

image = Image.open("Speech_Recognition.jpg")
st.image(image, use_column_width=True)
loaded_model = pickle.load(open('finalized_model.sav', 'rb'))
        
def extract_feature(file_name, mfcc, chroma, mel):
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate=sound_file.samplerate
        if chroma:
            stft=np.abs(librosa.stft(X))
            result=np.array([])
        if mfcc:
            mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result=np.hstack((result, mfccs))
        if chroma:
            chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
            result=np.hstack((result, chroma))
        if mel:
            mel=np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
            result=np.hstack((result, mel))
    return result

st.sidebar.write("How would you like to give input?")
option = st.sidebar.selectbox("",["Choose","Demo App","Upload File","Record Audio"])

if option=="Demo App":
    audio_bytes = open("03-01-01-01-01-02-02.wav", 'rb').read()
    st.write("#### Input sound:")
    st.audio(audio_bytes, format=f'audio/sav', start_time=0)
    feature = np.array(extract_feature("03-01-01-01-01-02-02.wav", mfcc=True, chroma=True, mel=True)).reshape(1, -1)
    result = loaded_model.predict(feature)
    st.write("#### Prediction:")
    st.write("##### The emotion of sound is " + result[0])
elif option=="Upload File":
    file = st.sidebar.file_uploader("Please Upload Audio File Here",type=["wav"])
    if file is None:
        st.write("### Please upload a wav file")
    else:
        wav = AudioSegment.from_wav(file)
        wav.export("extracted.wav", format='wav')
        audio_bytes = open("extracted.wav", 'rb').read()
        st.write("#### Input sound:")
        st.audio(audio_bytes, format=f'audio/sav', start_time=0)
        feature = np.array(extract_feature("extracted.wav", mfcc=True, chroma=True, mel=True)).reshape(1, -1)
        result = loaded_model.predict(feature)
        st.write("#### Prediction:")
        st.write("##### The emotion of sound is " + result[0])
elif option=="Record Audio":
    fs = 44100  # Sample rate
    seconds = 10  # Duration of recording
    sd.default.device = -1
    #st.write(sd.DeviceList())
    if (st.sidebar.button("Start recording")):
        myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
        sd.wait()  
        write('output.wav', fs, myrecording)
        audio_bytes = open("output.wav", 'rb').read()
        st.write("#### Input sound:")
        st.audio(audio_bytes, format=f'audio/sav', start_time=0)
        feature = np.array(extract_feature("output.wav", mfcc=True, chroma=True, mel=True)).reshape(1, -1)
        result = loaded_model.predict(feature)
        st.write("#### Prediction:")
        st.write("##### The emotion of sound is " + result[0])
else:
    st.write("#### For testing this website: ")
    st.write("* Upload any sample file or")
    st.write("* Try the demo app or")
    st.write("* Try live recording in sidebar")
    
st.text("")
st.text("")    
st.text("")
st.text("")
st.text("Made by Atinderpal Kaur")
