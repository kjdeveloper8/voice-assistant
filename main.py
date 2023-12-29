''' voice assistant '''
import os
import pyaudio
import requests
import openwakeword
import numpy as np
import speech_recognition as sr
from typing import Optional
from openwakeword.model import Model
from gtts import gTTS
from tempfile import NamedTemporaryFile
from playsound import playsound
from huggingface_hub import HfFolder

HF_TOKEN = os.getenv("HF_API") # hf api token
DIR_NAME = "pretrained_model"
# Get microphone stream
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1280
RECORD_SECONDS = 5

text = "Start speaking ..."
model_path = f"{DIR_NAME}/alexa_v0.1.onnx"

def detect_wakeword(model_path:str = model_path):
    ''' Detects wakeword 
        args:
            model_path (str): pretrained model path
    '''
    audio = pyaudio.PyAudio()
    mic_stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

    model = Model(wakeword_models=[model_path])

    # Get predictions for the frame from streaming audio
    while True:
        frame = np.frombuffer(mic_stream.read(CHUNK), dtype=np.int16)
        prediction = model.predict(frame)
        if prediction['alexa_v0.1'] >= 0.7:
            mic_stream.stop_stream()
            audio.terminate()
            return prediction

def record_voice():
    ''' records voice from microphone '''
    r = sr.Recognizer()
    with sr.Microphone() as source:
        audio = r.listen(source)
    try:
        record = r.recognize_google(audio)
        return record
    except sr.UnknownValueError:
        print("Sorry, could not understands it.")

def speak(text:str, lang:str="en"):
    ''' speech to text (female)'''
    gTTS(text=text, lang=lang).write_to_fp(voice := NamedTemporaryFile())
    playsound(voice.name)
    voice.close()

def query(text, model_id="tiiuae/falcon-7b-instruct"):
    ''' Request a query to hf model and get response '''
    api_url = f"https://api-inference.huggingface.co/models/{model_id}"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {"inputs": text}

    # print(f"Query --> {text}")
    response = requests.post(api_url, headers=headers, json=payload)
    return response.json()[0]["generated_text"]

def response():
    try:
        pred_wakeword = detect_wakeword()
        print(f"wakeword score-- {pred_wakeword['alexa_v0.1']}") 
        if pred_wakeword:
            speak("Yes, how can i help you")
            print(f"Now {text}")
            record = record_voice()
            print(f"record -->  {record}")
            answer = query(record) #res[0]['generated_text'].split('\n')[-1]
            answer = answer.split('\n')[-1]
            print(answer)
            if answer:
                speak(answer)
        else:
            return
    except KeyboardInterrupt:
            return

if __name__ == "__main__":
    print('Say \'alexa\' ...')
    response()




    