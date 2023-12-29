## voice-assistant
Simple voice assistant

#### Env
- Implemented and tested on python 3.10
```
    openwakeword==0.5.1
    playsound ==1.3.0
    gTTS==2.5.0
    SpeechRecognition==3.10.1
    torch==2.1.2
    torchaudio==2.1.2
    transformers==4.35.2
    numpy==1.22.0
```

#### Implementation
1. Wakeword detection
2. User query recording
3. Request response module
4. Text-to-speech
   
#### Model info
- Pretrained model for wakewords:[https://github.com/dscripka/openWakeWord/releases/]
- HF model: ["tiiuae/falcon-7b-instruct"]
- TTS: gTTS