# Speech Recorder

Minimalna aplikacja Android działająca lokalnie. Po ręcznym uruchomieniu utrzymuje foreground service mikrofonu, analizuje dźwięk w RAM i zapisuje WAV tylko po wykryciu aktywności głosowej.

- wersja 1.0.1
- 16 kHz mono PCM WAV
- 5 s pre-buffer
- 8 s ciszy kończy klip
- dynamiczny próg szumu + energia + zero-crossing rate
- Android 10+
- brak sieci i usług chmurowych
- nagrania w `Music/SpeechRecorder`
