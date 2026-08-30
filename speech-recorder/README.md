# Speech Recorder

Minimalna aplikacja Android działająca lokalnie. Po ręcznym uruchomieniu utrzymuje foreground service mikrofonu, analizuje dźwięk w RAM i zapisuje WAV tylko po wykryciu aktywności głosowej.

- wersja 1.1.0
- 16 kHz mono PCM WAV
- 5 s pre-buffer
- 8 s ciszy kończy klip
- dynamiczny próg szumu + energia + zero-crossing rate
- Android 10+
- brak sieci i usług chmurowych
- nagrania w `Music/SpeechRecorder`
- foreground service z `START_STICKY`
- `stopWithTask=false`, więc usunięcie aplikacji z ostatnich nie zatrzymuje serwisu
- automatyczna ponowna inicjalizacja `AudioRecord` po błędzie
- po restarcie telefonu powiadomienie pozwala wznowić mikrofon jednym tapnięciem
- własna ikona mikrofonu dla launchera i powiadomienia

Android może nadal zatrzymać aplikację po Force stop, odebraniu uprawnienia mikrofonu albo w wyniku ograniczeń systemowych. Android 14+ nie pozwala aplikacji uruchomić mikrofonowego foreground service bezpośrednio z `BOOT_COMPLETED`, dlatego po restarcie wymagane jest świadome tapnięcie powiadomienia przez użytkownika.
