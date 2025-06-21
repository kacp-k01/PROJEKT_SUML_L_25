# Portfolio Optimizer
Projekt na przedmiot SUML, wykonany przez studentów: 
- Kacper Kuc (s25822)
- Jakub Ośka (s24187)
- Cezary Klasicki (s25819)

## Rodzaj aplikacji:
Aplikacja desktopowa
- Język backend'u python
- Frontend zbudowany przy użyciu tinkera

## Opis aplikacji:
Dokładny opis aplikacji i raport z jej działania znajduje się w pliku `raport/Raport_Projekt_SUML.pdf`.

### Uruchomienie aplikacji:
Wymagania bibliotek zapisano w pliku `requirements.txt`. Aplikacja została utworzona na Pythonie `3.12`
Aplikację można uruchomić:
- poprzez plik `projekt_suml.py` znajdujący się w katalogu głównym projektu (`portfolio-optimizer/`)
- lub przygotowując plik .exe do uruchomienia lokalnego:
  - Przechodzimy do katalogu głównego projektu:
    ```shell
    cd .\portfolio-optimizer
    ```
  - Najpierw instalujemy bibliotekę pyinstaller (jeśli jej jeszcze nie mamy) - podczas testowego wypalenia projektu pracowano na wersji **6.12.0**:
    ```shell
    pip install pyinstaller
    ```
  - Potem przygotowujemy plik wykonawczy .exe (_podmoduły **tensorflow** i **keras** zostały dodane jako hidden-importy, aby uniknąć problemów z importem podczas uruchamiania_):
    ```shell
    pyinstaller --noconfirm --onefile --noconsole projekt_suml.py --hidden-import=tf_keras.src.engine.base_layer_v1 --hidden-import=tensorflow.keras
    ```
  - Po wykonaniu powyższego polecenia, w katalogu `dist/` zostanie utworzony plik `projekt_suml.exe`, który można uruchomić lokalnie.


## Dataset:
- Historyczne dane instrumentów finansowych (akcji, funduszy) pobierane z biblioteki Yahoo Finance.
- Dane na potrzeby analiz będą pobierane "on-demand" przez użytkownika, z wykorzystaniem darmowego API.
- Zapytania zwracają serię danych na podstawie podanych "ticker'ów" (oznaczeń instrumentów finansowych) - domyślnie wprowadzi je użytkownik.

### Predykcja:
Aplikacja będzie służyć do predykcji przyszłych cen wybranych instrumentów finansowych na podstawie danych historycznych.

### Uzasadnienie funkcjonalności aplikacji:
Aplikacja będzie pomocą dla prywatnych inwestorów, którzy chcieli by oszacować / porównać przewidywane ceny akcji/funduszu na podstawie danych historycznych. Jako wersja desktopowa, dodana zostanie:
- możliwość zapisywania ewentualnych wyników pozwoli także porównywać starsze analizy z nowszymi, w celu obserwacji wpływu zmian na rynku.
- Wyniki będą wizualizowane za pomocą raportu tekstowego i komponentów graficznych

Efektem będzie prosta w obsłudze aplikacja, dająca inwestorowi dostęp do możliwości, jakie daje uczenie maszynowe bez konieczności dokładnej znajomości strony technicznej wymaganej do samodzielnej implementacji takiego rozwiązania.

### Zastosowany model uczenia maszynowego:
Sekwencyjna sieć neuronowa LSTM bazująca na uporządkowanych danych historycznych. Użyta zostanie wersja z pakietu Keras. Będzie trenowana na zawołanie, na podstawie danych wybranych przez użytkownika.
