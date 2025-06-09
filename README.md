# Portfolio Optimizer
Projekt na przedmiot SUML, wykonany przez studentów: Kacper Kuc (s25822), Jakub Ośka (s24187) i Cezary Klasicki (s25819)

## Rodzaj aplikacji:
Aplikacja desktopowa
- Język backend'u python
- Frontend zbudowany przy użyciu tinkera
## Opis aplikacji:

## Dataset:
- Historyczne dane instrumentów finansowych (akcji, funduszy) pobierane z biblioteki YFinance.
- Dane na potrzeby analiz będą pobierane "on-demand" przez użytkownika, z wykorzystaniem darmowego API, będącego elementem pakietu yfiannce.
- Zapytania zwracają serię danych na podstawie podanych "ticker'ów" (oznaczeń instrumentów finansowych) - domyślnie wprowadzi je użytkownik.

### Predykcja:
Aplikacja będzie służyć do predykcji przyszłych cen wybranych instrumentów finansowych na podstawie danych historycznych.

### Uzasadnienie funkcjonalności aplikacji:
Aplikacja będzie pomocą dla prywatnych inwestorów, którzy chcieli by oszacować / porównać przewidywane ceny akcji/funduszu na podstawie danych historycznych. Jako wersja desktopowa, dodana zostanie:
- możliwość tworzeni kont użytkowników.
- możliwość zapisywania ewentualnych wyników pozwoli także porównywać starsze analizy z nowszymi, w celu obserwacji wpływu zmian na rynku.
- Wyniki będą wizualizowane za pomocą tabel i komponentów graficznych

Efektem będzie prosta w obsłudze aplikacja, dająca inwestorowi dostęp do możliwości, jakie daje uczenie maszynowe bez konieczności dokładnej znajomości strony technicznej wymaganej do samodzielnej implementacji takiego rozwiązania.

### Zastosowany model uczenia maszynowego:
Sekwencyjna sieć neuronowa LSTM bazująca na uporządkowanych danych historycznych. Użyta zostanie wersja z pakietu Keras. Będzie trenowana na zawołanie, na podstawie danych wybranych przez użytkownika.