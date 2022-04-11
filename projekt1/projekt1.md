# Pierwszy projekt. Złożoność metody sympleks

Złożoność metody sympleks zależy od przyjętej reguły wyboru zmiennych wchodzących i wychodzących. Tematem pierwszego projektu jest eksperymentalne zbadanie tej zależności na nietrywialnych przykładach w niskich wymiarach (kilkanaście zmiennych problemu w postaci równościowej).

## Na zaliczenie

Należy wybrać problem rozsądnej wielkości. 
Najlepiej, by zbiór rozwiązań dopuszczalnych miał kilkanaście wierzchołków i był trójwymiarowy. Należy przeprowadzić pełną analizę tego problemu odpowiadając między innymi na poniższe pytania.

1. Ile jest zbiorów bazowych dopuszczalnych dla tego problemu?
2. Jakie są możliwe przejścia pomiędzy zbiorami bazowymi? (Chodzi o przejścia, które można byłoby wykonać posługując się metodą sympleks.)
3. Czy jest możliwe wejście w cykl?
4. Czy zbiór rozwiązań dopuszczalnych jest ograniczony?
5. Jak wygląda zbiór rozwiązań dopuszczalnych?
6. Ile kroków wykona metoda sympleks dla różnych standardowych reguł przejścia (takich jak reguła Blanda, largest increase, largest coefficient, steepest edge) i różnych wierzchołków startowych? (Chodzi o pełne zestawienie: tabela, w której kolumny odpowiadają regułom przejścia i wiersze wierzchołkom startowym.)
7. Jaka będzie średnia i maksymalna liczba kroków dla tych reguł?

Należy też wymyślić i zbadać (punkty 6, 7) własną regułę przejścia (może być a może nawet powinna być "głupia").

## Na wyższe oceny

Na wyższą ocenę należy przeprowadzić powyższą analizę dla rodziny lub dwóch rodzin przykładów rosnącej wielkości generowanych w podobny sposób. Należy przeprowadzić analizę złożoności każdej z reguł (także własnej) wraz ze wzrostem rozmiaru problemu. 

Na czwórkę wystarczy zbadanie jednej rodziny. Na piątkę - dwóch.
Jedna z rodzin powinna być generowana w sposób losowy (np. losowe macierze współczynników o coraz większych rozmiarach, choć to nie jest akurat najlepszy pomysł!), druga powinna być konstruowana deterministycznie w oparciu o pomysł algebraiczny lub geometryczny (tak jak np. macierze w przykładzie Klee-Minty).

## Forma rozwiązania i termin

Rozwiązanie powinno zawierać raport zawierający opis problemu, szczegółowy opis przyjętego sposobu rozwiązania, arkusz zawierający wykonane obliczenia oraz uzyskane wyniki.

Rozwiązania należy wysyłać e-mailem do prowadzącego ćwiczenia (w grupie piątkowej z kopią do wykładowcy) do 10 maja do godz. 23:59. W przypadku problemu z dotrzymaniem terminu proszę o **wcześniejszy** kontakt z wykładowcą (zanim termin minie!).

Zaliczenie podczas ustnego egzaminu w trakcie sesji. Na egzamin trzeba przygotować prezentację (maksymalnie 10 minut, należy przygotować slajdy) na temat uzyskanych wyników i wykorzystanych metod.