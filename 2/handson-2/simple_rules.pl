:- dynamic father/2.

is_father_of_two(Y) :- father(Y, X1), father(Y, X2), X1 \= X2 .

brothers(X, Y) :- father(F, X), father(F, Y), X \= Y .

father(giuseppe,mario).
father(giuseppe,luigi).
