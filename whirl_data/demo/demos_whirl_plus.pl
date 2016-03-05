:- use_module('demos_whirl_plus.py').

:- consult('demosdata.pl').

Prob::q1(Name1, Name2, URL1, URL2) :-
    demox(URL2, Name2), 
    similarity(Name2, Name1_Prob),
    Name1_Prob = [Name1, URL1, Prob], 
    Prob > 0.5 .
query(q1(N1, N2, U1, U2)).


