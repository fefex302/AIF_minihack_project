:- dynamic position/4.
:- dynamic own/3.

action(wield(Weapon)) :- own(agent, weapon, Weapon).

action(pick_up) :- stepping_on(agent, ObjClass, _), is_pickable(ObjClass).

action(go_to_weapon(Direction)) :-  position(agent, _, Agent_Row, Agent_Col), position(weapon, _, Weapon_Row, Weapon_Col),
                                    next_step(Agent_Row, Agent_Col, Weapon_Row, Weapon_Col, Direction).


% compute the next step toward the goal
next_step(R1,C1,R2,C2, Direction) :-
    ( R1 == R2 -> ( C1 > C2 -> Direction = west; Direction = east );
    ( C1 == C2 -> ( R1 > R2 -> Direction = north; Direction = south);
    ( R1 > R2 ->
        ( C1 > C2 -> Direction = northwest; Direction = northeast );
        ( C1 > C2 -> Direction = southwest; Direction = southeast )
    ))).


%%%% known facts %%%%
opposite(north, south).
opposite(south, north).
opposite(east, west).
opposite(west, east).
opposite(northeast, southwest).
opposite(southwest, northeast).
opposite(northwest, southeast).
opposite(southeast, northwest).

resulting_position(Old_Row, Old_Col, New_Row, New_Col, north) :-
    New_Row is Old_Row-1, New_Col = Old_Col.
resulting_position(Old_Row, Old_Col, New_Row, New_Col, south) :-
    New_Row is Old_Row+1, New_Col = Old_Col.
resulting_position(Old_Row, Old_Col, New_Row, New_Col, west) :-
    New_Row = Old_Row, New_Col is Old_Col-1.
resulting_position(Old_Row, Old_Col, New_Row, New_Col, east) :-
    New_Row = Old_Row, New_Col is Old_Col+1.
resulting_position(Old_Row, Old_Col, New_Row, New_Col, northeast) :-
    New_Row is Old_Row-1, New_Col is Old_Col+1.
resulting_position(Old_Row, Old_Col, New_Row, New_Col, northwest) :-
    New_Row is Old_Row-1, New_Col is Old_Col-1.
resulting_position(Old_Row, Old_Col, New_Row, New_Col, southeast) :-
    New_Row is Old_Row+1, New_Col is Old_Col+1.
resulting_position(Old_Row, Old_Col, New_Row, New_Col, southwest) :-
    New_Row is Old_Row+1, New_Col is Old_Col-1.

close_direction(north, northeast).
close_direction(northeast, east).
close_direction(east, southeast).
close_direction(southeast, south).
close_direction(south, southwest).
close_direction(southwest, west).
close_direction(west, northwest).
close_direction(northwest, north).

own(agent, _, _) :- fail.


unsafe_position(_,_) :- fail.
safe_position(R,C) :- \+ unsafe_position(R,C).

is_pickable(weapon).
% is_pickable(armor).