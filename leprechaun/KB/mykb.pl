:- dynamic position/4.
:- dynamic wields_weapon/2.
% :- dynamic wears_armor/2.
:- dynamic own/3.
:- dynamic stepping_on/3.

%% care about the order %%

action(wield(Weapon)) :- own(agent, weapon, Weapon).
% action(wear((Armor))) :- own(agent, armor, Armor). 

action(pick_up) :- stepping_on(agent, ObjClass, _), is_pickable(ObjClass).

action(go_to_weapon(Direction)) :-  position(agent, _, Agent_Row, Agent_Col), position(weapon, _, Weapon_Row, Weapon_Col),
                                    next_step(Agent_Row, Agent_Col, Weapon_Row, Weapon_Col, Direction).
                                    % resulting_position(Agent_Row, Agent_Col, New_Agent_Row, New_Agent_Col, Direction).

% action(go_to_armor(Direction)) :-  position(agent, _, Agent_Row, Agent_Col), position(armor, _, Armor_Row, Armor_Col),
%                                    next_step(Agent_Row, Agent_Col, Armor_Row, Armor_Col, Direction).

action(go_to_stairs(Direction)) :-  position(agent, _, Agent_Row, Agent_Col), position(stairs, _, Stairs_Row, Stairs_Col),
                                    wields_weapon(agent, Weapon), 
                                    next_step(Agent_Row, Agent_Col, Stairs_Row, Stairs_Col, Direction).
                                    % resulting_position(Agent_Row, Agent_Col, New_Agent_Row, New_Agent_Col, Direction).

% compute the next step toward the goal
next_step(R1,C1,R2,C2, Direction) :-
    ( R1 == R2 -> ( C1 > C2 -> Direction = west; Direction = east );
    ( C1 == C2 -> ( R1 > R2 -> Direction = north; Direction = south);
    ( R1 > R2 ->
        ( C1 > C2 -> Direction = northwest; Direction = northeast );
        ( C1 > C2 -> Direction = southwest; Direction = southeast )
    ))).

% check if the selected direction is safe
% safe_direction(R, C, D, Direction) :- resulting_position(R, C, NewR, NewC, D),
%                                       ( safe_position(NewR, NewC) -> Direction = D;
%                                       % else, get a new close direction
%                                       % and check its safety
%                                       close_direction(D, ND), safe_direction(R, C, ND, Direction)
%                                       ).


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

is_pickable(weapon).
% is_pickable(armor).