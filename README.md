# AlphaGomoku
A deep q network implementation to find a strategy for the game traditional Gomoku.
## Description
Gomoku, also called Five in a Row, is an abstract strategy board game. It is traditionally played with Go pieces (black and white stones) on a Go board. It is played using a 15×15 board while in the past a 19×19 board was standard. Because pieces are typically not moved or removed from the board, gomoku may also be played as a paper-and-pencil game. The game is known in several countries under different names.

Players alternate turns placing a stone of their color on an empty intersection. **Black plays first.** The winner is the first player to form an unbroken chain of five stones horizontally, vertically, or diagonally. Placing so that a line of more than five stones of the same color is created does not result in a win.

## Scope
We will limit ourself to the following properties from the original game:
 - 15x15 board
 - No opening rule

## Game implementation
The board is defined as a matrix using:<br>
-1   black<br>
 0    empty<br>
 1    white<br>

## Network
 - In: Board
 - CNN
 - FCN
 - Out: Q table

 ### Selecting an action
 Masking the q table output of the network with the possible moves and then using the epsilon greedy policy.
 