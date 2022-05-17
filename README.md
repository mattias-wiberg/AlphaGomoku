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

## Observations
- Not using convolution on the input seems to lead to much longer training times (not surprising since the weights are not shared anymore)
    -> Use CNN on the input!
- Removing pooling slowns down training due to larger dimensionality, but it seems to learn about as well in the end. Don't think we should have pooling; it's not a picture after all.
    -> Don't use pooling!
- Only having 1 filter also seems to learn about as well but the dimensionality is much smaller so it is thus faster.
    -> Don't have too many filters (1 might be enough)!
- If appending the board to the output of the convolution it seems that it won't work unless the board has first been passed through a dense layer.
    -> Pass board through a dense layer before appending with convolutional output.
- Seems to learn faster with tanh instead of relu. Perhaps not so weird since we should probably allow negative values since it is not a picture.
    -> Use activation functions that also have a negative part?
- Since the q_table is limited to [-1,1] it might be a very good idea to use tanh on the last layer instead of linear activation.
    -> Use tanh on the last layer.

## To Try
- two different agents or NN architectures?/Use two networks, play against older versions to see if it gets better?
	-> different action depending on piece (qnhat for one of the pieces)
    -> have 1 (white) play as qnhat (unimproved) and then you can log rewards as well?
    -> having it play against the same network might be causing some form of numerical instability/weird behaviour?
- vary hyperparams (loss,optimizer,layers,neurons,activation functions)
- read papers
- give it an AI to play against: one just tries to get 5 in a row?
- real replay buffer as well?

