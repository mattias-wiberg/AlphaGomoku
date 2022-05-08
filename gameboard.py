import tensorflow as tf
class GameBoard:
    def __init__(self,N_row,N_col):
        self.N_row=N_row
        self.N_col=N_col
        # Create table for game board
        # -1   black
        #  0   empty
        #  1   white
        self.board = tf.zeros((1,N_row,N_col),dtype=tf.float64)
        self.gameOver = False
        self.piece = -1

    def restart(self):
        self.board = tf.zeros((1,self.N_row,self.N_col),dtype=tf.float64)
        self.gameOver = False
        self.piece = -1

    def move(self, row, col):
        self.board[0,row,col] = self.piece
        reward = self.get_reward()
        if reward != 0:
            self.restart()
        else:
            self.piece *= -1 # Alternate between white and black
        return reward

    # -1: loss
    # +1: win
    # 0: game not ended
    def get_reward(self):
        # TODO: Alexey is working here
        pass

        



            
            



