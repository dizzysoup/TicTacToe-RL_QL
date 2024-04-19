from Game import TicTacToe
from QLearning import  Qlearning
import numpy as np
import matplotlib.pyplot as plt


game = TicTacToe(True) #game instance, True means training
total_episodes = 100000
# epsilon、alpha 、 gamma, stop_training, total_episodes
player1= Qlearning(0.2,0.3,0.9,False, total_episodes) 
player2 =Qlearning(0.2,0.3,0.9,True, total_episodes) 

game.startTraining(player1,player2) #start training

episodes = np.arange(1, total_episodes + 1)
game.train(total_episodes) 
game.saveStates() 

#game.train(10000) #train for 200,000 iterations

plt.plot(episodes, player1.cumulative_wins , label='Player 1 Wins')
plt.plot(episodes, player2.cumulative_wins, label='Player 2 Wins')
plt.xlabel('Episodes')
plt.ylabel('Cumulative Wins')
plt.title('Cumulative Wins vs. Episodes')
plt.legend()
plt.show()