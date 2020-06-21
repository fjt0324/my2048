import torch
import numpy as np
import torchvision.transforms as transforms
from myNet import RNN
from time import sleep
import sys
import csv, os
PATH = 'model_final.pkl'

class Agent:
    '''Agent Base.'''

    def __init__(self, game, display=None):
        self.game = game
        self.display = display

    def play(self, max_iter=np.inf, verbose=False):
        n_iter = 0
        while (n_iter < max_iter) and (not self.game.end):
            direction = self.step()
            self.game.move(direction)
            n_iter += 1
            if verbose:
                print("Iter: {}".format(n_iter))
                print("======Direction: {}======".format(
                    ["left", "down", "right", "up"][direction]))
                if self.display is not None:
                    self.display.display(self.game)

    def step(self):
        direction = int(input("0: left, 1: down, 2: right, 3: up = ")) % 4
        return direction

class MyAgent(Agent):
    
    def __init__(self, game, display=None):
        if game.size != 4:
            raise ValueError(
                "`%s` can only work with game of `size` 4." % self.__class__.__name__)
        super().__init__(game, display)
        self.model = torch.load(PATH,map_location=lambda storage, loc: storage.cuda(0))
        self.model.cuda()
        self.model.eval()

    def step(self):
        board = np.where(self.game.board == 0, 1, self.game.board)
        board = np.log2(board)
        board1=board.T
        board2=np.vstack((board,board1))
        board = board2[:, :, np.newaxis]


        board = board/ 11.0

        trans = transforms.Compose([transforms.ToTensor()])
        board = trans(board)
        board = board.type(torch.float)
        device1 = torch.device("cuda")
        board=board.to(device1)
        out = self.model(board)


        direction = torch.max(out, 1)[1]
        return int(direction)




