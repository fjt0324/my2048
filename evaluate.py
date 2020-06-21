from game import Game
from displays import Display
from myNet import RNN



def single_run(size, score_to_win, AgentClass, **kwargs):
    game = Game(size, score_to_win)
    agent = AgentClass(game, display=Display(), **kwargs)
    agent.play(verbose=True)
    return game.score
def single_run_getdata(size, writeFile, LB, HB, AgentClass, **kwargs):
    game = Game(size, HB)
    agent = AgentClass(game, display=None, **kwargs)
    agent.writeBoard(writeFile, LB,verbose=True)
    return game.score

if __name__ == '__main__':
    GAME_SIZE = 4
    SCORE_TO_WIN = 2048
    N_TESTS = 50
    WRITEFILE = 'game2048/DATA.csv'
    LB = 256
    HB = 2048

    '''====================
    Use your own agent here.'''
    from agents import MyAgent as TestAgent
    '''===================='''

    scores = []
    for _ in range(N_TESTS):
        score = single_run(GAME_SIZE, SCORE_TO_WIN,
                           AgentClass=TestAgent)
        scores.append(score)
    print("Average scores: @%s times" % N_TESTS, sum(scores) / len(scores))
