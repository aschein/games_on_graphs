import sys
import pylab as pl
from collections import defaultdict
from contextlib import contextmanager

import numpy as np
import numpy.random as rn

from game import TwoPlayerImitationGame, TwoPlayerBestResponseGame


# global reference to all of the plots
def newax():
    return pl.figure().add_subplot(111)
AX = defaultdict(newax)


@contextmanager
def plt(name):
    pl.ion()
    ax = AX[name]
    ax.clear()
    ax.set_title(name)
    yield ax
    ax.figure.canvas.draw()
    pl.show()


def update_plots(game):
    state = game.get_state()
    with plt('State') as ax:
        ax.imshow(state)
        ax.set_title('State')
    # marg = game.get_marginal()


def simulate(game, payoffs, init, n_iter):
    game.play(payoffs, init=init, n_iter=1)
    update_plots(game)

    while True:
        try:
            game.play(payoffs, init=None, n_iter=n_iter)
            update_plots(game)
            print game.get_marginal()

        except KeyboardInterrupt:
            if raw_input("\nSimulation paused.\
                          \nEnter 't' to terminate.\
                          \nEnter anything else to continue.") == 't':
                sys.exit()
            pass

    raw_input("\nSimulation over.\nPress anything to terminate.")





def rock_paper_scissors(shp=(200, 200), init='random pure', n_iter=10000):
    game = TwoPlayerImitationGame(shp=shp, n_strategies=3, version='Boltzman')
    
    payoffs = np.array([[ 0, -1,  1],
                        [ 1,  0, -1],
                        [-1,  1,  0]])

    simulate(game=game,
             payoffs=payoffs,
             init=init,
             n_iter=n_iter)





def toxic_red(shp=(100, 100), init='random pure', n_iter=10000):
    game = TwoPlayerBestResponseGame(shp=shp, n_strategies=3, version='Boltzman')
    
    payoffs = np.array([[0,    0,   -1],
                        [0,    0,    0],
                        [1,    0,    0]])

    simulate(game=game,
             payoffs=payoffs,
             init=init,
             n_iter=n_iter)


def bad_recessive(shp=(50, 50), init='random pure', n_iter=10000):
    game = TwoPlayerBestResponseGame(shp=shp, n_strategies=3, version='Boltzman')
    
    payoffs = np.array([[ 0,    0,    0],
                        [  0,    0,    0],
                        [  0,    0,   10]])

    simulate(game=game,
             payoffs=payoffs,
             init=init,
             n_iter=n_iter)


def irony(n_iter=1000):
    payoffs = np.array([[ 0,   0,  -1],
                        [ 0,   0,   1],
                        [ 1,  -1,   0]])
    shp = (100, 100)
    init = np.ones(shp + (3,))
    init[:, :] = [0., 1., 0.]
    init[30:70, 30:70] = [1., 0., 0.]
    init[45:50, 45:50] = [0., 0., 1.]

    simulate(game_class=TwoPlayerImitationGame,
             payoffs=payoffs,
             shp=shp,
             n_strategies=3,
             init=init,
             n_iter=n_iter)


if __name__ == '__main__':
    # bad_recessive()
    # toxic_red()
    rock_paper_scissors()
    # irony()
