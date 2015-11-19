import sys
import pylab as pl
import numpy as np
import numpy.random as rn

from collections import defaultdict
from contextlib import contextmanager

import ternary

from game import TwoPlayerImitationGame, TwoPlayerBestResponseGame


# global reference to all of the plots
def newax():
    return pl.figure().add_subplot(111)
AX = defaultdict(newax)

marginals = []
fig, tax = ternary.figure(scale=1.0)
tax.boundary()

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

    marginals.append(tuple(game.get_marginal()))    
    pl.ion()
    tax.ax.clear()
    tax.boundary()
    tax.plot(marginals, linewidth=2, label="Curve")
    tax.ax.figure.canvas.draw()
    pl.show()

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


def rock_paper_scissors(shp=(50, 50), init='random pure', n_iter=1000):
    game = TwoPlayerImitationGame(shp=shp, n_strategies=3, version='Boltzman')

    payoffs = np.array([[ 0,  -1,  1],
                        [ 1,  0, -1],
                        [-1,  1,  0]])

    simulate(game=game,
             payoffs=payoffs,
             init=init,
             n_iter=n_iter)

def unbalanced_rps():
    shp = (200, 200)
    init = rn.multinomial(1, [0.8,0.1,0.1], size=shp)
    rock_paper_scissors(shp=shp, init=init, n_iter=10000)


def irony(n_iter=1000):
    payoffs = np.array([[ 0,   0,  -1],
                        [ 0,   0,   1],
                        [ 1,  -1,   0]])
    shp = (100, 100)
    init = np.ones(shp + (3,))
    init[:, :] = [0., 1., 0.]
    init[30:70, 30:70] = [1., 0., 0.]
    init[45:50, 45:50] = [0., 0., 1.]
    game = TwoPlayerImitationGame(shp=shp, n_strategies=3, version='Boltzman')
    simulate(game=game,
             payoffs=payoffs,
             init=init,
             n_iter=n_iter)


if __name__ == '__main__':
    # bad_recessive()
    # toxic_red()
    # shp = (200, 200)
    # init = rn.multinomial(1, [0.8,0.1,0.1], size=shp)
    # rock_paper_scissors(shp=shp, init=init, n_iter=10000)
    irony()
