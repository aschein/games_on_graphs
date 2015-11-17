import sys
import pylab as pl
from collections import defaultdict
from contextlib import contextmanager

import numpy as np
from game import TwoPlayerGame

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


def simulate_2D(payoffs, shp=(200, 200), n_strategies=3, init='random pure', n_iter=1000):
    assert len(shp) == 2
    game = TwoPlayerGame(shp=shp, n_strategies=n_strategies)
    game.play(payoffs, init=init, n_iter=1)

    while True:
        try:
            game.play(payoffs, init=None, n_iter=n_iter)
            state = game.get_state()
            with plt('State') as ax:
                ax.imshow(state)
                ax.set_title('State')
            marg = game.get_marginal()

        except KeyboardInterrupt:
            if raw_input("\nSimulation paused.\
                          \nEnter 't' to terminate.\
                          \nEnter anything else to continue.") == 't':
                sys.exit()
            pass

    raw_input("\nSimulation over.\nPress anything to terminate.")



if __name__ == '__main__':
    shp = (600, 600)
    n_strategies = 3

    payoffs = np.array([[  0.,  19.,  -1.],
                        [ -20.,  0.,   1.],
                        [  1.,  -1.,  0.]])  # rock-paper-scissors

    simulate_2D(payoffs, init='random pure')
