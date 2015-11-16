"""
Game interfaces for simulating evolutionary games on graphs.
"""
import sys
import numpy as np
import numpy.random as rn


def is_valid_coords(coords, shp):  # hacky
    try:
        np.ravel_multi_index(coords, shp)
        return True
    except ValueError:
        return False


def get_random_neighbor(coords, shp):
    tmp = np.asarray(coords)
    valid = False
    while (tuple(tmp) == coords) or (not valid):
        rand_dim = rn.choice(tmp.ndim)
        tmp[rand_dim] += rn.choice([-1, 1])
        valid = is_valid_coords(tmp, shp)
    return tuple(tmp)


class TwoPlayerGame:
    """ The basic 2-player Game interface to inherit from.

    Parameters
    ----------
    shp : tuple, (N1, ..., Nd)
        Game is played on a d-dimensional lattice.
        shp provides the dimensions: d = len(shp)
        lattice is N1 x ... x Nd.
    n_strategies : int, optional
        Number of possible strategies.
        Default is 3, which is best for visualization.

    Attributes
    ----------
    state : numpy array, shape = (N1, ... , Nd, n_strategies)
            Every agent (i1, ... iD) on the lattice has a mixed strategy.
    """

    def __init__(self, shp, n_strategies=3):
        self.shp = shp
        self.n_agents = np.prod(shp)
        self.n_strategies = n_strategies
        self.state = np.zeros(shp + (n_strategies,))

    def _preprocess(self, payoffs):
        assert isinstance(payoffs, np.ndarray)
        assert payoffs.shape == (self.n_strategies, self.n_strategies)
        if not (payoffs >= 0).all():
            print 'Warning: Negative payoffs, proceeding with gain matrix.'
            payoffs = payoffs - payoffs.min(axis=1).T
        assert (payoffs >= 0).all()
        return payoffs

    def _get_matchup(self):
        agent1 = rn.choice(self.n_agents)
        agent1_coords = np.unravel_index(agent1, self.shp)
        agent2_coords = get_random_neighbor(agent1_coords, self.shp)
        return agent1_coords, agent2_coords

    def _update_plot(self):
        raise NotImplementedError

    def play(self, payoffs, n_iter=10000, plot_every=100):
        payoffs = self._preprocess(payoffs)

        for itn in xrange(n_iter):
            try:
                agent1_coords, agent2_coords = self._get_matchup()
                agent1_strat = self.state[agent1_coords]
                agent2_strat = self.state[agent2_coords]

                agent1_payoff = np.dot(np.dot(agent1_strat, payoffs), agent2_strat)
                agent2_payoff = np.dot(np.dot(agent1_strat, payoffs.T), agent2_strat)

                net_payoff = agent1_payoff + agent2_payoff

                new_strat = agent1_strat * (agent1_payoff / net_payoff) + \
                            agent2_strat * (agent2_payoff / net_payoff)

                self.state[agent1_coords] = new_strat
                self.state[agent2_coords] = new_strat

                if itn % plot_every == 0:
                    self._update_plot()

            except KeyboardInterrupt:
                if raw_input("\nSimulation paused.\
                              \nEnter 't' to terminate.\
                              \nEnter anything else to continue.") == 't':
                    sys.exit()
                pass

        raw_input("\nSimulation over.\nPress anything to terminate.")















