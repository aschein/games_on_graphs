"""
Game interfaces for simulating evolutionary games on graphs.
"""
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
        rand_dim = rn.choice(tmp.size)
        if (tmp[rand_dim] == 0):
            tmp[rand_dim] = 1
        elif (tmp[rand_dim] == shp[rand_dim] - 1):
            tmp[rand_dim] -= 1
        else:
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

    def __init__(self, shp, n_strategies=3, alpha=0.):
        self.shp = shp
        self.n_agents = np.prod(shp)
        self.n_strategies = n_strategies
        self._state = np.zeros(shp + (n_strategies,))

    def _preprocess(self, payoffs):
        assert isinstance(payoffs, np.ndarray)
        assert payoffs.shape == (self.n_strategies, self.n_strategies)
        # if not (payoffs >= 0).all():
        #     print 'Warning: Negative payoffs, proceeding with gain matrix.'
        #     payoffs = (payoffs - payoffs.min()) + 1
        #     print payoffs
        # assert (payoffs >= 0).all()
        return payoffs

    def _initalize(self, version):
        if version == 'uniform':
            self._state[:] = 1. / float(n_strategies)

        elif version == 'random mixed':
            self._state[:] = rn.dirichlet(np.ones(self.n_strategies), size=self.shp)

        elif version == 'random pure':
            probs = np.ones(self.n_strategies) / float(self.n_strategies)
            self._state[:] = rn.multinomial(1, probs, size=self.shp)

        else:
            raise TypeError

        marg = self.get_marginal()
        assert np.allclose(marg.sum(), 1)

    def _get_matchup(self):
        agent1 = rn.choice(self.n_agents)
        agent1_coords = np.unravel_index(agent1, self.shp)
        agent2_coords = get_random_neighbor(agent1_coords, self.shp)
        return agent1_coords, agent2_coords

    def get_state(self):
        return self._state.copy()

    def get_marginal(self):
        return self._state.reshape((-1, self.n_strategies)).mean(axis=0)

    def play(self, payoffs, init=None, n_iter=1000):
        if init is not None:
            self._initalize(version=init)

        payoffs = self._preprocess(payoffs)

        for itn in xrange(n_iter):
            agent1_coords, agent2_coords = self._get_matchup()
            agent1_strat = self._state[agent1_coords]
            agent2_strat = self._state[agent2_coords]

            agent1_payoff = np.dot(np.dot(agent1_strat, payoffs), agent2_strat)
            agent2_payoff = np.dot(np.dot(agent1_strat, payoffs.T), agent2_strat)

            net_payoff = agent1_payoff + agent2_payoff

            if net_payoff == 0:  # zero-sum game
                agent1_wins = agent1_payoff > agent2_payoff
                new_strat = agent1_strat if agent1_wins else agent2_strat

            else:
                rel_payoffs = np.zeros(2)
                rel_payoffs[0] = agent1_payoff / net_payoff
                rel_payoffs[1] = agent2_payoff / net_payoff

                new_strat = agent1_strat * (agent1_payoff / net_payoff) + \
                            agent2_strat * (agent2_payoff / net_payoff)

            self._state[agent1_coords] = new_strat
            self._state[agent2_coords] = new_strat


if __name__ == '__main__':
    shp = (100, 100)
    n_strategies = 3
    alpha = 0

    payoffs = np.array([[  0.,  1.,  1.],
                        [ -1.,  0.,  0.],
                        [ -1.,  0.,  0.]])  # rock-paper-scissors

    game = TwoPlayerGame(shp=shp, n_strategies=n_strategies, alpha=alpha)
    game.play(payoffs, init='random pure')
