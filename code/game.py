"""
Game interfaces for simulating evolutionary games on graphs.
"""
import numpy as np
import numpy.random as rn

from utils import get_random_neighbor


class Game:
    """ The basic Game interface to inherit from.

    Inheriting sub-classes must implement the methods:

            1) _get_matchup: Randomly match 2+ players
            2) _update: Update the game state based on payoffs and randomly matched players

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
        self._state = np.zeros(shp + (n_strategies,))

    def _get_matchup(self):
        raise NotImplementedError

    def _update(self, payoffs, *players):
        raise NotImplementedError

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
            self._state[:] = 1. / float(self.n_strategies)

        elif version == 'random mixed':
            self._state[:] = rn.dirichlet(np.ones(self.n_strategies), size=self.shp)

        elif version == 'random pure':
            probs = np.ones(self.n_strategies) / float(self.n_strategies)
            self._state[:] = rn.multinomial(1, probs, size=self.shp)

        else:
            raise TypeError

        marg = self.get_marginal()
        assert np.allclose(marg.sum(), 1)

    def get_state(self):
        return self._state.copy()

    def get_marginal(self):
        return self._state.reshape((-1, self.n_strategies)).mean(axis=0)

    def play(self, payoffs, init=None, n_iter=1000):
        payoffs = self._preprocess(payoffs)
        if isinstance(init, str):
            self._initalize(version=init)
        elif isinstance(init, np.ndarray):
            assert init.shape == self._state.shape
            self._state[:] = init

        for itn in xrange(n_iter):
            players = self._get_matchup()
            self._update(payoffs, *players)


class TwoPlayerImitationGame(Game):
    """ A basic TwoPlayerGame interface.

    _get_matchup : Random neighbor matching.
    _update  : Loser adopts strategy of winner with some probability.

    Basic diffusion process.
    """

    def __init__(self, shp, n_strategies=3, version='Boltzman'):
        self.version = version
        Game.__init__(self, shp=shp, n_strategies=n_strategies)

    def _get_matchup(self):
        agent1 = rn.choice(self.n_agents)
        agent1_coords = np.unravel_index(agent1, self.shp)
        agent2_coords = get_random_neighbor(agent1_coords, self.shp)
        return agent1_coords, agent2_coords

    def _update(self, payoffs, *players):
        p1_coords, p2_coords = players
        p1_strat = self._state[p1_coords]
        p2_strat = self._state[p2_coords]

        p1_payoff = np.dot(np.dot(p1_strat, payoffs), p2_strat)
        p2_payoff = np.dot(np.dot(p1_strat, payoffs.T), p2_strat)

        p1_wins = p1_payoff > p2_payoff
        losing_player = p2_coords if p1_wins else p1_coords
        winning_strat = p1_strat if p1_wins else p2_strat

        if self.version is None:
            # Loser always imitates
            self._state[losing_player] = winning_strat

        elif self.version == 'Boltzman':
            # Probability of imitation is Boltzman form
            Z = np.exp(p1_payoff) + np.exp(p2_payoff)
            winning_payoff = p1_payoff if p1_wins else p2_payoff
            if (np.exp(winning_payoff) / Z) > rn.random():
                self._state[losing_player] = winning_strat
        else:
            raise ValueError


class TwoPlayerBestResponseGame(Game):
    """ A basic TwoPlayerGame interface.

    _get_matchup : Random neighbor matching.
    _update  :  Players choose a pure strategy at random from their distribution.
                Both players adopt a weighted average of their strategies.
    """

    def __init__(self, shp, n_strategies=3, version='Boltzman'):
        self.version = version
        Game.__init__(self, shp=shp, n_strategies=n_strategies)

    def _get_matchup(self):
        agent1 = rn.choice(self.n_agents)
        agent1_coords = np.unravel_index(agent1, self.shp)
        agent2_coords = get_random_neighbor(agent1_coords, self.shp)
        return agent1_coords, agent2_coords

    def _update(self, payoffs, *players):
        p1_coords, p2_coords = players
        p1_strat = self._state[p1_coords]
        p2_strat = self._state[p2_coords]

        p1_payoff = np.dot(np.dot(p1_strat, payoffs), p2_strat)
        p2_payoff = np.dot(np.dot(p1_strat, payoffs.T), p2_strat)
        p1_wins = p1_payoff > p2_payoff

        winning_strat = p1_strat if p1_wins else p2_strat
        winning_payoff = p1_payoff if p1_wins else p2_payoff

        losing_strat = p2_strat if p1_wins else p1_strat
        losing_payoff = p2_payoff if p1_wins else p1_payoff

        Z = np.exp(winning_payoff) + np.exp(losing_payoff)
        new_strat = winning_strat * (np.exp(winning_payoff) / Z) + \
                    losing_strat * (np.exp(losing_payoff) / Z)
        assert (new_strat >= 0).all() and np.allclose(new_strat.sum(), 1.)

        self._state[p1_coords] = self._state[p2_coords] = new_strat
