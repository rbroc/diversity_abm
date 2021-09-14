import pandas as pd 
import numpy as np 
from .utils import load_matrix
from .matrix import Matrix
from copy import deepcopy

class Agent:
    ''' Initializes an agent
    Args:
        name (str): specifies an ID for the agent
        matrix_filename (str): path to agent's model.
        path (str or Path): folder to agent's model in current wd.
        matrix_kwargs: named arguments for Matrix initialization
    '''

    def __init__(self, agent_name, matrix_filename, 
                path=None,
                **matrix_kwargs):
        self.name = agent_name or matrix_filename
        self.matrix = Matrix(filename=matrix_filename, path=path,
                             **matrix_kwargs)
        self.matrix_backup = deepcopy(self.matrix)

    @property
    def model(self):
        return self.matrix.data
    
    def speak(self, seed, pop=True):
        ''' Picks response word based on cue (seed).
            Returns probability of cue-response association, and response word.
            Also pops response/cue value if pop=True
        '''
        resp_idx = np.argmin(self.matrix.data[seed])
        resp_wd = self.matrix.data[seed].index[resp_idx]
        prob = self.listen(seed, resp_wd, pop)
        return prob, resp_wd

    def listen(self, seed, resp_wd, pop=True):
        ''' Listens to response (resp_wd), return probability of response
            given the cue in the agent's space and pops response/cue value
            from agent's memory is pop is true
        '''
        prob = self._return_prob(seed, resp_wd)
        if pop:
            self._pop_words(resp_wd)
        return prob

    def _return_prob(self, seed, resp_wd, pop=True):
        ''' Returns association score for seed-resp_wd pair '''
        return self.matrix.data[seed][resp_wd]

    def _pop_words(self, resp_wd):
        ''' Pop response word for possible options '''
        self.matrix.data.loc[resp_wd] = np.nan