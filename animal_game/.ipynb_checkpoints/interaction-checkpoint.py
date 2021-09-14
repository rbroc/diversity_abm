import numpy as np
import scipy
import pandas as pd 
import random
from .agents import Agent
import itertools
from .utils import generate_turn_idx
from pathlib import Path
from datetime import datetime
import logging

class Interaction:

    ''' Interaction class
    Args:
        agents (Agent or list): single Agent or list taking part in the 
            interaction. If not defined, nr_agents and matrix_filenames 
            must be defined, and agents will be created within Interaction
            init call.
        threshold (float): Lowest possible association threshold. If no
            value above it is found, the game will stop.
        nr_sim (int): How many interactions to run
        max_exchanges (int): Max number of exchanges in the game for early 
            stopping (optional)
        log_id (str): filename for logfile
        save_folder (str): relative path for logfile
        nr_agents (int): if agents is not defined, this parameter must be set.
            Indicates how many agents must be initialized
        matrix_filenames (str or list): path to matrix filename from which agents 
            will be initialized. If a list, different files can be passed, 
            and must be of length nr_agents.
        agent_kwargs: named arguments for Agent initialization
    '''

    def __init__(self, agents=None, 
                 threshold=0.006,
                 nr_sim=1, max_exchanges=None,
                 log_id=None, save_folder=None, 
                 nr_agents=None, matrix_filenames=None,
                 **agent_kwargs):
        
        self.nr_agents = nr_agents
        if agents is None:
            agents = []
            if nr_agents is None:
                raise ValueError("Please pass Agents or specify number of "
                                 "agents to be initialized via nr_agents")
            matrix_filenames = self._check_agents_parameters(matrix_filenames, 
                                                             'matrix_filenames')
            for i in range(nr_agents):
                agent_name = 'agent' + str(i + 1)
                agent = Agent(agent_name=agent_name,
                              matrix_filename=matrix_filenames[i],
                              **agent_kwargs)
                agents.append(agent)

        self.agents = [agents] if isinstance(agents, Agent) else agents
        for a in self.agents:
            if not isinstance(a, Agent):
                raise ValueError('agents must be a list of Agent types')
        self.agent_names = [a.name for a in self.agents]
        self.threshold = threshold
        self.nr_sim = nr_sim
        self.max_exchanges = max_exchanges or self.agents[0].matrix.data.shape[0]
        self.log_id = log_id or 'log_' + datetime.now().strftime('%Y%m%d%H%M%S')
        self.save_folder = save_folder

    def _check_agents_parameters(self, par, parname):
        if isinstance(par, list):
            if len(par) != self.nr_agents:
                raise ValueError(f"Length of {parname} should "
                                   "match value of nr_agents")
        else:
            par = [par] * self.nr_agents
        return par

        
    def _run_single_trial(self, speaker, seed, turn, itr, init_seed, log=None):
        ''' Run a single trial (one agent) 
        Args:
            speaker (Agent): agent performing speaking act
            seed (str): Cue word
            turn (int): turn number
            log (df): dataframe containing interaction log'''
        prob_agent, resp = speaker.speak(seed=seed)
        prob = [a.listen(seed, resp) if a is not speaker else prob_agent
                for a in self.agents]
        # ADD HERE
        ndens = [np.sum(a.matrix_backup.data[seed].values < self.threshold)
                 for a in self.agents]
        # ADD HERE
        ndens_current = [np.sum(a.matrix.data[seed].values < self.threshold) + 1
                         for a in self.agents]
        log = self._append_data(log, speaker, turn, itr, seed, init_seed,
                                resp, prob, ndens, ndens_current)
        return log, resp

    def _append_data(self, log, agent, turn, itr, seed, init_seed, resp, 
                     prob, ndens, ndens_current):
        ''' Append all trial data to the log dataframe'''
        turn_data = [agent.name, turn, itr, seed, resp, *prob, *ndens, *ndens_current]
        int_data = [self.threshold, self.nr_sim, 
                    self.max_exchanges, init_seed, self.log_id,
                    len(self.agents)]
        metadata = pd.Series(turn_data + int_data)
        metadata.index = log.columns
        log = log.append(pd.Series(metadata), ignore_index=True)
        return log

    def _create_outpath(self, sep='\t'):
        ''' Create path for whole interaction '''
        fname = '_'.join([self.log_id, 
                         str(len(self.agents)), 
                         str(self.threshold)]) + '.txt'
        if self.save_folder:
            as_path = Path(self.save_folder)
            as_path.mkdir(parents=True, exist_ok=True)
            fpath = as_path / fname
        else:
            fpath = Path('logs') / fname
        return fpath

    def run_interaction(self, seeds=None):
        ''' Run a full interaction between agents 
            Args:
                seeds (str or list): name(s) of initial seeds
        '''
        if seeds:
            if isinstance(seeds, list):
                if len(seeds) != self.nr_sim:
                    raise ValueError(f"Length of init_seed should "
                                    "match value of nr_sim")
            else:
                seeds = [seeds] * self.nr_sim
        else:
            seeds = np.random.choice(a=self.agents[0].matrix.data.index, size=self.nr_sim)
        nr_turns = self.max_exchanges
        turn_idx = generate_turn_idx(nr_turns, self.agents)
        fpath = self._create_outpath()
        for itr in range(self.nr_sim):
            log = pd.DataFrame(columns=['agent', 'turn', 'iter', 'seed', 'response',
                                        *['prob' + str(i) 
                                          for i,a in enumerate(self.agents)],
                                        *['ndens' + str(i) 
                                          for i,a in enumerate(self.agents)],
                                        *['ndens_current' + str(i) 
                                          for i,a in enumerate(self.agents)],
                                        'threshold', 'nr_sim', 
                                        'max_exchanges', 'init_seed',
                                        'log_id', 'nr_agents'])
            init_seed = seeds[itr]
            for agent in self.agents:
                agent._pop_words(init_seed)
            for idx in turn_idx:
                turn, agent = idx
                if turn == 0:
                    seed = init_seed
                if (agent.matrix.data[seed] < self.threshold).any():
                    log, seed = self._run_single_trial(agent, seed, 
                                                       turn, itr, 
                                                       init_seed, log)
                else:
                    break
            if itr == 0:
                log.to_csv(fpath, index=False)
            else:
                log.to_csv(fpath, mode='a', index=False, header=False)
            for agent in self.agents:
                agent.matrix.data = agent.matrix_backup.data.copy()
        print(f'{self.log_id} done!')
        return log
