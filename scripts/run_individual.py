import glob
from animal_game.agents import Agent
import itertools
import pandas as pd
import numpy as np
from animal_game.utils import compute_thresholds
from animal_game.interaction import Interaction
from multiprocessing import Pool

# Date
date = '21_08_20'

# Load models
models = ['../animal_game/models/wiki_euclidean_distance.tsv']
animals = pd.read_csv('../animal_game/models/animal_list.csv')
thresholds = compute_thresholds(models, 
                                q=[round(n,2) for n in np.arange(0.05, 1.0, 0.05)], 
                                round_at=5)

# Load matrices and create agents
matrices = glob.glob(f'../animal_game/models/{date}/noised_distance_matrices/*')
agents = []
for m in matrices:
    agent = Agent(agent_name=m.split('/')[-1][:-4], 
                  matrix_filename=m)
    agents.append(agent)
    
# Interaction parameters
nr_sim = len(animals['Animals'].tolist())
outpath = f'../animal_game/logs/{date}/individual'

# Main function
def run_individual(agent, outpath):

    print(f'Agent Name: {agent.name}')
    log_id = f'{agent.name}'
    i = Interaction(agents=agent,
                    threshold=thresholds[0.15],
                    save_folder=outpath,
                    log_id=log_id,
                    nr_sim=nr_sim)
    i.run_interaction(seeds=animals['Animals'].tolist())


if __name__=='__main__':
    pool = Pool(processes=22)
    pool.starmap(run_individual, zip(agents,
                                     [outpath] * len(agents)))
