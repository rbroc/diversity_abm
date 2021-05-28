import numpy as np
import glob
import random
import itertools
import random
import pandas as pd
from animal_game.agents import Agent
from animal_game.utils import compute_thresholds
from animal_game.interaction import Interaction
from multiprocessing import Pool

# Define date
date = '21_05_27'

# Define key vars
models = ['animal_game/models/wiki_euclidean_distance.tsv']
animals = pd.read_csv('animal_game/models/animal_list.csv')
thresholds = compute_thresholds(models, 
                                q=[round(n,2) for n in np.arange(0.05, 1.0, 0.05)], 
                                round_at=5)

# Run params
nr_sim = len(animals['Animals'].tolist())
outpath = f'animal_game/logs/{date}/pairs'

# Create pairs
pair_df = pd.read_csv(f'animal_game/models/{date}/sampled_pairs.tsv', sep='\t')
fnames_1 = [f'animal_game/models/{date}/noised_distance_matrices/' + f 
            for f in pair_df.fname_1.tolist()]
fnames_2 = [f'animal_game/models/{date}/noised_distance_matrices/' + f 
            for f in pair_df.fname_1.tolist()]
afiles_list = list(zip(fnames_1, fnames_2))
anames_list = [(af[0].split('/')[-1].strip('.tsv'), 
                af[1].split('/')[-1].strip('.tsv')) for af in afiles_list]

# Create agents
agents = []
for i in range(len(afiles_list)):
    a0 = Agent(agent_name=anames_list[i][0], 
               matrix_filename=afiles_list[i][0])
    a1 = Agent(agent_name=anames_list[i][1], 
               matrix_filename=afiles_list[i][1])
    agents.append((a0,a1))
print(f'{len(agents)} pairs created')

# Main loop
def run_pair(a, opath):
    print(f'Agent Names: {a[0].name}, {a[1].name}')
    log_id = f'{a[0].name}_{a[1].name}'
    i = Interaction(agents=a,
                    threshold=thresholds[0.15],
                    save_folder=opath,
                    log_id=log_id,
                    nr_sim=nr_sim)
    i.run_interaction(seeds=animals['Animals'].tolist())

# Run
if __name__=='__main__':
    pool = Pool(processes=20)
    pool.starmap(run_pair, zip(agents,
                               [outpath] * len(agents)))
