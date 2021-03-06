# ABM of word association game 
This repository contains the code to run agent-based simulations of a word association game performed either individually or interactively by agents of varying diversity, whose semantic memories are noised versions of a skip-gram word2vec model. \n
The repository includes:
- ```animal_game```
    - Code for the mechanisms of the simulation;
- ```baseline.ipynb```
    - Notebook to run and analyse baseline performance (of the non-noised word2vec agent);
- ```noise_and_pair.ipynb```
    - Code to add varying levels of noise to the word2vec model and generate agents of different "diversity levels";
- ```analysis.ipynb``` and ```supplementary.ipynb```
    - analysis of the effect of interaction and diversity on different metrics of creativity in the association behavior produced by agents/pairs.

Results are currently being written up as a conference submission, and write-up of a more elaborate journal paper is planned.
Code requires some tidying up, coming soon :)
