import glob
import itertools
import pandas as pd
import numpy as np
from utils import (get_individual_aggs, 
                   get_pair_aggs,
                   concat_dfs, 
                   merge_pairs_inds, 
                   get_unique_named,
                   get_wd_originality_scores,
                   get_originality,
                   add_metrics,
                   get_pair_level_aggregates)
from multiprocessing import Pool
from sklearn.preprocessing import MinMaxScaler


DATE = '21_08_20'
MODELS_PATH = '../animal_game/models'
LOG_PATH = '../animal_game/logs'
ANALYSES_PATH = '../animal_game/analyses'

fs = glob.glob(f'{LOG_PATH}/{DATE}/individual/*')
pair_fs = glob.glob(f'{LOG_PATH}/{DATE}/pairs/*')
scaler = MinMaxScaler()


def _get_aggs(fn, flist):
    pool = Pool(20)
    results = pool.map(fn, flist)
    pool.close()
    aggs = concat_dfs(results)
    return aggs
 
    
def _merge_aggs(pdf, idf):
    if all([i in idf.columns for i in ['iter', 'threshold']]):
        idf.drop(['iter', 'threshold'], axis=1, inplace=True)
    if all([p in pdf.columns for p in ['iter', 'threshold']]):
        pdf.drop(['iter', 'threshold'], axis=1, inplace=True)
    pdf = merge_pairs_inds(pdf, idf, '0')
    pdf = merge_pairs_inds(pdf, idf, '1')
    return pdf


def _get_unique_named(flist, pdf):
    pool = Pool(20)
    results = pool.starmap(get_unique_named, zip(flist,
                                                 [LOG_PATH]*len(flist),
                                                 [DATE]*len(flist)))
    pool.close()
    unique_named = concat_dfs(results)
    pdf = pdf.merge(unique_named, 
                    on=['init_seed', 'pair'])
    pdf['collective_inhibition'] = pdf['unique_individual'] - \
                                   pdf['unique_pair']
    return pdf


def _get_originality(iflist, pflist, pdf):
    pool = Pool(20)
    results = pool.map(get_wd_originality_scores, iflist)
    pool.close()
    wd_orig = concat_dfs(results)
    wd_orig = wd_orig.groupby('word')['count'].sum().reset_index()
    wd_orig['originality_score'] = 1 / wd_orig['count']
    wd_orig['originality_score'] = scaler.fit_transform(wd_orig[['originality_score']])
    pool = Pool(20)
    results = pool.starmap(get_originality, zip(pflist,
                                                [wd_orig]*len(pflist),
                                                [LOG_PATH]*len(pflist),
                                                [DATE]*len(pflist)))
    pool.close()
    orig_df = concat_dfs(results)
    pdf = pdf.merge(orig_df, on=['init_seed', 'pair'])
    return pdf


def postprocess():
    
    # Get aggregates
    print('*** Computing aggregates ***')
    ia = _get_aggs(get_individual_aggs, fs)
    pa = _get_aggs(get_pair_aggs, pair_fs)
    pa = _merge_aggs(pa, ia)
    print('*** Computing collective inhibition metrics ***')
    pa = _get_unique_named(pair_fs, pa)
    print('*** Computing originality ***')
    pa = _get_originality(fs, pair_fs, pa)
    
    # Add metrics, compute aggregates, and save
    print('*** Summarizing ***')
    pa = add_metrics(pa)
    aggs = get_pair_level_aggregates(pa)
    
    # Save
    print('*** Saving ***')
    pa.to_csv(f'{ANALYSES_PATH}/{DATE}/processed.tsv', 
              sep='\t', 
              index=False)
    aggs.to_csv(f'{ANALYSES_PATH}/{DATE}/aggregates.tsv', 
                sep='\t', 
                index=False)
                            
                            
if __name__=='__main__':
    postprocess()
                            