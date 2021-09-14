import pandas as pd
import numpy as np


INDIVIDUAL_DICT = {'turn': 'max',
                   'threshold': 'first',
                   'agent': 'first',
                   'init_seed': 'first',
                   'prob0': 'mean',
                   'ndens0': 'mean',
                   'ndens_current0': 'mean',
                   'response': 'last'}

PAIR_DICT = {'turn':'max', 
             'log_id': 'first',
             'agent_0': 'first',
             'agent_1': 'first',
             'threshold':'first',
             'init_seed': 'first',
             'prob0': 'mean', 
             'prob1': 'mean',
             'jump_speaker': 'mean',
             'jump_listener': 'mean',
             'ndens0': 'mean',
             'ndens1': 'mean',
             'ndens_current0': 'mean',
             'ndens_current1':'mean',
             'response': 'last'}

INDIVIDUAL_NAMES = ['iter', 'performance', 'threshold', 
                           'agent_name', 'init_seed', 
                           'mean_jump',
                           'mean_neighborhood_density',
                           'mean_neighborhood_density_current',
                           'last_response']
PAIR_NAMES = ['iter', 'performance', 'pair',
                     'agent_0', 'agent_1',
                     'threshold', 'init_seed',
                     'mean_jump_0',
                     'mean_jump_1', 
                     'mean_jump_speaker', 'mean_jump_listener',
                     'mean_neighborhood_density_0', 
                     'mean_neighborhood_density_1',
                     'mean_neighborhood_density_current_0', 
                     'mean_neighborhood_density_current_1', 
                     'last_response']

RENAME_0 = {'performance_x': 'performance_pair',
            'performance_y': 'performance_a0',
            'mean_neighborhood_density': 'neighborhood_density_a0',
            'mean_neighborhood_density_current': 'neighborhood_density_current_a0_individual',
            'noise_level_y': 'noise_level_a0',
            'mean_jump': 'mean_jump_a0',
            'last_response_x': 'last_response_pair',
            'last_response_y': 'last_response_a0'}

RENAME_1 = {'performance': 'performance_a1',
            'mean_neighborhood_density': 'neighborhood_density_a1_individual',
            'mean_neighborhood_density_current': 'neighborhood_density_current_a1_individual',
            'mean_jump': 'mean_jump_a1',
            'last_response': 'last_response_a1',
            'noise_level': 'noise_level_a1'}


PAIR_LEVEL_DICT = {'pos_gain': 'mean', 
                   'is_gain': ['mean', lambda x: np.std(x)],
                   'amount_gain': ['mean', lambda x: np.std(x)],
                   'noise_level_a0': 'mean',
                   'performance_difference_individuals': 'mean',
                   'performance_best': 'mean',
                   'performance_pair': ['mean', 'std'],
                   'mean_jump_speaker': 'mean',
                   'mean_jump_listener': 'mean',
                   'orig_pair': 'mean',
                   'orig_best_difference': 'mean',
                   'collective_inhibition': 'mean'}


PAIR_LEVEL_NAMES = ['pair', 
                    'pos_gain', 
                    'is_gain_mean', 'is_gain_std',
                    'amount_gain_mean', 'amount_gain_std',
                    'noise_level_a0',
                    'performance_difference_individuals', 
                    'performance_best', 
                    'performance_pair_mean', 
                    'performance_pair_std',
                    'mean_jump_speaker', 
                    'mean_jump_listener',
                    'orig_pair',
                    'orig_best_difference',
                    'collective_inhibition']


def get_individual_aggs(f):
    ''' Function to compute aggregates in individual performance data'''
    log = pd.read_csv(f)
    ind_agg = log.groupby('iter').agg(INDIVIDUAL_DICT).reset_index()
    ind_agg.columns = INDIVIDUAL_NAMES
    ind_agg['noise_level'] = ind_agg['agent_name'].str.split('_').str[1].astype(float)
    return ind_agg


def get_pair_aggs(f):
    ''' Function to compute aggregates in pair performance data'''
    log = pd.read_csv(f)
    log['agent_0'] = log['log_id'].str.split('_').str[:3].str.join('_').iloc[0]
    log['agent_1'] = log['log_id'].str.split('_').str[3:].str.join('_').iloc[0]
    log['agent_speaking'] = np.where(log['agent']==log['agent_0'], 
                                     'agent_0', 
                                     'agent_1')
    log['jump_speaker'] = np.where(log['agent_speaking']=='agent_0', 
                                   log['prob0'], 
                                   log['prob1'])
    log['jump_listener'] = np.where(log['agent_speaking']=='agent_0', 
                                    log['prob1'], 
                                    log['prob0'])
    log['jump_difference'] = log['jump_listener'] - log['jump_speaker']
    pair_agg = log.groupby('iter').agg(PAIR_DICT).reset_index()
    pair_agg.columns = PAIR_NAMES
    pair_agg['noise_level'] = pair_agg['agent_0'].str.split('_').str[1].astype(float)
    return pair_agg


def concat_dfs(result_list):
    ''' Concatenate dataframes in list as single df'''
    for idx, r in enumerate(result_list):
        if idx == 0:
            out = r
        else:
            out = pd.concat([out, r], 
                            ignore_index=True)
    return out


def merge_pairs_inds(pdf, idf, agent_nr):
    ''' Merge df of pair aggregate metrics with individual aggregates '''
    pdf = pdf.merge(idf, 
                    right_on=['agent_name', 'init_seed'],
                    left_on=[f'agent_{agent_nr}', 'init_seed']).drop(['agent_name'], axis=1)
    if agent_nr == '0':
        pdf.drop('noise_level_x', axis=1, inplace=True)
        pdf = pdf.rename(RENAME_0, axis=1)
    else:
        pdf.drop('noise_level', axis=1, inplace=True)
        pdf = pdf.rename(RENAME_1, axis=1)
    return pdf


def get_unique_named(f, log_path, date):
    '''
        Get dataframe with number of unique words named per trial, 
        both for pairs (= performance) and for concatenated lists
        of animals named by individuals 
    '''
    log = pd.read_csv(f)
    pair_id = log.log_id.iloc[0]
    pair_counts = log.groupby('init_seed').response.count().reset_index()
    pair_counts = pair_counts.rename({'response': 
                                      'unique_pair'}, axis=1)
    a0_name = log['log_id'].str.split('_').str[:3].str.join('_').iloc[0]
    a1_name = log['log_id'].str.split('_').str[3:].str.join('_').iloc[0]
    a0_log = pd.read_csv(f'{log_path}/{date}/individual/{a0_name}_1_0.01179.txt')
    a1_log = pd.read_csv(f'{log_path}/{date}/individual/{a1_name}_1_0.01179.txt')
    a0_list = a0_log.groupby('init_seed').agg({'response': list}).reset_index()
    a1_list = a1_log.groupby('init_seed').agg({'response': list}).reset_index()
    counts = a0_list.merge(a1_list, on='init_seed')
    counts['unique_individual'] = (counts['response_x'] + \
                                   counts['response_y']).apply(lambda x: len(set(x)))
    counts = counts.drop(['response_x', 'response_y'], axis=1)
    counts = pair_counts.merge(counts, on='init_seed')
    counts['pair'] = pair_id
    return counts


def get_wd_originality_scores(f):
    ''' Compute the originality score for each word '''
    counts = pd.read_csv(f).groupby('response')['agent'].count().reset_index()
    counts.columns = ['word', 'count']
    return counts


def _process_originality_df(log, merge_col, idx, wd_orig):
    ''' Process datasets for originality '''
    merged = log.merge(wd_orig[merge_col], 
                       left_on='response', 
                       right_on='word').drop('word', axis=1)
    merged = merged.groupby('init_seed').agg({'originality_score': 
                                              'mean'}).reset_index()
    merged.columns = ['init_seed', f'orig_{idx}']
    return merged
    
    
def get_originality(f, wd_orig, log_path, date):
    ''' Compute avg originality for individuals and pairs '''
    # Process pair
    merge_col = ['word', 'originality_score']
    log = pd.read_csv(f)
    pair_id = log.log_id.iloc[0]
    merged_pair = _process_originality_df(log, merge_col, 'pair', wd_orig)
    merged_pair['pair'] = pair_id
    # Process individual
    a0_name = log['log_id'].str.split('_').str[:3].str.join('_').iloc[0]
    a0_log = pd.read_csv(f'{log_path}/{date}/individual/{a0_name}_1_0.01179.txt')
    a1_name = log['log_id'].str.split('_').str[3:].str.join('_').iloc[0]
    a1_log = pd.read_csv(f'{log_path}/{date}/individual/{a1_name}_1_0.01179.txt')
    merged_a0 = _process_originality_df(a0_log, merge_col, 'a0', wd_orig)
    merged_a1 = _process_originality_df(a1_log, merge_col, 'a1', wd_orig)
    # Merge stuff
    merged_pair = merged_pair.merge(merged_a0, on='init_seed')
    merged_pair = merged_pair.merge(merged_a1, on='init_seed')
    return merged_pair



def add_metrics(df): 
    df['performance_best'] = np.where(df['performance_a1']>\
                                      df['performance_a0'], 
                                      df['performance_a1'],
                                      df['performance_a0']).astype(int)
    df['performance_difference_individuals'] = abs(df['performance_a1'] - \
                                                   df['performance_a0'])
    df['is_gain'] = (df['performance_pair'] > \
                     df['performance_best'])
    df['amount_gain'] = df['performance_pair'] - \
                        df['performance_best']
    df['pos_gain'] = np.where(df['amount_gain']>0, 
                              df['amount_gain'], 0)
    df['mean_jump_best_individual'] = np.where(df['performance_best']==\
                                               df['performance_a0'],
                                               df['mean_jump_a0'], 
                                               df['mean_jump_a1'])
    df['pair_best_jump_difference'] = df['mean_jump_speaker'] - \
                                      df['mean_jump_best_individual']
    df['orig_best_difference'] = np.where(df['performance_a1']>\
                                          df['performance_a0'],
                                          df['orig_pair']-df['orig_a1'],
                                          df['orig_pair']-df['orig_a0'],)
    return df


def get_pair_level_aggregates(pdf):
    ''' Compute pair-level aggregates '''
    aggs = pdf.groupby('pair').agg(PAIR_LEVEL_DICT).reset_index()
    aggs.columns = PAIR_LEVEL_NAMES
    return aggs
