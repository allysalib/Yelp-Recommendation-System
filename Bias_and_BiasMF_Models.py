#!/usr/bin/env python

import pandas as pd
import numpy as np
pd.set_option('display.max_columns', None)
import numba
import concurrent.futures
from lenskit.datasets import MovieLens
from lenskit.crossfold import partition_users, SampleFrac
from lenskit.algorithms.basic import Bias
from lenskit.algorithms.als import BiasedMF, ImplicitMF
from lenskit.algorithms import Recommender
from lenskit.metrics.predict import rmse, mae
from lenskit.metrics.topn import recip_rank, precision, recall, ndcg
from lenskit.batch import predict, recommend
from lenskit import batch, topn, util
from lenskit import crossfold as xf
from lenskit.algorithms import Recommender, als, item_knn as knn

#small data

json_file_list_subset = ['/scratch/as12453/base_data_restauants_subset_unique.json/part-00000-e8e3aebd-9381-4252-bafa-acb9e82eecb0-c000.json',
'/scratch/as12453/base_data_restauants_subset_unique.json/part-00001-e8e3aebd-9381-4252-bafa-acb9e82eecb0-c000.json',
'/scratch/as12453/base_data_restauants_subset_unique.json/part-00002-e8e3aebd-9381-4252-bafa-acb9e82eecb0-c000.json',
'/scratch/as12453/base_data_restauants_subset_unique.json/part-00003-e8e3aebd-9381-4252-bafa-acb9e82eecb0-c000.json',
'/scratch/as12453/base_data_restauants_subset_unique.json/part-00004-e8e3aebd-9381-4252-bafa-acb9e82eecb0-c000.json',
'/scratch/as12453/base_data_restauants_subset_unique.json/part-00005-e8e3aebd-9381-4252-bafa-acb9e82eecb0-c000.json',
'/scratch/as12453/base_data_restauants_subset_unique.json/part-00006-e8e3aebd-9381-4252-bafa-acb9e82eecb0-c000.json',
'/scratch/as12453/base_data_restauants_subset_unique.json/part-00007-e8e3aebd-9381-4252-bafa-acb9e82eecb0-c000.json',
'/scratch/as12453/base_data_restauants_subset_unique.json/part-00008-e8e3aebd-9381-4252-bafa-acb9e82eecb0-c000.json',
'/scratch/as12453/base_data_restauants_subset_unique.json/part-00009-e8e3aebd-9381-4252-bafa-acb9e82eecb0-c000.json',
'/scratch/as12453/base_data_restauants_subset_unique.json/part-00010-e8e3aebd-9381-4252-bafa-acb9e82eecb0-c000.json',
'/scratch/as12453/base_data_restauants_subset_unique.json/part-00011-e8e3aebd-9381-4252-bafa-acb9e82eecb0-c000.json']

dfs = []

for file in json_file_list_subset:
    data = pd.read_json(file, lines=True, orient = 'columns')
    dfs.append(data)
    
full_data = pd.concat(dfs)

full_data = full_data.loc[~full_data.index.duplicated(keep='first')]

full_data2 = full_data.reset_index()

ratings_data = full_data2[['user_id', 'business_id', 'review_stars']].rename(columns={'user_id':'user', 'business_id':'item', 'review_stars':'rating'})
ratings_data['rating_binary'] = 0
ratings_data.loc[(ratings_data['rating'] > 3), 'rating_binary'] = 1

N_SPLITS = 1
FRAC_SPLIT = 0.2 

train_test = list(partition_users(ratings_data, N_SPLITS, SampleFrac(FRAC_SPLIT), rng_spec=27))

train_val = []

for i in range(0, N_SPLITS):
    x = list(partition_users(train_test[i][0], 1, SampleFrac(FRAC_SPLIT), rng_spec=27))
    train_val.append(x)

test_data = []

for train, test in train_test:
    x = [test]
    for i in x:
        test_data.append(i)

def fit_eval(aname, algo, train, test):
    fittable = util.clone(algo)
    fittable = Recommender.adapt(fittable)
    model = fittable.fit(train)
    users = test.user.unique()
    # now we run the recommender
    recs_10 = batch.recommend(model, users, 10)
    recs_100 = batch.recommend(model, users, 100)
    # add the algorithm name for analyzability
    recs_10['Algorithm'] = aname
    recs_100['Algorithm'] = aname
    return model, recs_10, recs_100

def test_eval(model, test):
    users = test.user.unique()
    recs_10 = batch.recommend(model, users, 10)
    recs_100 = batch.recommend(model, users, 100)
    return recs_10, recs_100


def main():
#Bias Model

    damping_values = [0, 1, 5, 10, 15]

    bias_models = []
    bias_val_rmse = []
    bias_validation_prediction_scores_list = []
    bias_validation_evals_list = []

    print("Bias Model Validation RMSE Scores:")

    for d in damping_values:
        print("Damping Value: {}".format(d))
        count = 0
        for i in train_val:
            for train, val in i:
                B = Bias(items=True, users=False, damping=d)
                model, recs_10, recs_100 = fit_eval("Bias, Damping={}".format(d), B, train, val)
                bias_models.append([d, model])
                predictions = model.predict(val[['user', 'item']])
                rmse_score = rmse(predictions, val['rating'])
                mae_score = mae(predictions, val['rating'])
                bias_validation_prediction_scores_list.append([d, rmse_score, mae_score])
                val_binary = val[['user', 'item', 'rating_binary']].rename(columns={"rating_binary": "rating"})
                rla = topn.RecListAnalysis()
                rla.add_metric(topn.recip_rank)
                rla.add_metric(topn.precision)
                rla.add_metric(topn.recall)
                rla.add_metric(topn.ndcg)
                results_10recs = rla.compute(recs_10, val_binary)
                evals_10 = results_10recs[['recip_rank', 'precision', 'recall', 'ndcg']].rename(columns={"precision": "precision@10", "recall": "recall@10"})
                results_100recs = rla.compute(recs_100, val_binary)
                evals_100 = results_100recs[['precision', 'recall']].rename(columns={"precision": "precision@100", "recall": "recall@100"})
                evals_full = pd.concat([evals_10, evals_100], axis=1, sort=False)
                bias_validation_evals_list.append([d, evals_full])

        bias_validation_prediction_scores = pd.DataFrame(bias_validation_prediction_scores_list, columns = ['damping_factor', 'rmse', 'mae'])              

    bias_validation_evals = []
    
    for i in range(len(bias_validation_evals_list)):
        data = bias_validation_evals_list[i][1]
        data['damping_factor'] = bias_validation_evals_list[i][0]
        bias_validation_evals.append(data)

    bias_validation_evals_df = pd.concat(bias_validation_evals, ignore_index=True)

    print("Bias models validation prediction scores by damping factor:")
    print(bias_validation_prediction_scores)

    bias_validation_evals_aggregated = bias_validation_evals_df.groupby(['damping_factor']).agg({'recip_rank': ['mean', 'std'], 'precision@10': ['mean', 'std'], 'precision@100': ['mean', 'std'], 'recall@10': ['mean', 'std'], 'recall@100': ['mean', 'std'], 'ndcg': ['mean', 'std'], })
    print("Bias models validation evaluations by damping factor:")
    print(bias_validation_evals_aggregated)

    #chose parameter to opt:
    best_bias_params = bias_validation_evals_aggregated['ndcg']['mean'].idxmax()
    print("Best Hyperparameters:")
    print("Damping value = {}".format(best_bias_params))

    best_bias_models = []

    for b in bias_models:
        if b[0] == best_bias_params: best_bias_models.append(b)
        
    bias_test_prediction_scores_list = []
    bias_test_evals_list = []

    print("Best Bias Model Test Evaluations:")

    for i in best_bias_models:
        model = i[1]
        test = test_data[0]
        predictions = model.predict(test[['user', 'item']])
        rmse_score = rmse(predictions, test['rating'])
        mae_score = mae(predictions, test['rating'])
        bias_test_prediction_scores_list.append([rmse_score, mae_score])
        recs_10, recs_100 = test_eval(model, test)
        test_binary = test[['user', 'item', 'rating_binary']].rename(columns={"rating_binary": "rating"})
        rla = topn.RecListAnalysis()
        rla.add_metric(topn.recip_rank)
        rla.add_metric(topn.precision)
        rla.add_metric(topn.recall)
        rla.add_metric(topn.ndcg)
        results_10recs = rla.compute(recs_10, test_binary)
        print("evals running")
        evals_10 = results_10recs[['recip_rank', 'precision', 'recall', 'ndcg']].rename(columns={"precision": "precision@10", "recall": "recall@10"})
        results_100recs = rla.compute(recs_100, test_binary)
        evals_100 = results_100recs[['precision', 'recall']].rename(columns={"precision": "precision@100", "recall": "recall@100"})
        evals_full = pd.concat([evals_10, evals_100], axis=1, sort=False)
        bias_test_evals_list.append(evals_full)

    bias_test_prediction_scores = pd.DataFrame(bias_test_prediction_scores_list, columns = ['rmse', 'mae'])              

    bias_test_evals = bias_test_evals_list[0]

    print("Best Bias model test aggregated prediction scores:")
    print(bias_test_prediction_scores)

    bias_test_evals_agg = bias_test_evals.agg({'recip_rank': ['mean', 'std'], 'precision@10': ['mean', 'std'], 'precision@100': ['mean', 'std'], 'recall@10': ['mean', 'std'], 'recall@100': ['mean', 'std'], 'ndcg': ['mean', 'std'], })
    print("Best Bias model test aggregated evaluations:")
    print(bias_test_evals_agg)

if __name__ == '__main__':
    main()
