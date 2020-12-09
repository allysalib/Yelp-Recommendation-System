#!/usr/bin/env python

import pandas as pd
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

data = pd.read_csv('/scratch/as12453/Subset/base_subset.csv')

ratings = data.rename(columns={'user_id':'user', 'business_id':'item', 'review_stars':'rating'})
ratings['rating_binary'] = 0
ratings.loc[(ratings['rating'] > 3), 'rating_binary'] = 1

FRAC_SPLIT = 0.2
N_SPLIT = 5

def fit_eval(aname, algo, train, test):
    fittable = util.clone(algo)
    fittable = Recommender.adapt(fittable)
    model = fittable.fit(train)
    users = test.user.unique()
    # now we run the recommender
    recs_10 = batch.recommend(model, users, 10)
    #recs_100 = batch.recommend(model, users, 100)
    # add the algorithm name for analyzability
    recs_10['Algorithm'] = aname
    #recs_100['Algorithm'] = aname
    return model, recs_10

def test_eval(model, test):
    users = test.user.unique()
    recs_10 = batch.recommend(model, users, 10)
    #recs_100 = batch.recommend(model, users, 100)
    return recs_10

def main():
    damping_values = [0, 2, 5, 10]

    bias_test_splits = []
    bias_validation_prediction_scores_list = []
    bias_validation_evals_list = []
    bias_models = []

    for d in damping_values:
        print(d)
        count = 0
        for train_val, test in partition_users(ratings, N_SPLIT, SampleFrac(FRAC_SPLIT, rng_spec=111), rng_spec=111):
            bias_test_splits.append(test)
            for train, val in partition_users(train_val, 1, SampleFrac(FRAC_SPLIT, rng_spec=111), rng_spec=111):
                count += 1
                print(count)
                B = Bias(items=True, users=False, damping=d)
                model, recs_10 = fit_eval("Bias, Damping={}".format(d), B, train, val)
                bias_models.append([(count-1), d, model])
                predictions = model.predict(val[['user', 'item']])
                rmse_score = rmse(predictions, val['rating'])
                mae_score = mae(predictions, val['rating'])
                bias_validation_prediction_scores_list.append([(count-1), d, rmse_score, mae_score])
                val_binary = val[['user', 'item', 'rating_binary']].rename(columns={"rating_binary": "rating"})
                rla = topn.RecListAnalysis()
                rla.add_metric(topn.recip_rank)
                rla.add_metric(topn.precision)
                rla.add_metric(topn.recall)
                rla.add_metric(topn.ndcg)
                results_10recs = rla.compute(recs_10, val_binary)
                evals_10 = results_10recs[['recip_rank', 'precision', 'recall', 'ndcg']].rename(columns={"precision": "precision@10", "recall": "recall@10"})
                #results_100recs = rla.compute(recs_100, val_binary)
                #evals_100 = results_100recs[['precision', 'recall']].rename(columns={"precision": "precision@100", "recall": "recall@100"})
                #evals_full = pd.concat([evals_10, evals_100], axis=1, sort=False)
                bias_validation_evals_list.append([(count-1), d, evals_10])

    bias_validation_prediction_scores = pd.DataFrame(bias_validation_prediction_scores_list, columns = ['split', 'damping_factor', 'rmse', 'mae'])              

    bias_validation_evals = []

    for i in range(len(bias_validation_evals_list)):
        data = bias_validation_evals_list[i][2]
        data['damping_factor'] = bias_validation_evals_list[i][1]
        data['split'] = bias_validation_evals_list[i][0]
        bias_validation_evals.append(data)

    bias_validation_evals_df = pd.concat(bias_validation_evals, ignore_index=True)

    print("Bias models validation prediction scores by split:")
    print(bias_validation_prediction_scores)

    bias_validation_evals_splits = bias_validation_evals_df.groupby(['damping_factor', 'split']).agg({'recip_rank': ['mean', 'std'], 'precision@10': ['mean', 'std'], 'recall@10': ['mean', 'std'], 'ndcg': ['mean', 'std'], })
    print("Bias models validation evaluations by split:")
    print(bias_validation_evals_splits)

    bias_validation_prediction_scores_aggregated = bias_validation_prediction_scores.groupby(['damping_factor']).agg({'rmse': ['mean', 'std'], 'mae': ['mean', 'std']})
    print("Aggregated bias models validation prediction scores by parameters:")
    print(bias_validation_prediction_scores_aggregated)

    bias_validation_evals_aggregated = bias_validation_evals_df.groupby(['damping_factor']).agg({'recip_rank': ['mean', 'std'], 'precision@10': ['mean', 'std'], 'recall@10': ['mean', 'std'], 'ndcg': ['mean', 'std'], })
    print("Aggregated bias models validation evaluations by parameters:")
    print(bias_validation_evals_aggregated)

    #chose parameter to opt:
    best_bias_params = bias_validation_evals_aggregated['ndcg']['mean'].idxmax()
    print("Best Hyperparameters:")
    print("Damping value = {}".format(best_bias_params))

    best_bias_models = []

    for b in bias_models:
        if b[1] == best_bias_params: best_bias_models.append(b)

    bias_test_prediction_scores_list = []
    bias_test_evals_list = []

    print("Best Bias Model Test Evaluations:")

    for i in best_bias_models:
        model = i[2]
        count = i[0]
        test_data = bias_test_splits[count]
        predictions = model.predict(test_data[['user', 'item']])
        rmse_score = rmse(predictions, test_data['rating'])
        mae_score = mae(predictions, test_data['rating'])
        bias_test_prediction_scores_list.append([count, rmse_score, mae_score])
        test_binary = test_data[['user', 'item', 'rating_binary']].rename(columns={"rating_binary": "rating"})
        recs_10 = test_eval(model, test_data)
        rla = topn.RecListAnalysis()
        rla.add_metric(topn.recip_rank)
        rla.add_metric(topn.precision)
        rla.add_metric(topn.recall)
        rla.add_metric(topn.ndcg)
        results_10recs = rla.compute(recs_10, test_binary)
        evals_10 = results_10recs[['recip_rank', 'precision', 'recall', 'ndcg']].rename(columns={"precision": "precision@10", "recall": "recall@10"})
        #results_100recs = rla.compute(recs_100, test_binary)
        #evals_100 = results_100recs[['precision', 'recall']].rename(columns={"precision": "precision@100", "recall": "recall@100"})
        evals_full = pd.concat([evals_10], axis=1, sort=False)
        bias_test_evals_list.append([count, evals_10])

    bias_test_prediction_scores = pd.DataFrame(bias_test_prediction_scores_list, columns = ['split', 'rmse', 'mae'])              

    bias_test_evals = []

    for i in range(len(bias_test_evals_list)):
        data = bias_test_evals_list[i][1]
        data['split'] = bias_test_evals_list[i][0]
        bias_test_evals.append(data)

    bias_test_evals_df = pd.concat(bias_test_evals, ignore_index=True)

    print("Bias models test prediction scores by split:")
    print(bias_test_prediction_scores)

    bias_test_evals_splits = bias_test_evals_df.groupby(['split']).agg({'recip_rank': ['mean', 'std'], 'precision@10': ['mean', 'std'], 'recall@10': ['mean', 'std'], 'ndcg': ['mean', 'std'], })
    print("Bias models test evaluations by split:")
    print(bias_test_evals_splits)

    bias_test_prediction_scores_aggregated = bias_test_prediction_scores.agg({'rmse': ['mean', 'std'], 'mae': ['mean', 'std']})
    print("Aggregated Bias models test prediction scores for best parameters:")
    print(bias_test_prediction_scores_aggregated)

    bias_test_evals_aggregated = bias_test_evals_df.agg({'recip_rank': ['mean', 'std'], 'precision@10': ['mean', 'std'],  'recall@10': ['mean', 'std'], 'ndcg': ['mean', 'std'], })
    print("Aggregated Bias models test evaluations for best parameters:")
    print(bias_test_evals_aggregated)

    regularization_values = [0, 0.1, 0.5]
    dimensionality_values = [50, 250]

    biasMF_test_splits = []
    biasMF_validation_prediction_scores_list = []
    biasMF_validation_evals_list = []
    biasMF_models = []

    for r in regularization_values:
        for f in dimensionality_values:
            count = 0
            for train_val, test in partition_users(ratings2, N_SPLIT, SampleFrac(FRAC_SPLIT, rng_spec= 111), rng_spec=111):
                biasMF_test_splits.append(test)
                for train, val in partition_users(train_val, N_SPLIT, SampleFrac(FRAC_SPLIT, rng_spec=111), rng_spec=111):
                    count += 1
                    BMF = BiasedMF(features=f, reg=r, rng_spec=111)
                    model, recs_10 = fit_eval("BiasMF, Reg={},Dim={}".format(r,f), BMF, train, val)
                    biasMF_models.append([(count-1), r, f, model])
                    predictions = model.predict(val[['user', 'item']])
                    rmse_score = rmse(predictions, val['rating'])
                    mae_score = mae(predictions, val['rating'])
                    biasMF_validation_prediction_scores_list.append([(count-1), r, f, rmse_score, mae_score])
                    val_binary = val[['user', 'item', 'rating_binary']].rename(columns={"rating_binary": "rating"})
                    rla = topn.RecListAnalysis()
                    rla.add_metric(topn.recip_rank)
                    rla.add_metric(topn.precision)
                    rla.add_metric(topn.recall)
                    rla.add_metric(topn.ndcg)
                    results_10recs = rla.compute(recs_10, val_binary)
                    evals_10 = results_10recs[['recip_rank', 'precision', 'recall', 'ndcg']].rename(columns={"precision": "precision@10", "recall": "recall@10"})
                    #results_100recs = rla.compute(recs_100, val_binary)
                    #evals_100 = results_100recs[['precision', 'recall']].rename(columns={"precision": "precision@100", "recall": "recall@100"})
                    #evals_full = pd.concat([evals_10, evals_100], axis=1, sort=False)
                    biasMF_validation_evals_list.append([(count-1), r, f, evals_10])

    biasMF_validation_prediction_scores = pd.DataFrame(biasMF_validation_prediction_scores_list, columns = ['split', 'regularization', 'features', 'rmse', 'mae'])              

    biasMF_validation_evals = []

    for i in range(len(biasMF_validation_evals_list)):
        data = biasMF_validation_evals_list[i][3]
        data['regularization'] = biasMF_validation_evals_list[i][1]
        data['features'] = biasMF_validation_evals_list[i][2]
        data['split'] = biasMF_validation_evals_list[i][0]
        biasMF_validation_evals.append(data)

    biasMF_validation_evals_df = pd.concat(biasMF_validation_evals, ignore_index=True)

    print("BiasMF models validation prediction scores by split:")
    print(biasMF_validation_prediction_scores)

    biasMF_validation_evals_splits = biasMF_validation_evals_df.groupby(['regularization', 'features', 'split']).agg({'recip_rank': ['mean', 'std'], 'precision@10': ['mean', 'std'], 'precision@100': ['mean', 'std'], 'recall@10': ['mean', 'std'], 'recall@100': ['mean', 'std'], 'ndcg': ['mean', 'std'], })
    print("BiasMF models validation evaluations by split:")
    print(biasMF_validation_evals_splits)

    biasMF_validation_prediction_scores_aggregated = biasMF_validation_prediction_scores.groupby(['regularization', 'features']).agg({'rmse': ['mean', 'std'], 'mae': ['mean', 'std']})
    print("Aggregated BiasMF models validation prediction scores by parameters:")
    print(biasMF_validation_prediction_scores_aggregated)

    biasMF_validation_evals_aggregated = biasMF_validation_evals_df.groupby(['regularization', 'features']).agg({'recip_rank': ['mean', 'std'], 'precision@10': ['mean', 'std'], 'precision@100': ['mean', 'std'], 'recall@10': ['mean', 'std'], 'recall@100': ['mean', 'std'], 'ndcg': ['mean', 'std'], })
    print("Aggregated BiasMF models validation evaluations by parameters:")
    print(biasMF_validation_evals_aggregated)

    #chose parameter to opt:
    best_biasMF_params = biasMF_validation_evals_aggregated['ndcg']['mean'].idxmax()
    print(best_biasMF_params)
    print("Best Hyperparameters:")
    print("Regularization = {}, Features = {}".format(best_biasMF_params[0], best_biasMF_params[1]))

    best_biasMF_models = []

    for b in biasMF_models:
        if (b[1] == best_biasMF_params[0] and b[2] == best_biasMF_params[1]) : best_biasMF_models.append(b)

    biasMF_test_prediction_scores_list = []
    biasMF_test_evals_list = []

    print("Best BiasMF Model Test Evaluations:")

    for i in best_biasMF_models:
        model = i[3]
        count = i[0]
        test_data = biasMF_test_splits[count]
        predictions = model.predict(test_data[['user', 'item']])
        rmse_score = rmse(predictions, test_data['rating'])
        mae_score = mae(predictions, test_data['rating'])
        biasMF_test_prediction_scores_list.append([count, rmse_score, mae_score])
        test_binary = test_data[['user', 'item', 'rating_binary']].rename(columns={"rating_binary": "rating"})
        recs_10 = test_eval(model, test_data)
        rla = topn.RecListAnalysis()
        rla.add_metric(topn.recip_rank)
        rla.add_metric(topn.precision)
        rla.add_metric(topn.recall)
        rla.add_metric(topn.ndcg)
        results_10recs = rla.compute(recs_10, test_binary)
        evals_10 = results_10recs[['recip_rank', 'precision', 'recall', 'ndcg']].rename(columns={"precision": "precision@10", "recall": "recall@10"})
        #results_100recs = rla.compute(recs_100, test_binary)
        #evals_100 = results_100recs[['precision', 'recall']].rename(columns={"precision": "precision@100", "recall": "recall@100"})
        #evals_full = pd.concat([evals_10, evals_100], axis=1, sort=False)
        biasMF_test_evals_list.append([count, evals_10])

    biasMF_test_prediction_scores = pd.DataFrame(biasMF_test_prediction_scores_list, columns = ['split', 'rmse', 'mae'])              

    biasMF_test_evals = []

    for i in range(len(biasMF_test_evals_list)):
        data = biasMF_test_evals_list[i][1]
        data['split'] = biasMF_test_evals_list[i][0]
        biasMF_test_evals.append(data)

    biasMF_test_evals_df = pd.concat(biasMF_test_evals, ignore_index=True)

    print("BiasMF models test prediction scores by split:")
    print(biasMF_test_prediction_scores)

    biasMF_test_evals_splits = biasMF_test_evals_df.groupby(['split']).agg({'recip_rank': ['mean', 'std'], 'precision@10': ['mean', 'std'], 'recall@10': ['mean', 'std'], 'ndcg': ['mean', 'std'], })
    print("BiasMF models test evaluations by split:")
    print(biasMF_test_evals_splits)

    biasMF_test_prediction_scores_aggregated = biasMF_test_prediction_scores.agg({'rmse': ['mean', 'std'], 'mae': ['mean', 'std']})
    print("Aggregated BiasMF models test prediction scores for best parameters:")
    print(biasMF_test_prediction_scores_aggregated)

    biasMF_test_evals_aggregated = biasMF_test_evals_df.agg({'recip_rank': ['mean', 'std'], 'precision@10': ['mean', 'std'], 'recall@10': ['mean', 'std'],  'ndcg': ['mean', 'std'], })
    print("Aggregated BiasMF models test evaluations for best parameters:")
    print(biasMF_test_evals_aggregated)

if __name__ == '__main__':
    main()
