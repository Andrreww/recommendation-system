# -*- coding: utf-8 -*-
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('nbagg')
import time
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
import os
from scipy import sparse
from scipy.sparse import csr_matrix

from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import random
import math

print("creating the dataframe from data.csv file..")
df = pd.read_csv('C:/Users/lenovo/Desktop/ratings_Video_Games.csv', sep=',', 
                       names=['user','game','rating','datestamp','date'])
df.head()
df.date = pd.to_datetime(df.date)
df.sort_values(by='date', inplace=True)
df = df[df.datestamp>1259676800]
df.head()

vc = df.user.value_counts()
k = pd.DataFrame({'user':vc.index,'count2':vc.values})
df_final = pd.merge(df,k,on='user')
df_final2 = df_final[df_final.count2>=5]
Goodgame = df_final2.game.value_counts().head(3000).index
selectgame = random.sample(list(Goodgame),2000)
df = df_final2[df_final2.game.isin(selectgame)]

n_users = len(df.user.unique())
n_games = len(df.game.unique())

train_df = df.iloc[:int(df.shape[0]*0.80)]
test_df = df.iloc[int(df.shape[0]*0.80):]

n_users_t = len(train_df.user.unique())
n_games_t = len(train_df.game.unique())
n_users_test = len(test_df.user.unique())
n_games_test = len(test_df.game.unique())

train_user2idx = {user: i for i, user in enumerate(train_df.user.unique())}
train_idx2user = {i: user for user, i in train_user2idx.items()}
train_game2idx = {game: i for i, game in enumerate(train_df.game.unique())}
train_idx2game = {i: game for game, i in train_game2idx.items()}

test_user2idx = {user: i for i, user in enumerate(test_df.user.unique())}
test_idx2user = {i: user for user, i in test_user2idx.items()}
test_game2idx = {game: i for i, game in enumerate(test_df.game.unique())}
test_idx2game = {i: game for game, i in test_game2idx.items()}

user_idx_train = train_df['user'].apply(lambda x: train_user2idx[x]).values
game_idx_train = train_df['game'].apply(lambda x: train_game2idx[x]).values
rating_train = train_df['rating'].values

user_idx_test = test_df['user'].apply(lambda x: test_user2idx[x]).values
game_idx_test = test_df['game'].apply(lambda x: test_game2idx[x]).values
rating_test = test_df['rating'].values

train_sparse_matrix = sparse.csr_matrix((rating_train, (user_idx_train,game_idx_train)))
test_sparse_matrix = sparse.csr_matrix((rating_test, (user_idx_test,game_idx_test)))

def get_average_ratings(sparse_matrix, of_users):

    ax = 1 if of_users else 0 
    sum_of_ratings = sparse_matrix.sum(axis=ax).A1
    is_rated = sparse_matrix!=0
    no_of_ratings = is_rated.sum(axis=ax).A1
    u,m = sparse_matrix.shape
    average_ratings = { i : sum_of_ratings[i]/no_of_ratings[i]
                                 for i in range(u if of_users else m) 
                                    if no_of_ratings[i] !=0}
    return average_ratings
train_averages = dict()
train_global_average = train_sparse_matrix.sum()/train_sparse_matrix.count_nonzero()
train_averages['global'] = train_global_average
train_averages['user'] = get_average_ratings(train_sparse_matrix, of_users=True)
train_averages['movie'] =  get_average_ratings(train_sparse_matrix, of_users=False)

from sklearn.metrics.pairwise import cosine_similarity
g_g_sim_sparse_train = cosine_similarity(X=train_sparse_matrix.T, dense_output=False)

similar_games_train = dict()
for game in game_idx_train:
    sim_games = g_g_sim_sparse_train[game].toarray().ravel().argsort()[::-1][1:]
    similar_games_train[game] = sim_games[:100]

######Machine Learning Models
def get_sample_sparse_matrix(sparse_matrix, no_users, no_games,verbose = True):
    row_ind, col_ind, ratings = sparse.find(sparse_matrix)
    users = np.unique(row_ind)
    games = np.unique(col_ind)

    np.random.seed(15)
    sample_users = np.random.choice(users, no_users, replace=False)
    sample_games = np.random.choice(games, no_games, replace=False)
    mask = np.logical_and( np.isin(row_ind, sample_users),
                      np.isin(col_ind, sample_games) )
    
    sample_sparse_matrix = sparse.csr_matrix((ratings[mask], (row_ind[mask], col_ind[mask])),
                                             shape=(max(sample_users)+1, max(sample_games)+1))

    return sample_sparse_matrix

sample_train_sparse_matrix = get_sample_sparse_matrix(train_sparse_matrix, no_users=8000, no_games=1500)
sample_test_sparse_matrix = get_sample_sparse_matrix(test_sparse_matrix, no_users=2000, no_games=1200)

sample_train_averages = dict()
global_average = sample_train_sparse_matrix.sum()/sample_train_sparse_matrix.count_nonzero()
sample_train_averages['global'] = global_average
sample_train_averages['user'] = get_average_ratings(sample_train_sparse_matrix, of_users=True)
sample_train_averages['game'] =  get_average_ratings(sample_train_sparse_matrix, of_users=False)

sample_test_averages = dict()
global_average = sample_test_sparse_matrix.sum()/sample_test_sparse_matrix.count_nonzero()
sample_test_averages['global'] = global_average
sample_test_averages['user'] = get_average_ratings(sample_test_sparse_matrix, of_users=True)
sample_test_averages['game'] =  get_average_ratings(sample_test_sparse_matrix, of_users=False)

def regression(matrix,sample_train_averages):
    sample_train_users, sample_train_games, sample_train_ratings = sparse.find(matrix)
    count = 0
    reg_train=[]
    for (user, game, rating)  in zip(sample_train_users, sample_train_games, sample_train_ratings):
         
        user_sim = cosine_similarity(sample_train_sparse_matrix[user], sample_train_sparse_matrix).ravel()
        top_sim_users = user_sim.argsort()[::-1][1:]
        top_ratings = sample_train_sparse_matrix[top_sim_users, game].toarray().ravel()
        top_sim_users_ratings = list(top_ratings[top_ratings != 0][:5])
        top_sim_users_ratings.extend([sample_train_averages['game'][game]]*(5 - len(top_sim_users_ratings)))      
        game_sim = cosine_similarity(sample_train_sparse_matrix[:,game].T, sample_train_sparse_matrix.T).ravel()
        top_sim_games = game_sim.argsort()[::-1][1:]
        top_ratings = sample_train_sparse_matrix[user, top_sim_games].toarray().ravel()
        top_sim_games_ratings = list(top_ratings[top_ratings != 0][:5])
        top_sim_games_ratings.extend([sample_train_averages['user'][user]]*(5-len(top_sim_games_ratings))) 
        row = list()
        row.append(user)
        row.append(game)
        row.append(sample_train_averages['global'])
        row.extend(top_sim_users_ratings)
        row.extend(top_sim_games_ratings)
        row.append(sample_train_averages['user'][user])
        row.append(sample_train_averages['game'][game])
        row.append(rating)
        count = count + 1
        reg_train.append(row)
    return(reg_train)
reg_train = regression(sample_train_sparse_matrix,sample_train_averages)
reg_train_sample = pd.DataFrame(reg_train,columns = ['user', 'game', 'GAvg', 'sur1', 'sur2', 'sur3', 'sur4', 'sur5','smr1', 'smr2', 'smr3', 'smr4', 'smr5', 'UAvg', 'MAvg', 'rating'])
reg_train_sample.to_csv('reg_train_sample.csv')

reg_test = regression(sample_test_sparse_matrix,sample_test_averages)
reg_test_sample = pd.DataFrame(reg_test,columns = ['user', 'game', 'GAvg', 'sur1', 'sur2', 'sur3', 'sur4', 'sur5','smr1', 'smr2', 'smr3', 'smr4', 'smr5', 'UAvg', 'MAvg', 'rating'])
reg_test_sample.to_csv('reg_test_sample.csv')

######################training
import surprise
from surprise import Reader, Dataset

reader = Reader(rating_scale=(1,5))
train_data = Dataset.load_from_df(reg_train_sample[['user', 'game', 'rating']], reader)
trainset = train_data.build_full_trainset()
testset = list(zip(reg_test_sample.user.values, reg_test_sample.game.values, reg_test_sample.rating.values))

def get_error_metrics(y_true, y_pred):
    rmse = np.sqrt(np.mean([ (y_true[i] - y_pred[i])**2 for i in range(len(y_pred)) ]))
    mape = np.mean(np.abs( (y_true - y_pred)/y_true )) * 100
    return rmse, mape

def run_xgboost(algo,  x_train, y_train, x_test, y_test, verbose=True):
    train_results = dict()
    test_results = dict()
    print('Training the model..')
    algo.fit(x_train, y_train, eval_metric = 'rmse')
    print('Done \n')
    print('Evaluating the model with TRAIN data...')
    y_train_pred = algo.predict(x_train)
    rmse_train, mape_train = get_error_metrics(y_train.values, y_train_pred)
    train_results = {'rmse': rmse_train,
                    'mape' : mape_train,
                    'predictions' : y_train_pred}
    print('Evaluating Test data')
    y_test_pred = algo.predict(x_test) 
    rmse_test, mape_test = get_error_metrics(y_true=y_test.values, y_pred=y_test_pred)

    test_results = {'rmse': rmse_test,
                    'mape' : mape_test,
                    'predictions':y_test_pred}
    if verbose:
        print('\nTEST DATA')
        print('-'*30)
        print('RMSE : ', rmse_test)
        print('MAPE : ', mape_test)
    return train_results, test_results

my_seed = 15
random.seed(my_seed)
np.random.seed(my_seed)

def get_ratings(predictions):
    actual = np.array([pred.r_ui for pred in predictions])
    pred = np.array([pred.est for pred in predictions])
    return actual, pred

def get_errors(predictions, print_them=False):

    actual, pred = get_ratings(predictions)
    rmse = np.sqrt(np.mean((pred - actual)**2))
    mape = np.mean(np.abs(pred - actual)/actual)
    return rmse, mape*100

def run_surprise(algo, trainset, testset, verbose=True): 

    train = dict()
    test = dict()

    print('Training the model...')
    algo.fit(trainset)   
    print('Evaluating the model with train data..')
    train_preds = algo.test(trainset.build_testset())
    train_actual_ratings, train_pred_ratings = get_ratings(train_preds)
    train_rmse, train_mape = get_errors(train_preds)
    if verbose:
        print('-'*15)
        print('Train Data')
        print('-'*15)
        print("RMSE : {}\n\nMAPE : {}\n".format(train_rmse, train_mape))
    if verbose:
        print('adding train results in the dictionary..')
    train['rmse'] = train_rmse
    train['mape'] = train_mape
    train['predictions'] = train_pred_ratings

    print('\nEvaluating for test data...')
    test_preds = algo.test(testset)
    test_actual_ratings, test_pred_ratings = get_ratings(test_preds)
    test_rmse, test_mape = get_errors(test_preds)
    if verbose:
        print('-'*15)
        print('Test Data')
        print('-'*15)
        print("RMSE : {}\n\nMAPE : {}\n".format(test_rmse, test_mape))
        print('storing the test results in test dictionary...')
    test['rmse'] = test_rmse
    test['mape'] = test_mape
    test['predictions'] = test_pred_ratings
    
    return train, test

#######################################XGBoost

import xgboost as xgb

x_train = reg_train_sample.drop(['user','game','rating'], axis=1)
y_train = reg_train_sample['rating']
x_test = reg_test_sample.drop(['user','game','rating'], axis=1)
y_test = reg_test_sample['rating']
'''
params={'n_estimators':[50,70,90,110,120,130,140,150,160,170,180,190,200]}
first_xgb = xgb.XGBRegressor(silent=False, n_jobs=13, random_state=15)
gs_13=GridSearchCV(first_xgb,param_grid=params)
gs_13.fit(x_train,y_train)
train_results, test_results = run_xgboost(gs_13.best_estimator_, x_train, y_train, x_test, y_test)
'''
first_xgb = xgb.XGBRegressor(silent=False, n_jobs=13, random_state=15,n_estimators = 30)
train_results, test_results = run_xgboost(first_xgb , x_train, y_train, x_test, y_test)
xgb.plot_importance(first_xgb)
plt.show()
# store the results in models_evaluations dictionaries
models_evaluation_train = dict()
models_evaluation_test = dict()
models_evaluation_train['first_algo'] = train_results
models_evaluation_test['first_algo'] = test_results

#################################suprise baseline only
from surprise import BaselineOnly 
from surprise.model_selection import GridSearchCV as GridSearch
from surprise.model_selection import cross_validate

'''bsl_options = {'method': 'sgd',
               'learning_rate': .001
               }'''
'''
params={'bsl_options':{'learning_rate':[1.0,0.1,0.01,0.001,0.0001],'method':['sgd']}}
bsl_algo = BaselineOnly()
gs_bsl=GridSearch(bsl_algo,param_grid=params,measures=['RMSE'],cv=10)
gs_bsl.fit(train_data)
'''
bsl_options = {'learning_rate':0.003,'method':'sgd'}
bsl_algo = BaselineOnly(bsl_options = bsl_options)

bsl_train_results, bsl_test_results = run_surprise(bsl_algo, trainset, testset, verbose=True)

models_evaluation_train['bsl_algo'] = bsl_train_results 
models_evaluation_test['bsl_algo'] = bsl_test_results

###############################################KNNmodel
from surprise import KNNBaseline

sim_options = {'user_based' : True,
               'name': 'cosine',
              } 
bsl_options = {'method': 'sgd'}
'''
params={'k':[5,10,15,20,25,30,35,40,45,50,55,60]}
knn_bsl_u = KNNBaseline(sim_options = sim_options, bsl_options = bsl_options)
gs_knn_u=GridSearch(knn_bsl_u,param_grid=params,measures=['RMSE'],cv=5)
gs_knn_u.fit(train_data)
'''
knn_bsl_u = KNNBaseline(sim_options = sim_options, bsl_options = bsl_options, k = 5)
knn_bsl_u_train_results, knn_bsl_u_test_results = run_surprise(knn_bsl_u, trainset, testset, verbose=True)

models_evaluation_train['knn_bsl_u'] = knn_bsl_u_train_results
models_evaluation_test['knn_bsl_u'] = knn_bsl_u_test_results


sim_options = {'user_based' : False,
               'name': 'cosine',
              } 

bsl_options = {'method': 'sgd'}
'''
params={'k':[5,10,15,20,25,30,35,40,45,50,55,60]}
knn_bsl_u = KNNBaseline(sim_options = sim_options, bsl_options = bsl_options)
gs_knn_u=GridSearch(knn_bsl_u,param_grid=params,measures=['RMSE'],cv=5)
gs_knn_u.fit(train_data)
'''
knn_bsl_g = KNNBaseline(sim_options = sim_options, bsl_options = bsl_options, k = 5)
knn_bsl_g_train_results, knn_bsl_g_test_results = run_surprise(knn_bsl_u, trainset, testset, verbose=True)

models_evaluation_train['knn_bsl_g'] = knn_bsl_g_train_results
models_evaluation_test['knn_bsl_g'] = knn_bsl_g_test_results



################################################SVD
from surprise import SVD

params ={'n_factors':[10,20,30,40,50,60,70,80,90,100,120,140,160,180,200]}
svd = SVD(biased=True, random_state=15, verbose=True,n_factors = 50)
#gs_svm=GridSearch(svd,param_grid=params,measures=['RMSE'],cv=10)
#gs_svm.fit(train_data)
svd_train_results, svd_test_results = run_surprise(svd, trainset, testset, verbose=True)

# Just store these error metrics in our models_evaluation datastructure
models_evaluation_train['svd'] = svd_train_results 
models_evaluation_test['svd'] = svd_test_results

######################################################SVDpp
from surprise import SVDpp

#params ={'n_factors':[10,20,30,40,50,60,70,80,90,100,120,140,160,180,200]
svdpp = SVDpp(random_state=15, verbose=True,n_factors=80)
#gs_svmpp=GridSearch(svd,param_grid=params,measure=['RMSE'],cv=10)
#gs_svmpp.fit(train_data)
svdpp_train_results, svdpp_test_results = run_surprise(svdpp, trainset, testset, verbose=True)

# Just store these error metrics in our models_evaluation datastructure
models_evaluation_train['svdpp'] = svdpp_train_results 
models_evaluation_test['svdpp'] = svdpp_test_results

################XGBoost with initial 13 features + Surprise Baseline predictor
reg_train_sample['bslpr'] = models_evaluation_train['bsl_algo']['predictions']
reg_test_sample['bslpr']  = models_evaluation_test['bsl_algo']['predictions']

x_train = reg_train_sample.drop(['user','game','rating'], axis=1)
y_train = reg_train_sample['rating']

x_test = reg_test_sample.drop(['user','game','rating'], axis=1)
y_test = reg_test_sample['rating']

xgb_bsl = xgb.XGBRegressor(silent=False, n_jobs=13, random_state=15,n_estimators = 32)
train_results, test_results = run_xgboost(xgb_bsl , x_train, y_train, x_test, y_test)

xgb.plot_importance(xgb_bsl)
plt.show()
models_evaluation_train['xgb_bsl'] = train_results
models_evaluation_test['xgb_bsl'] = test_results


models = pd.DataFrame(models_evaluation_test)
models.loc['rmse'].sort_values()
models_train = pd.DataFrame(models_evaluation_train)
models_train.loc['rmse'].sort_values()




#########################XGBoost with initial 13 features + Surprise Baseline predictor + KNNBaseline predictor
reg_train_sample['knn_bsl_u'] = models_evaluation_train['knn_bsl_u']['predictions']
reg_train_sample['knn_bsl_g'] = models_evaluation_train['knn_bsl_g']['predictions']

reg_test_sample['knn_bsl_u'] = models_evaluation_test['knn_bsl_u']['predictions']
reg_test_sample['knn_bsl_g'] = models_evaluation_test['knn_bsl_g']['predictions']

# prepare the train data....
x_train = reg_train_sample.drop(['user','game','rating'], axis=1)
y_train = reg_train_sample['rating']

x_test = reg_test_sample.drop(['user','game','rating'], axis=1)
y_test = reg_test_sample['rating']

xgb_bsl_knn = xgb.XGBRegressor(silent=False, n_jobs=13, random_state=15,n_estimators = 32)
train_results, test_results = run_xgboost(xgb_bsl_knn , x_train, y_train, x_test, y_test)

xgb.plot_importance(xgb_bsl_knn)
plt.show()

# store the results in models_evaluations dictionaries
models_evaluation_train['xgb_knn_bsl'] = train_results
models_evaluation_test['xgb_knn_bsl'] = test_results

###############XgBoost with 13 features + Surprise Baseline + Surprise KNNbaseline
reg_train_sample['svd'] = models_evaluation_train['svd']['predictions']
reg_train_sample['svdpp'] = models_evaluation_train['svdpp']['predictions']

reg_test_sample['svd'] = models_evaluation_test['svd']['predictions']
reg_test_sample['svdpp'] = models_evaluation_test['svdpp']['predictions']

x_train = reg_train_sample.drop(['user','game','rating'], axis=1)
y_train = reg_train_sample['rating']

x_test = reg_test_sample.drop(['user','game','rating'], axis=1)
y_test = reg_test_sample['rating']

xgb_final = xgb.XGBRegressor(n_jobs=10, random_state=15,n_estimators = 35)
train_results, test_results = run_xgboost(xgb_final, x_train, y_train, x_test, y_test)

# store the results in models_evaluations dictionaries
models_evaluation_train['xgb_final'] = train_results
models_evaluation_test['xgb_final'] = test_results

###########################all lebal

x_train = reg_train_sample[['knn_bsl_u', 'knn_bsl_g', 'svd', 'svdpp']]
y_train = reg_train_sample['rating']

# test data
x_test = reg_test_sample[['knn_bsl_u', 'knn_bsl_g', 'svd', 'svdpp']]
y_test = reg_test_sample['rating']


xgb_all_models = xgb.XGBRegressor(n_jobs=10, random_state=15,n_estimators = 32)
train_results, test_results = run_xgboost(xgb_all_models, x_train, y_train, x_test, y_test)

# store the results in models_evaluations dictionaries
models_evaluation_train['xgb_all_models'] = train_results
models_evaluation_test['xgb_all_models'] = test_results

xgb.plot_importance(xgb_all_models)
plt.show()

















