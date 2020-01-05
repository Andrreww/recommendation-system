# -*- coding: utf-8 -*-
# this is just to know how much time will it take to run this entire ipython notebook 
from datetime import datetime
# globalstart = datetime.now()
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('nbagg')
import time
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})

import seaborn as sns
sns.set_style('whitegrid')
import os
from scipy import sparse
from scipy.sparse import csr_matrix

from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import random

print("creating the dataframe from data.csv file..")
df = pd.read_csv('C:/Users/lenovo/Desktop/ratings_Toys_and_Games.csv', sep=',', 
                       names=['user','toy','rating','datestamp','date'])
#print('Done.\n')
#df['date2'] = datetime.utcfromtimestamp(df['date'])
# we are arranging the ratings according to time.
#print('Sorting the dataframe by date..')
#df.sort_values(by='date', inplace=True)
#print('Done..')
df.head()
df.describe()['rating']
df.drop(['datestamp'],axis=1)
# just to make sure that all Nan containing rows are deleted..
print("No of Nan values in our dataframe : ", sum(df.isnull().any()))
dup_bool = df.duplicated(['toy','user','rating'])
dups = sum(dup_bool) # by considering all columns..( including timestamp)
print("There are {} duplicate rating entries in the data..".format(dups))

print("Total data ")
print("-"*50)
print("\nTotal no of ratings :",df.shape[0])
print("Total No of Users   :", len(np.unique(df.user)))
print("Total No of toys  :", len(np.unique(df.toy)))

if not os.path.isfile('train.csv'):
    # create the dataframe and store it in the disk for offline purposes..
    df.iloc[:int(df.shape[0]*0.80)].to_csv("train.csv", index=False)

if not os.path.isfile('test.csv'):
    # create the dataframe and store it in the disk for offline purposes..
    df.iloc[int(df.shape[0]*0.80):].to_csv("test.csv", index=False)

train_df = pd.read_csv("train.csv", parse_dates=['date'])
test_df = pd.read_csv("test.csv")

# method to make y-axis more readable
def human(num, units = 'M'):
    units = units.lower()
    num = float(num)
    if units == 'k':
        return str(num/10**3) + " K"
    elif units == 'm':
        return str(num/10**6) + " M"
    elif units == 'b':
        return str(num/10**9) +  " B"

train_df = df.iloc[:int(df.shape[0]*0.004)]
fig, ax = plt.subplots()
plt.title('Distribution of ratings over Training dataset', fontsize=15)
sns.countplot(train_df.rating)
ax.set_yticklabels([human(item, 'K') for item in ax.get_yticks()])
ax.set_ylabel('No. of Ratings(Millions)')
plt.show()

train_df = df.sample(frac=0.04,weights=None, random_state=None, axis=0).drop(['datestamp'],axis=1)
train_df.date = pd.to_datetime(train_df.date)
train_df['day_of_week'] = train_df.date.dt.weekday_name
train_df['day_of_week'].value_counts()

ax = train_df.resample('m', on='date')['rating'].count().plot()
ax.set_title('No of ratings per month (Training data)')
plt.xlabel('Months')
plt.ylabel('No of ratings(per month)')
ax.set_yticklabels([human(item, 'k') for item in ax.get_yticks()])
plt.show()

fig = plt.figure(figsize=plt.figaspect(.45))
sns.boxplot(y='rating', x='day_of_week', data=train_df)
plt.show()

fig , ax = plt.subplots()
plt.title('Distribution of days with respect to ratings')
sns.countplot(train_df.day_of_week)
ax.set_yticklabels([human(item, 'k') for item in ax.get_yticks()])
ax.set_ylabel('No. of Ratings')
plt.show()


#user
train_df = df.sample(frac=0.04,weights=None, random_state=None, axis=0).drop(['datestamp'],axis=1)
no_of_rated_toys_per_user = train_df.groupby(by='user')['rating'].count().sort_values(ascending=False)
no_of_rated_toys_per_user.head()
no_of_rated_toys_per_user.mean()
fig = plt.figure(figsize=plt.figaspect(.5))

ax1 = plt.subplot(121)
sns.kdeplot(no_of_rated_toys_per_user, shade=True, ax=ax1)
plt.xlabel('No of ratings by user')
plt.title("PDF")

ax2 = plt.subplot(122)
sns.kdeplot(no_of_rated_toys_per_user, shade=True, cumulative=True,ax=ax2)
plt.xlabel('No of ratings by user')
plt.title('CDF')

plt.show()

no_of_rated_toys_per_user.describe()
#toys
no_of_ratings_per_toy = train_df.groupby(by='toy')['rating'].count().sort_values(ascending=False)
no_of_ratings_per_toy.head()
fig = plt.figure(figsize=plt.figaspect(.5))
ax = plt.gca()
plt.plot(no_of_ratings_per_toy.values)
plt.title('# RATINGS per toy')
plt.xlabel('toy')
plt.ylabel('No of Users who rated a toy')
ax.set_xticklabels([])
plt.show()


#PDF's & CDF's of Avg.Ratings of Users & Movies (In Train Data)




#Creating sparse matrix from data frame 
train_df = df.iloc[:int(df.shape[0]*0.002)]
user2idx = {user: i for i, user in enumerate(train_df.user.unique())}
idx2user = {i: user for user, i in user2idx.items()}

toy2idx = {toy: i for i, toy in enumerate(train_df.toy.unique())}
idx2toy = {i: toy for toy, i in toy2idx.items()}

# Convert the user and games to idx
user_idx = train_df['user'].apply(lambda x: user2idx[x]).values
toy_idx = train_df['toy'].apply(lambda x: toy2idx[x]).values
rating = train_df['rating'].values

train_sparse_matrix = np.zeros(shape = (len(train_df.user.unique()), len(train_df.toy.unique()))) 
train_sparse_matrix[user_idx,toy_idx] = rating
train_matrix = pd.DataFrame(train_sparse_matrix)

'''
#Creating sparse matrix from data frame 
train_df = test_df
user2idx = {user: i for i, user in enumerate(train_df.user.unique())}
idx2user = {i: user for user, i in user2idx.items()}

toy2idx = {toy: i for i, toy in enumerate(train_df.toy.unique())}
idx2toy = {i: toy for toy, i in toy2idx.items()}

# Convert the user and games to idx
user_idx = train_df['user'].apply(lambda x: user2idx[x]).values
toy_idx = train_df['toy'].apply(lambda x: toy2idx[x]).values
rating = train_df['rating'].values

train_sparse_matrix = np.zeros(shape = (len(train_df.user.unique()), len(train_df.toy.unique()))) 
train_sparse_matrix[user_idx,toy_idx] = rating
'''
train_sparse_matrix = sparse.csr_matrix((rating, (user_idx,toy_idx)))
type(train_sparse_matrix)
def get_average_ratings(sparse_matrix, of_users):
    
    # average ratings of user/axes
    ax = 1 if of_users else 0 # 1 - User axes,0 - Movie axes

    # ".A1" is for converting Column_Matrix to 1-D numpy array 
    sum_of_ratings = sparse_matrix.sum(axis=ax).A1
    # Boolean matrix of ratings ( whether a user rated that movie or not)
    is_rated = sparse_matrix!=0
    # no of ratings that each user OR movie..
    no_of_ratings = is_rated.sum(axis=ax).A1
    
    # max_user  and max_movie ids in sparse matrix 
    u,m = sparse_matrix.shape
    # creae a dictonary of users and their average ratigns..
    average_ratings = { i : sum_of_ratings[i]/no_of_ratings[i]
                                 for i in range(u if of_users else m) 
                                    if no_of_ratings[i] !=0}

    # return that dictionary of average ratings
    return average_ratings

#Finding Global average of all movie ratings, Average rating per user, and Average rating per movie
train_averages = dict()
# get the global average of ratings in our train set.
train_global_average = train_sparse_matrix.sum()/train_sparse_matrix.count_nonzero()
train_averages['global'] = train_global_average

train_averages['user'] = get_average_ratings(train_sparse_matrix, of_users=True)
print('\nAverage rating of user 10 :',train_averages['user'][54])
'''
#Computing User-User Similarity matrix
from sklearn.metrics.pairwise import cosine_similarity


def compute_user_similarity(sparse_matrix, compute_for_few=False, top = 100, verbose=False, verb_for_n_rows = 20,
                            draw_time_taken=True):
    no_of_users, _ = sparse_matrix.shape
    # get the indices of  non zero rows(users) from our sparse matrix
    row_ind, col_ind = sparse_matrix.nonzero()
    row_ind = sorted(set(row_ind)) # we don't have to
    time_taken = list() #  time taken for finding similar users for an user..
    
    # we create rows, cols, and data lists.., which can be used to create sparse matrices
    rows, cols, data = list(), list(), list()
    if verbose: print("Computing top",top,"similarities for each user..")
    
    start = datetime.now()
    temp = 0
    
    for row in row_ind[:top] if compute_for_few else row_ind:
        temp = temp+1
        prev = datetime.now()
        
        # get the similarity row for this user with all other users
        sim = cosine_similarity(sparse_matrix.getrow(row), sparse_matrix).ravel()
        # We will get only the top ''top'' most similar users and ignore rest of them..
        top_sim_ind = sim.argsort()[-top:]
        top_sim_val = sim[top_sim_ind]
        
        # add them to our rows, cols and data
        rows.extend([row]*top)
        cols.extend(top_sim_ind)
        data.extend(top_sim_val)
        time_taken.append(datetime.now().timestamp() - prev.timestamp())
        if verbose:
            if temp%verb_for_n_rows == 0:
                print("computing done for {} users [  time elapsed : {}  ]"
                      .format(temp, datetime.now()-start))
            
        
    # lets create sparse matrix out of these and return it
    if verbose: print('Creating Sparse matrix from the computed similarities')
    #return rows, cols, data
    
    #if draw_time_taken:
    #    plt.plot(time_taken, label = 'time taken for each user')
    #    plt.plot(np.cumsum(time_taken), label='Total time')
    #    plt.legend(loc='best')
    #    plt.xlabel('User')
    #    plt.ylabel('Time (seconds)')
    #    plt.show()
        
    return sparse.csr_matrix((data, (rows, cols)), shape=(no_of_users, no_of_users)), time_taken     
'''
#omputing Movie-Movie Similarity matrix
from sklearn.metrics.pairwise import cosine_similarity
m_m_sim_sparse = cosine_similarity(X=train_sparse_matrix.T, dense_output=False)
#top similar movie
toy_ids = np.unique(m_m_sim_sparse.nonzero()[1])
similar_toys = dict()
for toy in toy_ids:
    # get the top similar movies and store them in the dictionary
    sim_toys = m_m_sim_sparse[toy].toarray().ravel().argsort()[::-1][1:]
    similar_toys[toy] = sim_toys[:5]
    
#########################################################################################################    
#Suprise BaselineModel
# it is just to makesure that all of our algorithms should produce same results
# everytime they run...

my_seed = 15
random.seed(my_seed)
np.random.seed(my_seed)

##########################################################
# get  (actual_list , predicted_list) ratings given list 
# of predictions (prediction is a class in Surprise).    
##########################################################
def get_ratings(predictions):
    actual = np.array([pred.r_ui for pred in predictions])
    pred = np.array([pred.est for pred in predictions])
    
    return actual, pred

################################################################
# get ''rmse'' and ''mape'' , given list of prediction objecs 
################################################################
def get_errors(predictions, print_them=False):

    actual, pred = get_ratings(predictions)
    rmse = np.sqrt(np.mean((pred - actual)**2))
    mape = np.mean(np.abs(pred - actual)/actual)

    return rmse, mape*100

##################################################################################
# It will return predicted ratings, rmse and mape of both train and test data   #
##################################################################################
def run_surprise(algo, trainset, testset, verbose=True): 
    '''
        return train_dict, test_dict
    
        It returns two dictionaries, one for train and the other is for test
        Each of them have 3 key-value pairs, which specify ''rmse'', ''mape'', and ''predicted ratings''.
    '''
    start = datetime.now()
    # dictionaries that stores metrics for train and test..
    train = dict()
    test = dict()
    
    # train the algorithm with the trainset
    st = datetime.now()
    print('Training the model...')
    algo.fit(trainset)
    print('Done. time taken : {} \n'.format(datetime.now()-st))
    
    # ---------------- Evaluating train data--------------------#
    st = datetime.now()
    print('Evaluating the model with train data..')
    # get the train predictions (list of prediction class inside Surprise)
    train_preds = algo.test(trainset.build_testset())
    # get predicted ratings from the train predictions..
    train_actual_ratings, train_pred_ratings = get_ratings(train_preds)
    # get ''rmse'' and ''mape'' from the train predictions.
    train_rmse, train_mape = get_errors(train_preds)
    print('time taken : {}'.format(datetime.now()-st))
    
    if verbose:
        print('-'*15)
        print('Train Data')
        print('-'*15)
        print("RMSE : {}\n\nMAPE : {}\n".format(train_rmse, train_mape))
    
    #store them in the train dictionary
    if verbose:
        print('adding train results in the dictionary..')
    train['rmse'] = train_rmse
    train['mape'] = train_mape
    train['predictions'] = train_pred_ratings
    
    #------------ Evaluating Test data---------------#
    st = datetime.now()
    print('\nEvaluating for test data...')
    # get the predictions( list of prediction classes) of test data
    test_preds = algo.test(testset)
    # get the predicted ratings from the list of predictions
    test_actual_ratings, test_pred_ratings = get_ratings(test_preds)
    # get error metrics from the predicted and actual ratings
    test_rmse, test_mape = get_errors(test_preds)
    print('time taken : {}'.format(datetime.now()-st))
    
    if verbose:
        print('-'*15)
        print('Test Data')
        print('-'*15)
        print("RMSE : {}\n\nMAPE : {}\n".format(test_rmse, test_mape))
    # store them in test dictionary
    if verbose:
        print('storing the test results in test dictionary...')
    test['rmse'] = test_rmse
    test['mape'] = test_mape
    test['predictions'] = test_pred_ratings
    
    print('\n'+'-'*45)
    print('Total time taken to run this algorithm :', datetime.now() - start)
    
    # return two dictionaries train and test
    return train, test

import surprise
from surprise import Reader, Dataset
# It is to specify how to read the dataframe.
# for our dataframe, we don't have to specify anything extra..
reader = Reader(rating_scale=(1,5))

# create the traindata from the dataframe...
train_data = Dataset.load_from_df(train_df[['user', 'toy', 'rating']], reader)
# build the trainset from traindata.., It is of dataset format from surprise library..
trainset = train_data.build_full_trainset()
testset = list(zip(train_df.user.values, train_df.toy.values, train_df.rating.values))

from surprise import BaselineOnly 
from surprise.model_selection import GridSearchCV
from surprise.model_selection import cross_validate
param_dict = {'bsl_options':{
                              'learning_rate':[1.0,0.1,0.01,0.001,0.0001],
                              'method':['sgd']
    
                            }}

gridSearchModel = GridSearchCV(BaselineOnly,param_grid=param_dict,measures=['RMSE','MAE'],cv=10)
#bsl_algo = BaselineOnly(bsl_options=bsl_options)
# run this algorithm.., It will return the train and test results..
gridSearchModel.fit(train_data)
bsl_train_results, bsl_test_results = run_surprise(gridSearchModel.best_estimator['mae'], trainset, testset, verbose=True)
#models_evaluation_train['bsl_algo'] = bsl_train_results 
#models_evaluation_test['bsl_algo'] = bsl_test_results
print('best estimators of models',gridSearchModel.best_params)
######################################################################################################
#Surprise KNNBaseline predictor
from surprise import KNNBaseline
'''
# we specify , how to compute similarities and what to consider with sim_options to our algorithm
sim_options = {'user_based' : True,
               'name': 'pearson_baseline',
               'shrinkage': 100,
               'min_support': 2
              } 
param_dict ={'sim_options':{'shrinkage':[100],
                            'min_support': [2]},
             'bsl_options':{'method':['sgd'],
                            
                           },
             'k':[2,4,5,6,7,10,20,30,40,50,60]
             }
                            
#we keep other parameters like regularization parameter and learning_rate as default values.
bsl_options = {'method': 'sgd'} 

gridSearchModel = GridSearchCV(KNNBaseline,param_grid=param_dict,measures=['RMSE','MAE'],cv=10)
#knn_bsl_u = KNNBaseline(k=40, sim_options = sim_options, bsl_options = bsl_options)
gridSearchModel.fit(train_data)
knn_bsl_u_train_results, knn_bsl_u_test_results = run_surprise(gridSearchModel.best_estimator['rmse'], trainset, testset, verbose=True)

# Just store these error metrics in our models_evaluation datastructure
models_evaluation_train['knn_bsl_u'] = knn_bsl_u_train_results 
models_evaluation_test['knn_bsl_u'] = knn_bsl_u_test_results
print('Best Estimator',gridSearchModel.best_params)
##
param_dict ={'sim_options':{'shrinkage':[100],
                            'min_support': [2]},
             'bsl_options':{'method':['sgd']                           
                           },
             'k':[60]
             }

knn = KNNBaseline(sim_options={'shrinkage':100,'min_support':2},bsl_options={'method':'sgd'},k=60)
knn_bsl_u_train_results, knn_bsl_u_test_results = run_surprise(knn,trainset, testset, verbose=True)

# Just store these error metrics in our models_evaluation datastructure
models_evaluation_train['knn_bsl_u'] = knn_bsl_u_train_results 
models_evaluation_test['knn_bsl_u'] = knn_bsl_u_test_results
'''
###
sim_options = {'user_based' : False,
               'name': 'pearson_baseline',
               'shrinkage': 100,
               'min_support': 2
              } 
param_dict ={'sim_options':{'shrinkage':[100],
                            'min_support': [2]},
             'bsl_options':{'method':['sgd'],
                            
                           },
             'k':[2,4,5,6,7,10,20,30,40,50,60]
             }

gridSearchModel = GridSearchCV(KNNBaseline,param_grid=param_dict,measures=['RMSE','MAE'],cv=10)
#knn_bsl_u = KNNBaseline(k=40, sim_options = sim_options, bsl_options = bsl_options)
gridSearchModel.fit(train_data)
knn_bsl_m_train_results, knn_bsl_m_test_results = run_surprise(gridSearchModel.best_estimator['rmse'], trainset, testset, verbose=True)

# Just store these error metrics in our models_evaluation datastructure
models_evaluation_train['knn_bsl_m'] = knn_bsl_m_train_results 
models_evaluation_test['knn_bsl_m'] = knn_bsl_m_test_results
print('Best Estimator',gridSearchModel.best_params)

knn = KNNBaseline(sim_options={'shrinkage':100,'min_support':2,'user_base':False},bsl_options={'method':'sgd'},k=60)
knn_bsl_m_train_results, knn_bsl_m_test_results = run_surprise(knn,trainset, testset, verbose=True)

# Just store these error metrics in our models_evaluation datastructure
models_evaluation_train['knn_bsl_m'] = knn_bsl_m_train_results 
models_evaluation_test['knn_bsl_m'] = knn_bsl_m_test_results

#Matrix Factorization
from surprise import SVD
param_dict ={'n_factors':[10,20,30,40,50,60,70,80,90,100,120,140,160,180,200]
    
}
gridSearchModel = GridSearchCV(SVD,param_grid=param_dict,measures=['RMSE','MAE'],cv=10)
gridSearchModel.fit(train_data)
#svd = SVD(n_factors=100, biased=True, random_state=15, verbose=True)
svd_train_results, svd_test_results = run_surprise(gridSearchModel.best_estimator['rmse'], trainset, testset, verbose=True)

# Just store these error metrics in our models_evaluation datastructure
models_evaluation_train['svd'] = svd_train_results 
models_evaluation_test['svd'] = svd_test_results
print('Best paramaters',gridSearchModel.best_params)

svd_train_results, svd_test_results = run_surprise(SVD(n_factors=20), trainset, testset, verbose=True)

# Just store these error metrics in our models_evaluation datastructure
models_evaluation_train['svd'] = svd_train_results 
models_evaluation_test['svd'] = svd_test_results


#########################################################################
#SVD Matrix Factorization with implicit feedback from user ( user rated movies ) 
from surprise import SVDpp
param_dict={'n_factors':[10,20,30,40,50,60,70,80,90,100,120,130,140,150]}

#svdpp = SVDpp(n_factors=50, random_state=15, verbose=True)
gridSearchModel = GridSearchCV(SVDpp,param_grid=param_dict,measures=['RMSE','MAE'],cv=10)
gridSearchModel.fit(train_data)
#svd = SVD(n_factors=100, biased=True, random_state=15, verbose=True)
svdpp_train_results , svdpp_test_results = run_surprise(gridSearchModel.best_estimator['rmse'], trainset, testset, verbose=True)


# Just store these error metrics in our models_evaluation datastructure
models_evaluation_train['svdpp'] = svdpp_train_results 
models_evaluation_test['svdpp'] = svdpp_test_results
print(grinSearchModel.best_params)

svdpp_train_results , svdpp_test_results = run_surprise(SVDpp(n_factors=30), trainset, testset, verbose=True)


# Just store these error metrics in our models_evaluation datastructure
models_evaluation_train['svdpp'] = svdpp_train_results 
models_evaluation_test['svdpp'] = svdpp_test_results











