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
df = pd.read_csv('C:/Users/lenovo/Desktop/ratings_Video_Games.csv', sep=',', 
                       names=['user','game','rating','datestamp','date'])
df.head()
df = df.drop(['datestamp'],axis=1)
n_users = len(df.user.unique())
n_games = len(df.game.unique())

sparsity = df.shape[0] / float(n_users * n_games)
print('{:.5%} of the user-item matrix is filled'.format(sparsity))

vc = df.user.value_counts()

k = pd.DataFrame({'user':vc.index,'count2':vc.values})

df_final = pd.merge(df,k,on='user')

df_final2 = df_final[df_final.count2>=8]

len(df_final2.user.unique())

Goodgame = df_final2.game.value_counts().head(1000).index

selectgame = random.sample(list(Goodgame),100)

train_df = df_final2[df_final2.game.isin(selectgame)]

#train_df = df.sample(frac=0.04,weights=None, random_state=None, axis=0)
#train_df.date = pd.to_datetime(train_df.date)

n_users = len(train_df.user.unique())
n_games = len(train_df.game.unique())

user2idx = {user: i for i, user in enumerate(train_df.user.unique())}
idx2user = {i: user for user, i in user2idx.items()}

game2idx = {game: i for i, game in enumerate(train_df.game.unique())}
idx2game = {i: game for game, i in game2idx.items()}

# Convert the user and games to idx
user_idx = train_df['user'].apply(lambda x: user2idx[x]).values
game_idx = train_df['game'].apply(lambda x: game2idx[x]).values
rating = train_df['rating'].values
timing = train_df['datestamp'].values

zero_matrix = np.zeros(shape = (n_users, n_games)) # Create a zero matrix
user_game_time = zero_matrix.copy()
user_game_time[user_idx, game_idx] = timing # Fill the matrix will preferences (bought)
user_game_interactions = zero_matrix.copy()
user_game_pre = zero_matrix.copy()
# Fill the confidence with (hours played)
# Added 1 to the hours played so that we have min. confidence for games bought but not played.
user_game_interactions[user_idx, game_idx] = rating

(datetime.utcfromtimestamp(user_game_time[0][0]) - datetime.utcfromtimestamp(user_game_time[1][0])).days


import math

def ItemSimilarity(n_users,n_toys, alpha):
	#calculate co-rated users between items
    C = np.zeros((n_toys,n_toys))
    N = np.zeros(n_toys)
    W = np.zeros((n_toys,n_toys))
    for i in range(n_toys):
        for j in range(n_toys):
            if i == j:
                continue
            for u in range(n_users):
                C[i][j] += user_game_interactions[u][i]*user_game_interactions[u][j]*(1 / (1 + alpha * abs((datetime.utcfromtimestamp(user_game_time[u][i]) - datetime.utcfromtimestamp(user_game_time[u][j])).days)))
	#calculate finial similarity matrix W
    for i in range(n_toys):
        N[i] = sum([ s*s for s in user_game_interactions[:,i]])
    for i in range(n_toys):
        for j in range(n_toys):
            W[i][j] = C[i][j] / math.sqrt(N[i] * N[j])
    return W
import matplotlib.pyplot as plt
import seaborn as sns

Similaritym = ItemSimilarity(n_users,n_games,0.001)
np.max(Similaritym)


Similaritynormal = ItemSimilarity(n_users,n_games,0)
np.max(Similaritynormal)


def ItemmostSimilarity(Similaritymatrix):
    n_toys = Similaritymatrix.shape[0]
    similar_toys = dict()
    for i in range(n_toys):
    # get the top similar movies and store them in the dictionary
        sim_toys = Similaritymatrix[i].ravel().argsort()[::-1][1:]
        similar_toys[i] = sim_toys[:30]
    return similar_toys

Similar_toys = ItemmostSimilarity(Similaritym)
Similar_toys_normal = ItemmostSimilarity(Similaritynormal)


def Recommendation(similar_toys, Similaritymatrix,u, i, t0,alpha):
    s1 = 0
    s2 = 0
    for j in similar_toys[i]:
        if user_game_interactions[u][j] == 0:
            continue
#        if (datetime.utcfromtimestamp(t0) - datetime.utcfromtimestamp(user_game_time[u][j])) < 0:
#            continue
        s1 += Similaritymatrix[i][j]*user_game_interactions[u][j]*(1 / (1 + alpha * abs((datetime.utcfromtimestamp(t0) - datetime.utcfromtimestamp(user_game_time[u][j])).days)))
        s2 += Similaritymatrix[i][j]
    if s2 == 0:
        return 0
    return s1/s2

Recommendation(Similar_toys, Similaritym,8, 1, 1390089600,0)
t0 = int(time.time())
'''
def Predict(n_users,n_games,Similaritym):
    user_game_pre = zero_matrix.copy()
    for u in range(n_users):
        for i in range(n_games):
            if user_game_interactions[u][i]!=0:
                continue
            user_game_pre[u][i] = Recommendation(Similar_toys, Similaritym,u, i, t0,0)
    return user_game_pre

pre_matrix = Predict(n_users,n_games,Similaritym)
pre_matrix_normal = Predict(n_users,n_games,Similaritynormal)
'''
t0 = 1390089600
def Predict(n_users,n_games,Similaritym,alpha):
    user_game_pre = zero_matrix.copy()
    for u in range(n_users):
        for i in range(n_games):
            if user_game_interactions[u][i]==0:
                continue
            user_game_pre[u][i] = Recommendation(Similar_toys, Similaritym,u, i, user_game_time[u][i],alpha)
    return user_game_pre

pre_matrix = Predict(n_users,n_games,Similaritynormal,0)

def get_error_metrics(user_game_interactions, user_game_pre):
    rmse = 0
    count = 0
    for u in range(n_users):
        for i in range(n_games):
            if pre_matrix[u][i]==0:
                continue
            rmse += (user_game_interactions[u][i]-user_game_pre[u][i])**2
            count += 1
    return np.sqrt(rmse/count)

get_error_metrics(user_game_interactions,pre_matrix)




#########################################################
m,n = (np.argmax(pre_matrix)/n_games,np.argmax(pre_matrix)%n_games)
user_game_time[int(m)][n]
t0 = int(time.time())
tmin = np.min(train_df.datestamp)
time_recom = dict()

for ts in range(tmin,t0,15768000):
    time_recom[ts] = Recommendation(Similar_toys, Similaritym,int(m),n, ts,0.001)
#x_y = sorted(time_recom.items(),key = lambda item:item[0])
x = list(time_recom.keys())
xnew = []
for i in range(len(x)):
    xnew.append(datetime.utcfromtimestamp(x[i]))
xnewt = pd.to_datetime(xnew)
y = list(time_recom.values())


time_recom = dict()
for ts in range(tmin,t0,15768000):
    time_recom[ts] = Recommendation(Similar_toys, Similaritym,int(m),n, ts,0.005)
x2 = list(time_recom.keys())
xnew = []
for i in range(len(x2)):
    xnew.append(datetime.utcfromtimestamp(x2[i]))
xnewk = pd.to_datetime(xnew)
y2 = list(time_recom.values())

plt.plot(xnewt,y,c = 'r')
plt.plot(xnewk,y2,c = 'b')
plt.show()

fig = plt.figure()
ax1 = fig.add_subplot(2,1,1) # 画2行1列个图形的第1个
ax2 = fig.add_subplot(2,1,2) # 画2行1列个图形的第2个

ax1.plot(xnewt,y,c = 'r')
ax2.plot(xnewk,y2,c = 'b')
plt.show()











