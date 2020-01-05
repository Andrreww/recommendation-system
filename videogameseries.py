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
import math

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
user_game_action = zero_matrix.copy()
# Fill the confidence with (hours played)
# Added 1 to the hours played so that we have min. confidence for games bought but not played.
user_game_interactions[user_idx, game_idx] = rating
user_game_action[user_idx, game_idx] = 1

##########################################################################序列性
user_relation = np.zeros(shape = (n_games, n_games))
user_relation_meantime = np.zeros(shape = (n_games, n_games))
user_relation_count = np.zeros(shape = (n_games, n_games)) 
games_count = np.zeros(n_games)
game_relation_percent = np.zeros(shape = (n_games, n_games)) 
for i in range(n_games):
    for u in range(n_users):
        if user_game_action[u][i] != 1:
            continue
        games_count[i] += 1
        for j in range(n_games):
            if (user_game_action[u][j] == 1) & (user_game_time[u][j] > user_game_time[u][i]):
                temp = (datetime.utcfromtimestamp(user_game_time[u][j]) - datetime.utcfromtimestamp(user_game_time[u][i])).days
                if temp >= 365:
                    continue
                user_relation_meantime[i][j] += temp * math.exp(-0.0126*temp)
                user_relation[i][j] += math.exp(-0.0126*temp)
                user_relation_count[i][j] += 1
for i in range(n_games):
    for j in range(n_games):
        game_relation_percent[i][j] = user_relation_count[i][j]/games_count[i]          
for i in range(n_games):
    for j in range(n_games):
        if user_relation[i][j] != 0:
            user_relation_meantime[i][j] /= user_relation[i][j]

temp2 = user_relation_count.reshape(-1)
temp3 = temp2[temp2>1]
plt.hist(temp3)
sns.heatmap(user_relation_count)

user_relation_count_abs = np.zeros(shape = (n_games, n_games))
for i in range(n_games):
    for j in range(n_games):
        user_relation_count_abs[i][j] = abs(user_relation_count[i][j] - user_relation_count[j][i])
sns.heatmap(user_relation_count_abs)

np.max(game_relation_percent)
mg,ng = (np.argmax(game_relation_percent)/n_games,np.argmax(game_relation_percent)%n_games)


######################################################################################
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
        similar_toys[i] = sim_toys[:20]
    return similar_toys

Similar_toys = ItemmostSimilarity(Similaritym)
Similar_toys_normal = ItemmostSimilarity(Similaritynormal)

'''
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
'''
def Recommendation(similar_toys, Similaritymatrix,u, i,user_relation_meantime,gamma):
    s1 = 0
    s2 = 0
    for j in similar_toys[i]:
        if user_game_interactions[u][j] == 0:
            continue
#        if (datetime.utcfromtimestamp(t0) - datetime.utcfromtimestamp(user_game_time[u][j])) < 0:
#            continue
        s1 += Similaritymatrix[i][j]*user_game_interactions[u][j]
        s2 += Similaritymatrix[i][j]
    if s2 == 0:
        return 0
    sum = s1/s2
    for j in range(n_games):
        if (user_game_action[u][j]==1)&(user_relation_meantime[j][i]>1):
            sum *=(1+gamma/user_relation_meantime[j][i])
    return sum

t0 = int(time.time())
def Predict(n_users,n_games,Similaritym,user_relation_meantime,Similar_toys):
    user_game_pre = zero_matrix.copy()
    for u in range(n_users):
        for i in range(n_games):
            if user_game_interactions[u][i]==0:
                continue
            user_game_pre[u][i] = Recommendation(Similar_toys, Similaritym,u, i,user_relation_meantime,0.15)
    return user_game_pre

pre_matrix = Predict(n_users,n_games,Similaritynormal,user_relation_meantime,Similar_toys_normal)

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



1.24





m,n = (np.argmax(pre_matrix)/n_games,np.argmax(pre_matrix)%n_games)
pre_matrix[int(m)][n]

x = []
x2 =[]
y = []
y2 =[]
for para in range(0,100,5):
    x.append(para/100)
    temp = Recommendation(Similar_toys, Similaritym,int(m), n,user_relation_meantime,para/100)
    y.append(temp)
    temp = Recommendation(Similar_toys, Similaritym,22, 7,user_relation_meantime,para/100)
    y2.append(temp)
plt.plot(x,y,c = 'r')
plt.plot(x,y2,c = 'b')
plt.title("gamma")
plt.show()






