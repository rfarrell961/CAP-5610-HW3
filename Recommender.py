import pandas as pd
from surprise import Reader
from surprise import Dataset
from surprise.model_selection import cross_validate
from surprise import KNNBasic
from surprise import SVD
from surprise.accuracy import rmse
from surprise import accuracy
from surprise.model_selection import train_test_split
import matplotlib.pyplot as plt




df = pd.read_csv("movie_data/ratings_small.csv")
keywordDf = pd.read_csv("movie_data/keywords.csv")
keywordDf.sort_values('id', axis=0, ascending=True, inplace=True, kind='quicksort', na_position='last')
reader = Reader(sep=',')
data = Dataset.load_from_df(df[['userId', 'movieId', 'rating']], reader)

"""
benchmark = []
algorithms = [SVD(), KNNBasic(sim_options={'user_based':True}), KNNBasic(sim_options={'user_based':False})]
for algorithm in algorithms:
    results = cross_validate(algorithm, data, cv=5, verbose=False)

    tmp = pd.DataFrame.from_dict(results).mean(axis=0)
    tmp = pd.concat([tmp, pd.Series([str(algorithm).split(' ')[0].split('.')[-1]], index=['Algorithm'])])
    benchmark.append(tmp)

df2 = pd.DataFrame(benchmark).set_index('Algorithm')
print(df2)
"""
trainset, testset = train_test_split(data, test_size=0.25)

"""
##### PMF #####
print("PMF:")
predictions = SVD().fit(trainset).test(testset)
accuracy.rmse(predictions)
accuracy.mae(predictions)

##### User Based Collaborative Filtering #####
print("User Based Collaborative Filtering:")
predictions = KNNBasic(sim_options={'user_based':True}).fit(trainset).test(testset)
accuracy.rmse(predictions)
accuracy.mae(predictions)

##### Item Based Collaborative Filtering #####
print("User Based Collaborative Filtering:")
predictions = KNNBasic(sim_options={'user_based':False}).fit(trainset).test(testset)
accuracy.rmse(predictions)
accuracy.mae(predictions)


##### User Based Collaborative Filtering Pearson#####
print("User Based Collaborative Filtering Pearson:")
predictions = KNNBasic(sim_options={'name': 'Pearson', 'user_based':True}).fit(trainset).test(testset)
accuracy.rmse(predictions)
accuracy.mae(predictions)

##### User Based Collaborative Filtering MSD#####
print("User Based Collaborative Filtering MSD:")
predictions = KNNBasic(sim_options={'name': 'MSD', 'user_based':True}).fit(trainset).test(testset)
accuracy.rmse(predictions)
accuracy.mae(predictions)

##### User Based Collaborative Filtering Cosine#####
print("User Based Collaborative Filtering Cosine:")
predictions = KNNBasic(sim_options={'name': 'cosine', 'user_based':True}).fit(trainset).test(testset)
accuracy.rmse(predictions)
accuracy.mae(predictions)

##### Item Based Collaborative Filtering Pearson#####
print("User Based Collaborative Filtering Pearson:")
predictions = KNNBasic(sim_options={'name': 'Pearson','user_based':False}).fit(trainset).test(testset)
accuracy.rmse(predictions)
accuracy.mae(predictions)

##### Item Based Collaborative Filtering MSD#####
print("User Based Collaborative Filtering MSD:")
predictions = KNNBasic(sim_options={'name': 'MSD','user_based':False}).fit(trainset).test(testset)
accuracy.rmse(predictions)
accuracy.mae(predictions)

##### Item Based Collaborative Filtering Cosine#####
print("User Based Collaborative Filtering Cosine:")
predictions = KNNBasic(sim_options={'name': 'cosine', 'user_based':True}).fit(trainset).test(testset)
accuracy.rmse(predictions)
accuracy.mae(predictions)
"""

minNumberNeighbors = 10
maxNumberNeighbors = 200

df_user = pd.DataFrame(columns=['K', 'RMSE'])
df_item = pd.DataFrame(columns=['K', 'RMSE'])

for k in range(minNumberNeighbors, maxNumberNeighbors, 10):
    ##### User Based Collaborative Filtering Cosine#####
    predictions = KNNBasic(k=k, sim_options={'name': 'Pearson', 'user_based': True}).fit(trainset).test(testset)
    RMSE = round(accuracy.rmse(predictions), 4)
    dict = {'K': [k],
            'Type': ['User'],
            'RMSE': [RMSE]}
    df2 = pd.DataFrame(dict)
    df_user = pd.concat([df_user, df2], ignore_index=True)

    ##### Item Based Collaborative Filtering Pearson#####
    predictions = KNNBasic(k=k, sim_options={'name': 'Pearson', 'user_based': False}).fit(trainset).test(testset)
    RMSE = round(accuracy.rmse(predictions), 4)
    dict = {'K': [k],
            'Type': ['Item'],
            'RMSE': [RMSE]}
    df2 = pd.DataFrame(dict)
    df_item = pd.concat([df_item, df2], ignore_index=True)

print(df_user)
print(df_item)
plt.plot(df_user['K'], df_user['RMSE'], label='User Based')
plt.plot(df_item['K'], df_item['RMSE'], label='Item Based')
plt.legend()
plt.show()


