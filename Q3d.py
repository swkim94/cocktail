import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import cluster
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from sklearn import decomposition
from mpl_toolkits.mplot3d import Axes3D
import requests

# interested whether they would be grouped into 5 neighbors, corresponding to five basic tastes.
#
# using API for http://www.thecocktaildb.com/

# # load first cocktail data and initialize dataframe
# url = 'http://www.thecocktaildb.com/api/json/v1/1/lookup.php?i=15112'
# r = requests.get(url)
# json_data = r.json()
# key = list(json_data.keys())[0]  # only one top level key
# dict = json_data[key][0]
# df = pd.DataFrame.from_dict(dict,orient='index').T
# print(dict.keys())
#
# # get more cocktail data and add to dataframe.
# # url = 'http://www.thecocktaildb.com/api/json/v1/1/random.php'
# # url = 'http://www.thecocktaildb.com/api/json/v1/1/lookup.php?i=15112'
# # print('downloading data from '+ url)
# # n_random = 1000
# print ('estimated time for scanning 7000 ids = 7000/100 * 18s = 1260 seconds = 21 minutes')
# input()
# j = 0
# for i in range(11000,18000):
# 	a=format(i,'04d')
# 	url = 'http://www.thecocktaildb.com/api/json/v1/1/lookup.php?i='+a
# 	print('downloading data from ' + url)
# 	r = requests.get(url)
# 	json_data = r.json()
# 	if json_data[key] != None:
# 		key = list(json_data.keys())[0]
# 		dict = json_data[key][0]
# 		df_temp = pd.DataFrame.from_dict(dict,orient='index').T
# 		df = df.append(df_temp, ignore_index=True)
# 		j += 1
# print('finished downloading %i records of cocktails' % j)
# # scanning 100 ids takes 18 seconds,
#
#
# # remove duplicates
# df_temp = pd.DataFrame.drop_duplicates(df, subset = 'idDrink')
# df_temp.sort('idDrink')
# df = df_temp
# # df = df_temp.reset_index(inplace=True)
# print('removed duplicate records. now %i records of cocktails' % df.shape[0])
#
# print(df.keys())
# # Index(['strMeasure6', 'strMeasure9', 'strMeasure4', 'strIngredient5',
# #        'strDrink', 'strInstructions', 'strIngredient1', 'strIngredient11',
# #        'strMeasure13', 'strIngredient3', 'strMeasure5', 'idDrink',
# #        'strMeasure14', 'strMeasure15', 'strMeasure8', 'strIngredient14',
# #        'strIngredient15', 'strIngredient10', 'strMeasure12', 'strAlcoholic',
# #        'strMeasure7', 'strMeasure11', 'strMeasure3', 'strIngredient6',
# #        'strMeasure2', 'strIngredient4', 'strIngredient12', 'strGlass',
# #        'strIngredient9', 'strIngredient7', 'strDrinkThumb', 'strMeasure10',
# #        'strCategory', 'strIngredient2', 'strMeasure1', 'dateModified',
# #        'strIngredient13', 'strIngredient8'],
# #       dtype='object')
#
# df2 = df[['idDrink','strDrink','strCategory','strGlass','strAlcoholic',
# 		  'strIngredient1', 'strMeasure1',
# 		  'strIngredient2', 'strMeasure2',
# 		  'strIngredient3', 'strMeasure3',
# 		  'strIngredient4', 'strMeasure4',
# 		  'strIngredient5', 'strMeasure5',
# 		  'strIngredient6', 'strMeasure6',
# 		  'strIngredient7', 'strMeasure7',
# 		  'strIngredient8', 'strMeasure8',
# 		  'strIngredient9', 'strMeasure9',
# 		  'strIngredient10', 'strMeasure10',
# 		  'strIngredient11', 'strMeasure11',
# 		  'strIngredient12', 'strMeasure12',
# 		  'strIngredient13', 'strMeasure13',
# 		  'strIngredient14', 'strMeasure14',
# 		  'strIngredient15', 'strMeasure15',
# 		  'strInstructions','strDrinkThumb']]
#
# df2.to_pickle('cocktails2.pkl')
#
df = pd.read_pickle('cocktails2.pkl')
df2 = df.set_index('idDrink',drop =True)
df2 = df2.sort_index()
# df_temp = df2[['strIngredient'+str(i) for i in range(1,16)]]
# df_temp2 = df2[['strMeasure'+str(i) for i in range(1,16)]]
# ingredients = pd.Series(pd.unique(df_temp.values.ravel()))
# ingredients = ingredients.dropna()
# n_dim = ingredients.size
# measures = pd.Series(pd.unique(df_temp2.values.ravel()))
# measures = measures.dropna()
# df_temp3 = df_temp2['strMeasure1'].str.extract('[ab]?(\d)', expand=True).fillna(0)
# df_temp3.columns=['strMeasure1']
# for i in range(2,16):
# 	name = 'strMeasure'+str(i)
# 	df_temp3[name] = df_temp2[name].str.extract('[ab]?(\d)', expand=True).fillna(0)
# df_temp3 = df_temp3.astype(dtype=int)
# print('number of possible ingredients = %i' % n_dim)
# print('Each cocktail will be vectorized and embedded into a %i-dimensional vector space.' % n_dim)
# print('later plan to add amplitude for each vector dimension using the quantity of ingredient.')
# print('vectorizing...')
# array = np.zeros(n_dim)
# for i, row in df_temp.iterrows():
# 	print(i)
# 	if i == df_temp.first_valid_index():
# 		for j, item in enumerate(ingredients):
# 			array[j] = df_temp3.loc[i][(row == item).values].sum()
# 		df3 = pd.DataFrame(array, columns=[i], index=ingredients).T
# 	else:
# 		for j, item in enumerate(ingredients):
# 			array[j] = df_temp3.loc[i][(row == item).values].sum()
# 		df_temp = pd.DataFrame(array, columns=[i], index=ingredients).T
# 		df3 = df3.append(df_temp)
# df3.to_pickle('cocktails4.pkl')
# print('finished vectorizing cocktails and saved them into a local file')

df3 = pd.read_pickle('cocktails4.pkl')

input_str = 'Cosmopolitan Martini'
print('plotting ingredient spectrum of '+input_str)
id = df2[df2['strDrink']==input_str].index[0]
fig1 = plt.figure(figsize=(30, 9),tight_layout=True)
df3.loc[id].plot(kind='bar', x='ingredients', y='volume',title='ingredient spectrum of '+input_str)
plt.savefig('cocktail_fingerprint_'+input_str+'.png')
plt.show()

input_str = 'Cape Codder'
print('plotting ingredient spectrum of '+input_str)
id = df2[df2['strDrink']==input_str].index[0]
fig2 = plt.figure(figsize=(30, 9),tight_layout=True)
df3.loc[id].plot(kind='bar', x='ingredients', y='volume',title='ingredient spectrum of '+input_str)
plt.savefig('cocktail_fingerprint_'+input_str+'.png')
plt.show()

print('plotting histogram of ingredients from all cocktails')
fig3 = plt.figure(figsize=(30, 9),tight_layout=True)
df_temp = df3.sum()
df_temp.plot(kind='bar', x='ingredients', y='volume',title='histogram of ingredients from all cocktails')
plt.savefig('cocktail_histogram.png')
plt.show()

print('plotting top 10 ingredients')
fig4 = plt.figure(figsize=(20, 15),tight_layout=True)
df_temp = df3.sum().sort_values(ascending=False)
df_temp.head(10).plot(kind='bar', x='ingredients', y='volume',title='top 10 ingredients')
plt.savefig('cocktail_top10_ingredients.png')
plt.show()

X = df3.as_matrix()
nbrs = NearestNeighbors(n_neighbors=5, algorithm='auto').fit(X)
distances, indices = nbrs.kneighbors(X)

group = np.ones(X.shape[0])
i = 0
for index in indices:
	group[i] = np.argmax(index)
	i += 1

df4 = df3.copy()
df4['group'] = group
df4 = df4.groupby('group')

print('plotting profiles of five groups')
fig5 = plt.figure(figsize=(30, 20))
plt.title('profiles of five groups')
for i in range(5):
	plt.subplot(5, 1, i+1)
	df_temp = df4.get_group(i)
	df_temp = df_temp.drop('group', 1)
	df_temp = df_temp.sum()
	df_temp.plot.bar()

plt.savefig('cocktail_five_groups.png')
plt.show()

input_str = 'Cosmopolitan Martini'
df5 = df3.copy()
df5[df5.columns] = normalize(df3)
df6 = df5.dot(df5.T)
np.fill_diagonal(df6.values,0)
id = df2[df2['strDrink']==input_str].index[0]
id_recommendation = df6[id].argmax()
score =  df6[id].max()
recommendation = df2.loc[id_recommendation]['strDrink']
instruction = df2.loc[id_recommendation]['strInstructions']
print('For your input: '+input_str)
print('We recommend '+recommendation)
print('similarity score = %f' % score)
print(instruction)
for i in range(1,16):
	print(df2.loc[id_recommendation][['strIngredient'+str(i), 'strMeasure'+str(i)]].values)

input_str = 'Mojito'
df5 = df3.copy()
df5[df5.columns] = normalize(df3)
df6 = df5.dot(df5.T)
np.fill_diagonal(df6.values,0)
id = df2[df2['strDrink']==input_str].index[0]
id_recommendation = df6[id].argmax()
score =  df6[id].max()
recommendation = df2.loc[id_recommendation]['strDrink']
instruction = df2.loc[id_recommendation]['strInstructions']
print('For your input: '+input_str)
print('We recommend '+recommendation)
print('similarity score = %f' % score)
print(instruction)
for i in range(1,16):
	print(df2.loc[id_recommendation][['strIngredient'+str(i), 'strMeasure'+str(i)]].values)

X = df5.as_matrix()
pca = decomposition.PCA(n_components=3)
pca.fit(X)
X = pca.transform(X)
fig6 = plt.figure(figsize=(12, 10))
ax = Axes3D(fig6, rect=[0, 0, .95, 1], elev=48, azim=134)
ax.scatter(X[:, 0], X[:, 1], X[:, 2], cmap=plt.cm.spectral)

ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])

plt.savefig('cocktail_PCA.png')
plt.show()

spectral = cluster.SpectralClustering(n_clusters=5, eigen_solver='arpack', affinity="nearest_neighbors")
spectral.fit(X)
if hasattr(spectral, 'labels_'):
	group = spectral.labels_.astype(np.int)
else:
	group = spectral.predict(X)

fig7 = plt.figure(figsize=(12, 10))
ax = Axes3D(fig7, rect=[0, 0, .95, 1], elev=48, azim=134)
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=group, cmap=plt.cm.spectral)

ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])

plt.savefig('cocktail_PCA_SpectralClustering.png')
plt.show()

df4 = df3.copy()
df4['group'] = group
df4 = df4.groupby('group')

print('plotting profiles of five groups (PCA - spectral Clustering)')
fig8 = plt.figure(figsize=(30, 20))
plt.title('profiles of five groups')
for i in range(5):
	plt.subplot(5, 1, i+1)
	df_temp = df4.get_group(i)
	df_temp = df_temp.drop('group', 1)
	df_temp = df_temp.sum()
	df_temp.plot.bar()

plt.savefig('cocktail_five_groups_PCA_SpectralClustering.png')
plt.show()

