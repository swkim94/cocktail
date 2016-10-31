import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import requests

# interested whether they would be grouped into 5 neighbors, corresponding to five basic tastes.
#
# using API for http://www.thecocktaildb.com/

# load first cocktail data and initialize dataframe
url = 'http://www.thecocktaildb.com/api/json/v1/1/lookup.php?i=15112'
r = requests.get(url)
json_data = r.json()
key = list(json_data.keys())[0]  # only one top level key
dict = json_data[key][0]
df = pd.DataFrame.from_dict(dict,orient='index').T
print(dict.keys())

# get more cocktail data and add to dataframe.
url = 'http://www.thecocktaildb.com/api/json/v1/1/random.php'
print('downloading data from '+ url)
n_random = 800
j = 0
for i in range(0,n_random):
	r = requests.get(url)
	json_data = r.json()
	if json_data[key] != None:
		key = list(json_data.keys())[0]
		dict = json_data[key][0]
		df_temp = pd.DataFrame.from_dict(dict,orient='index').T
		df = df.append(df_temp, ignore_index=True)
		j += 1
print('finished downloading %i records of cocktails' % j)

# remove duplicates
df = pd.DataFrame.drop_duplicates(df, subset = 'idDrink')
df = df.reset_index(inplace=True)
print('removed duplicate records. now %i records of cocktails' % df.shape[0])

print(df.keys())
# Index(['strMeasure6', 'strMeasure9', 'strMeasure4', 'strIngredient5',
#        'strDrink', 'strInstructions', 'strIngredient1', 'strIngredient11',
#        'strMeasure13', 'strIngredient3', 'strMeasure5', 'idDrink',
#        'strMeasure14', 'strMeasure15', 'strMeasure8', 'strIngredient14',
#        'strIngredient15', 'strIngredient10', 'strMeasure12', 'strAlcoholic',
#        'strMeasure7', 'strMeasure11', 'strMeasure3', 'strIngredient6',
#        'strMeasure2', 'strIngredient4', 'strIngredient12', 'strGlass',
#        'strIngredient9', 'strIngredient7', 'strDrinkThumb', 'strMeasure10',
#        'strCategory', 'strIngredient2', 'strMeasure1', 'dateModified',
#        'strIngredient13', 'strIngredient8'],
#       dtype='object')

df2 = df[['idDrink','strDrink','strCategory','strGlass','strAlcoholic',
		  'strIngredient1', 'strMeasure1',
		  'strIngredient2', 'strMeasure2',
		  'strIngredient3', 'strMeasure3',
		  'strIngredient4', 'strMeasure4',
		  'strIngredient5', 'strMeasure5',
		  'strIngredient6', 'strMeasure6',
		  'strIngredient7', 'strMeasure7',
		  'strIngredient8', 'strMeasure8',
		  'strIngredient9', 'strMeasure9',
		  'strIngredient10', 'strMeasure10',
		  'strIngredient11', 'strMeasure11',
		  'strIngredient12', 'strMeasure12',
		  'strIngredient13', 'strMeasure13',
		  'strIngredient14', 'strMeasure14',
		  'strIngredient15', 'strMeasure15',
		  'strInstructions','strDrinkThumb']]

df2.to_pickle('cocktails2.pkl')

df=pd.read_pickle('cocktails2.pkl')
df2 = df.set_index('idDrink',drop =True)
df2 = df2.sort_index()
df2 = df2[['strIngredient'+str(i) for i in range(1,16)]]
ingredients = pd.Series(pd.unique(df2.values.ravel()))
ingredients = ingredients[ingredients!='']
n_dim = ingredients.size
print('number of possible ingredients = %i' % n_dim)
print('Each cocktail will be vectorized and embedded into a %i-dimensional vector space.' % n_dim)
print('later plan to add amplitude for each vector dimension using the quantity of ingredient.')
print('vectorizing...')
array = np.zeros(n_dim)
for i, row in df2.iterrows():
	if i == df2.first_valid_index():
		for j, item in enumerate(ingredients):
			array[j] = (row == item).sum()
		df3 = pd.DataFrame(array, columns=[i], index=ingredients).T
	else:
		for j, item in enumerate(ingredients):
			array[j] = (row == item).sum()
		df_temp = pd.DataFrame(array, columns=[i], index=ingredients).T
		df3 = df3.append(df_temp)
print('finished vectorizing cocktails')

print('plotting ingredient spectrum of first cocktail')
plt.figure(figsize=(30, 9))
df3.iloc[0].plot.bar()
plt.show()
plt.savefig('cocktail_fingerprint.png')

print('plotting histogram of ingredients from all cocktails')
plt.figure(figsize=(30, 9))
df_temp = df3.sum()
df_temp.plot.bar()
plt.show()
plt.savefig('cocktail_histogram.png')

X = df3.as_matrix()
nbrs = NearestNeighbors(n_neighbors=5, algorithm='auto').fit(X)
distances, indices = nbrs.kneighbors(X)

group = np.ones(X.shape[0])
i = 0
for index in indices:
	group[i] = np.argmax(index)
	i += 1
df3['group'] = group
df4 = df3.groupby('group')

print('plotting profiles of five groups')
plt.figure(figsize=(30, 20))
for i in range(5):
	plt.subplot(5, 1, i+1)
	df_temp = df4.get_group(i)
	df_temp = df_temp.drop('group', 1)
	df_temp = df_temp.sum()
	df_temp.plot.bar()

plt.show()
plt.savefig('cocktail_five_groups.png')
