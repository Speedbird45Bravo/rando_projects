import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_style("dark")

df = pd.read_csv("combined_data_1.txt", header=None, names=['id', 'rating'], usecols=[0,1])
df['rating'] = df['rating'].astype(float)
df = df.reset_index()

p = df.groupby('rating')['rating'].agg(['count'])
movie_count = df.isnull().sum()[1]
cust_count = df['id'].nunique()
rating_count = df['id'].count()

ax = p.plot(kind = 'barh', legend = False, figsize = (15,10))
plt.title('Total pool: {:,} Movies, {:,} customers, {:,} ratings given'.format(movie_count, cust_count, rating_count), fontsize=10)
plt.axis('off')

for i in range(1,6):
    ax.text(p.iloc[i-1][0]/4, i-1, 'rating {}: {:.0f}%'.format(i, p.iloc[i-1][0]*100 / p.sum()[0]), color = 'white', weight = 'bold')

df_nan = pd.DataFrame(pd.isnull(df.rating))
df_nan = df_nan[df_nan['rating'] == True]
df_nan = df_nan.reset_index()

movie_np = []
movie_id = 1

for i,j in zip(df_nan['index'][1:],df_nan['index'][:-1]):
    # numpy approach
    temp = np.full((1,i-j-1), movie_id)
    movie_np = np.append(movie_np, temp)
    movie_id += 1

# Account for last record and corresponding length
# numpy approach
last_record = np.full((1,len(df) - df_nan.iloc[-1, 0] - 1),movie_id)
movie_np = np.append(movie_np, last_record)

print('Movie numpy: {}'.format(movie_np))
print('Length: {}'.format(len(movie_np)))

# remove those Movie ID rows
df = df[pd.notnull(df['rating'])]

df['Movie_Id'] = movie_np.astype(int)
df['id'] = df['id'].astype(int)
print('-Dataset examples-')
print(df.iloc[::5000000, :])

f = ['count','mean']

df_movie_summary = df.groupby('Movie_Id')['rating'].agg(f)
df_movie_summary.index = df_movie_summary.index.map(int)
movie_benchmark = round(df_movie_summary['count'].quantile(0.7),0)
drop_movie_list = df_movie_summary[df_movie_summary['count'] < movie_benchmark].index

print('Movie minimum times of review: {}'.format(movie_benchmark))

df_cust_summary = df.groupby('id')['rating'].agg(f)
df_cust_summary.index = df_cust_summary.index.map(int)
cust_benchmark = round(df_cust_summary['count'].quantile(0.7),0)
drop_cust_list = df_cust_summary[df_cust_summary['count'] < cust_benchmark].index

print('Original Shape: {}'.format(df.shape))
df = df[~df['Movie_Id'].isin(drop_movie_list)]
df = df[~df['id'].isin(drop_cust_list)]
print('After Trim Shape: {}'.format(df.shape))
print('-Data Examples-')
print(df.iloc[::5000000, :])

df_p = pd.pivot_table(df, values='rating', index='id', columns='Movie_Id')
df_title = pd.read_csv('movie_titles.csv', encoding="ISO-8859-1", header=None, names=['Movie_Id', 'Year', 'Name'])
df_title.set_index('Movie_Id', inplace=True)


def recommend(movie_title, min_count):
    print("For movie ({})".format(movie_title))
    print("- Top 10 Movies Recommended Based on Pearsons R correlation")
    i = int(df_title.index[df_title['Name'] == movie_title][0])
    target = df_p[i]
    similar_to_target = df_p.corrwith(target)
    corr_target = pd.DataFrame(similar_to_target, columns=['PearsonR'])
    corr_target.dropna(inplace=True)
    corr_target = corr_target.sort_values('PearsonR', ascending=False)
    corr_target.index = corr_target.index.map(int)
    corr_target = corr_target.join(df_title).join(df_movie_summary)[['PearsonR', 'Name', 'count', 'mean']]
    print(corr_target[corr_target['count'] > min_count][:10].to_string(index=False))


recommend("What the #$*! Do We Know!?", 0)