
# Goal for this Notebook:

Use analysis tools provided in Python to interpret the different musical characteristics from songs in Spotify database and how these characteristics influence the popularity factor associated with the song. We also reviewed songs on the Billboard Hot 100 list to compare to the data in the Spotify database. 

Data Handling
    Importing Data with Pandas
    Cleaning Data
    Exploring Data through Visualizations with Matplotlib
    
 Data Analysis
     Regression Models

# Motivation & Summary

For this project we decided to investigate a song’s popularity and if/how there are defining characteristics that make a particular song more popular than another.  

Our initial hypothesis is particular musical characteristics are direct correlators with a song’s popularity and based on these connections we can devise a prediction model to predict whether or not a song is popular.

In order to build our model we needed to analyze a population of songs and their musical characteristics. Where would 
we find a dataset that contained this info?

We discovered a dataset on Kaggle that contains 232,725 tracks pulled from Spotify’s music database via the Spotify API. The songs come from 26 different genres and the dataset also provided 14 different explanatory variables based on the song’s musical characteristics 

We did use the “popularity” variable in the dataset as our targeted dependent variable.

After exploring and running statistical models based on the data provided in the Spotify dataset, we were unable to find direct correlations between the inputs of the explanatory variables and the out of the song’s popularity. There was no significant correlation between any of the musical characteristics and the popularity variable. 

Therefore, we decided to move a different direction and conduct a couple different experiments help us define a model for measuring/predicting popularity among songs.


```python
# Dependencies

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import scipy.stats as stats
import warnings
warnings.filterwarnings('ignore')
import statsmodels.api as sm
from sklearn import datasets
from sklearn import linear_model
from sklearn import datasets
import seaborn as sns
sns.set(style="ticks")
```


```python
# Read in the data

data = pd.read_csv("Resources/SpotifyFeatures.csv")
```


```python
# View the header of the data
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>genre</th>
      <th>artist_name</th>
      <th>track_name</th>
      <th>track_id</th>
      <th>popularity</th>
      <th>acousticness</th>
      <th>danceability</th>
      <th>duration_ms</th>
      <th>energy</th>
      <th>instrumentalness</th>
      <th>key</th>
      <th>liveness</th>
      <th>loudness</th>
      <th>mode</th>
      <th>speechiness</th>
      <th>tempo</th>
      <th>time_signature</th>
      <th>valence</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Movie</td>
      <td>Henri Salvador</td>
      <td>C'est beau de faire un Show</td>
      <td>0BRjO6ga9RKCKjfDqeFgWV</td>
      <td>0</td>
      <td>0.611</td>
      <td>0.389</td>
      <td>99373</td>
      <td>0.910</td>
      <td>0.000</td>
      <td>C#</td>
      <td>0.3460</td>
      <td>-1.828</td>
      <td>Major</td>
      <td>0.0525</td>
      <td>166.969</td>
      <td>4/4</td>
      <td>0.814</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Movie</td>
      <td>Martin &amp; les fées</td>
      <td>Perdu d'avance (par Gad Elmaleh)</td>
      <td>0BjC1NfoEOOusryehmNudP</td>
      <td>1</td>
      <td>0.246</td>
      <td>0.590</td>
      <td>137373</td>
      <td>0.737</td>
      <td>0.000</td>
      <td>F#</td>
      <td>0.1510</td>
      <td>-5.559</td>
      <td>Minor</td>
      <td>0.0868</td>
      <td>174.003</td>
      <td>4/4</td>
      <td>0.816</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Movie</td>
      <td>Joseph Williams</td>
      <td>Don't Let Me Be Lonely Tonight</td>
      <td>0CoSDzoNIKCRs124s9uTVy</td>
      <td>3</td>
      <td>0.952</td>
      <td>0.663</td>
      <td>170267</td>
      <td>0.131</td>
      <td>0.000</td>
      <td>C</td>
      <td>0.1030</td>
      <td>-13.879</td>
      <td>Minor</td>
      <td>0.0362</td>
      <td>99.488</td>
      <td>5/4</td>
      <td>0.368</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Movie</td>
      <td>Henri Salvador</td>
      <td>Dis-moi Monsieur Gordon Cooper</td>
      <td>0Gc6TVm52BwZD07Ki6tIvf</td>
      <td>0</td>
      <td>0.703</td>
      <td>0.240</td>
      <td>152427</td>
      <td>0.326</td>
      <td>0.000</td>
      <td>C#</td>
      <td>0.0985</td>
      <td>-12.178</td>
      <td>Major</td>
      <td>0.0395</td>
      <td>171.758</td>
      <td>4/4</td>
      <td>0.227</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Movie</td>
      <td>Fabien Nataf</td>
      <td>Ouverture</td>
      <td>0IuslXpMROHdEPvSl1fTQK</td>
      <td>4</td>
      <td>0.950</td>
      <td>0.331</td>
      <td>82625</td>
      <td>0.225</td>
      <td>0.123</td>
      <td>F</td>
      <td>0.2020</td>
      <td>-21.150</td>
      <td>Major</td>
      <td>0.0456</td>
      <td>140.576</td>
      <td>4/4</td>
      <td>0.390</td>
    </tr>
  </tbody>
</table>
</div>



# Questions & Data

The main questions we focused on were “what are the measurable characteristics of a song that we can use to help explain the popularity”?

As mentioned earlier, we did discover the dataset built from the Spotify API that contained the very explanatory variables we believed would tell the story of song popularity. The Spotify API is a public resource for developers to use to mine through Spotify’s music library. The library is built of the song info such as artist name, genre and track name but also contains the data on the song’s musical characteristics. The Spotify API defines the features as the following:
    
    Key: Estimate of what key the song is in
    
    Mode: Estimate of what the mode of the song is in
    
    Instrumentalness: This value represents the amount of vocals in the song. The closer it is to 1.0, the more instrumental the song is.
    
    Acousticness: This value describes how acoustic a song is. A score of 1.0 means the song is most likely to be an acoustic one.

    Liveness: This value describes the probability that the song was recorded with a live audience. According to the official documentation “a value above 0.8 provides strong likelihood that the track is live”.
   
   Speechiness: “Speechiness detects the presence of spoken words in a track”. If the speechiness of a song is above 0.66, it is probably made of spoken words, a score between 0.33 and 0.66 is a song that may contain both music and words, and a score below 0.33 means the song does not have any speech.
    
    Energy: “(energy) represents a perceptual measure of intensity and activity. Typically, energetic tracks feel fast, loud, and noisy”.
    
    Danceability: “Danceability describes how suitable a track is for dancing based on a combination of musical elements including tempo, rhythm stability, beat strength, and overall regularity. A value of 0.0 is least danceable and 1.0 is most danceable”.
    
    Valence: “A measure from 0.0 to 1.0 describing the musical positiveness conveyed by a track. Tracks with high valence sound more positive (e.g. happy, cheerful, euphoric), while tracks with low valence sound more negative (e.g. sad, depressed, angry)”.


# Cleaning the Data and Data Exploration/Visual Analysis

For the purpose of this analysis we are trying to judge song popularity based on musical attributes. After reviewing the different genres within the set we decided to remove the genres that did not coincide with musical characteristics or are not considered contemporary popular genres. Therefore we need to clean some of the genres out of the data.


```python
new_data = data[(data.genre != 'Comedy') & (data.genre != 'Soundtrack') & (data.genre != 'Children’s Music') & (data.genre != 'Classical') & (data.genre != 'Anime') & (data.genre != 'Opera') & (data.genre != 'Movie') & (data.genre != "Children's Music") & (data.genre != "A Capella")]
```

Now that we have removed the extraneous genres, we ran the basic statistics for the entire datset and isolating the popularity factor on the updated data set.


```python
new_data.groupby('genre').boxplot(column=['popularity'], figsize=(12,30))
plt.tight_layout()
fig1 = plt.gcf()
plt.show()
fig1.savefig("Boxchart.png", bbox_inches='tight', dpi=100)
```


![png](Spoitfy_Analysis_Draft_files/Spoitfy_Analysis_Draft_9_0.png)


The above results show the 50% threshold for popularity falls at a score of 48. Since the goal of this analysis is to find what characteristics attribute to popularity score, we decided to break the data into two groups. The first group is the "unpopular" group made up of songs with popularity scores less than 48. The "popular" group is the second group made up of songs with scores 48 or higher in popularity. We break up the data into the 2 groups below. 


```python
new_data['popularity'].describe()
```




    count    164245.000000
    mean         47.542878
    std          14.162002
    min           0.000000
    25%          38.000000
    50%          48.000000
    75%          58.000000
    max         100.000000
    Name: popularity, dtype: float64




```python
pop_data_unpop = new_data.loc[new_data['popularity'] < 48]
pop_data_pop = new_data.loc[new_data['popularity'] >= 48]
```

Now we take a look at the distribution of the genres in each group below by looking at the value counts and then using that info in pie charts.

We can now see that the frequencies of genres appear inverses among the two groups. We can see that the Pop genre is the most frequent in the popular group but least frequent in the unpopular group and Ska is the most frequent in the unpopular group but least frequent in the unpopular group. This is important because it shows there is variability across the popularity scores among the genres.

We then decided to focus on the top 5 frequent genres for each group to focus our analysis. We chose Ska, Blues, World, Electronic, and Reggae for the unpopular group and Pop, Rap, Rock, Hip-Hop and Indie from the popular group.

To review the spread of the popularity scores among each of the genres, we plotted them using boxplots to show each genre's "popularity" score statistics as shown below.

In order to keep the charts cleaner, we only left the top 5 genres of each.


```python
pop_data_unpop['genre'].value_counts() 
pop_data_unpop_genre = pop_data_unpop.loc[(pop_data_unpop['genre'] == 'Ska') | (pop_data_unpop['genre'] == 'Blues') | 
                                         (pop_data_unpop['genre'] == 'World') | (pop_data_unpop['genre'] == 'Electronic') |
                                         (pop_data_unpop['genre'] == 'Reggae')]

pop_data_pop['genre'].value_counts()
pop_data_pop_genre = pop_data_pop.loc[(pop_data_pop['genre'] == 'Pop') | (pop_data_pop['genre'] == 'Rap') | 
                                         (pop_data_pop['genre'] == 'Rock') | (pop_data_pop['genre'] == 'Hip-Hop') |
                                         (pop_data_pop['genre'] == 'Indie')]
```


```python
plt.figure(figsize = (20,20))
genre_count_unpop = pop_data_unpop_genre['genre'].value_counts()
genre_names = genre_count_unpop.index
plt.pie(genre_count_unpop,labels=genre_names, autopct='%1.1f%%', shadow=True, textprops={'fontsize': 25})
fig2 = plt.gcf()
plt.show()
fig2.savefig('Unpopular_Pie.png', dpi=100)
```


![png](Spoitfy_Analysis_Draft_files/Spoitfy_Analysis_Draft_15_0.png)



```python
plt.figure(figsize = (20,20))
genre_count_pop = pop_data_pop_genre['genre'].value_counts()
genre_names = genre_count_pop.index
plt.pie(genre_count_pop,labels=genre_names, autopct='%1.1f%%', shadow=True, textprops={'fontsize': 25})
fig3 = plt.gcf()
plt.show()
fig3.savefig('Popular_Pie.png', dpi=100)
```


![png](Spoitfy_Analysis_Draft_files/Spoitfy_Analysis_Draft_16_0.png)


As we continue to review the data we notice that some of the columns are made of categorical data rather than numerical data. One of these categorical characteristics is the Key column that represents the key in which the song is in. To get a better idea of variability of the different keys across the two groups, we used a bar chart to show the frequencies of each key in both the popular and unpopular groups. 


```python
keys_unpop = pop_data_unpop_genre['key'].value_counts()
names = ['G', 'C', 'A', 'D', 'C#', 'F', 'B', 'E', 'A#', 'F#', 'G#', 'D#']
my_colors = ['red', 'gold', 'darkgreen', 'blue', 'grey', 'yellow', 'orange', 'purple', 'aqua', 'lightpink', 'crimson', 'maroon']

plt.bar(names, keys_unpop, align='center', alpha=0.5,color=my_colors)
plt.xticks(names)
plt.ylabel('Count of Key Usage Rate')
plt.title('Keys in the Unpopular Group')
fig4 = plt.gcf()
plt.show()
fig4.savefig('Keys_Unpopular.png', dpi=100)
```


![png](Spoitfy_Analysis_Draft_files/Spoitfy_Analysis_Draft_18_0.png)



```python
keys_pop = pop_data_pop_genre['key'].value_counts()
names = ['C#', 'C', 'G', 'D', 'A', 'F', 'B', 'G#', 'F#', 'E', 'A#', 'D#']
my_colors = ['red', 'gold', 'darkgreen', 'blue', 'grey', 'yellow', 'orange', 'purple', 'aqua', 'lightpink', 'crimson', 'maroon']

plt.bar(names, keys_pop, align='center', alpha=0.5,color=my_colors)
plt.xticks(names)
plt.ylabel('Count of Key Usage Rate')
plt.title('Keys in the Popular Group')
fig5 = plt.gcf()
plt.show()
fig5.savefig('Keys_Popular.png', dpi=100)
```


![png](Spoitfy_Analysis_Draft_files/Spoitfy_Analysis_Draft_19_0.png)


As we can see, the majority of the songs in the unpopular group are in the key of G whereas the majority of songs in the popular group are in the key of C#. This could lead us to believe that songs in the key of C# have a better chance of being considered popular. 

# Data Analysis

Now that we have visualized some of the variability in the dataset, we want to see if we can build a model based on the musical characteristics that determine the popularity factor of a song. In order to do so, we decided to start with regression analysis to see what characteristics correlate with popularity across the popular and unpopular groups. We started with some matrix plots and correlation heat map to see where the strongest correlations may be found.


```python
sns_plot_one = sns.pairplot(pop_data_pop_genre,
            x_vars = ['acousticness','danceability', 'duration_ms', 'energy', 'instrumentalness', 'liveness', 'loudness', 'speechiness','tempo', 'valence'],
            y_vars = ['popularity'])
sns_plot_one.savefig("Popular_Correlation.png")
```


![png](Spoitfy_Analysis_Draft_files/Spoitfy_Analysis_Draft_22_0.png)



```python
sns_plot_two = sns.pairplot(pop_data_unpop_genre,
            x_vars = ['acousticness','danceability', 'duration_ms', 'energy', 'instrumentalness', 'liveness', 'loudness', 'speechiness','tempo', 'valence'],
            y_vars = ['popularity'])
sns_plot_two.savefig("Unpopular_Correlation.png")
```


![png](Spoitfy_Analysis_Draft_files/Spoitfy_Analysis_Draft_23_0.png)



```python
plt.figure(figsize = (16,5))
sns.heatmap(pop_data_pop_genre.corr(), cmap="coolwarm", annot=True)
fig6 = plt.gcf()
plt.show()
fig6.savefig('Popular_Heat.png', dpi=100)
```


![png](Spoitfy_Analysis_Draft_files/Spoitfy_Analysis_Draft_24_0.png)



```python
plt.figure(figsize = (16,5))
sns.heatmap(pop_data_unpop_genre.corr(), cmap="coolwarm", annot=True)
fig7 = plt.gcf()
plt.show()
fig6.savefig('Unpopular_Heat.png', dpi=100)
```


![png](Spoitfy_Analysis_Draft_files/Spoitfy_Analysis_Draft_25_0.png)


As we can see, there is not much correlation between the popularity score and the musical characteristics but we do want to try to create some simple multiple regression functions to test our hypothesis. Below we created the regression models using the musical characteristics where the correlation coefficient was positive based on the heat maps above. 


```python
X = pop_data_pop_genre[['danceability','energy', 'loudness','tempo', 'valence']]
Y = pop_data_pop_genre['popularity']

regr = linear_model.LinearRegression()
regr.fit(X, Y)

print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_)
```

    Intercept: 
     64.36855230913588
    Coefficients: 
     [ 3.74188866e+00 -5.85320486e+00  4.89983918e-01  4.70539685e-03
      1.20277286e+00]



```python
X = pop_data_unpop_genre[['acousticness','danceability', 'duration_ms','instrumentalness']]
Y = pop_data_unpop_genre['popularity']

regr = linear_model.LinearRegression()
regr.fit(X, Y)

print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_)
```

    Intercept: 
     27.90755753101906
    Coefficients: 
     [1.51245569e+00 3.76743683e+00 5.77443604e-06 9.92881677e-01]


We use multiple regression models to help predict outcomes by creating formulas that will pull in the observed data. In our case we want to use the observed musical characteristics to predict the popularity score associated with that song. Based on our analysis we found that the popularity score is not correlated to any of the explanatory variables. Based on our initial hypothesis, we did believe there was a correlation but our findings show that we will need to reject the null hypothesis.

Instead of ending our analysis at this step, we decided to change our approach to find more trends that lead to popularity of a song. We focused our analysis on the the genre of the song instead of the particular musical characteristics that make up the song itself.

We did this by a few different approaches. The first was to develop a new output variable called "Partyability." This variable is made up of a aggregate of a few musical characteristics that are considered popular for popular bar/club songs. This new variable allowed us to determine which genres are well suited to play at a party, club, or bar. 

We also took the approach of using the Billboard top charts to pull info on top #1 songs throughout the last 5 years and compare those songs' genre against our Spotify dataset.

# Partybility 

According to Denis’ analysis, Pop, Rap, Rock, Hip-Hop, Dance, Indie, R&B, & Alternative were the top 8 most popular genres. As part of our overall analysis, we analyzed which of these top genres would be ideal for a party setting. Tracks played at parties, should be popular, easy to dance to, lively, loud, and upbeat. We refer to these characteristics as a track’s ‘partybality’. Thus, we made a new, calculated partybality column. We determined a genre’s ‘partybility’ by first calculating each respective genre’s averages for popularity score, danceability score, energy score, loudness (decibels), and tempo (BPM).
Then we added the all the average scores to calculate each genre’s partibility.


```python
#Dependencies and Setup
%matplotlib inline
import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np
```


```python

#Read Data
spotify_data = pd.read_csv("../Resources/SpotifyFeatures.csv")
spotify_data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>genre</th>
      <th>artist_name</th>
      <th>track_name</th>
      <th>track_id</th>
      <th>popularity</th>
      <th>acousticness</th>
      <th>danceability</th>
      <th>duration_ms</th>
      <th>energy</th>
      <th>instrumentalness</th>
      <th>key</th>
      <th>liveness</th>
      <th>loudness</th>
      <th>mode</th>
      <th>speechiness</th>
      <th>tempo</th>
      <th>time_signature</th>
      <th>valence</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Movie</td>
      <td>Henri Salvador</td>
      <td>C'est beau de faire un Show</td>
      <td>0BRjO6ga9RKCKjfDqeFgWV</td>
      <td>0</td>
      <td>0.611</td>
      <td>0.389</td>
      <td>99373</td>
      <td>0.910</td>
      <td>0.000</td>
      <td>C#</td>
      <td>0.3460</td>
      <td>-1.828</td>
      <td>Major</td>
      <td>0.0525</td>
      <td>166.969</td>
      <td>4/4</td>
      <td>0.814</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Movie</td>
      <td>Martin &amp; les fées</td>
      <td>Perdu d'avance (par Gad Elmaleh)</td>
      <td>0BjC1NfoEOOusryehmNudP</td>
      <td>1</td>
      <td>0.246</td>
      <td>0.590</td>
      <td>137373</td>
      <td>0.737</td>
      <td>0.000</td>
      <td>F#</td>
      <td>0.1510</td>
      <td>-5.559</td>
      <td>Minor</td>
      <td>0.0868</td>
      <td>174.003</td>
      <td>4/4</td>
      <td>0.816</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Movie</td>
      <td>Joseph Williams</td>
      <td>Don't Let Me Be Lonely Tonight</td>
      <td>0CoSDzoNIKCRs124s9uTVy</td>
      <td>3</td>
      <td>0.952</td>
      <td>0.663</td>
      <td>170267</td>
      <td>0.131</td>
      <td>0.000</td>
      <td>C</td>
      <td>0.1030</td>
      <td>-13.879</td>
      <td>Minor</td>
      <td>0.0362</td>
      <td>99.488</td>
      <td>5/4</td>
      <td>0.368</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Movie</td>
      <td>Henri Salvador</td>
      <td>Dis-moi Monsieur Gordon Cooper</td>
      <td>0Gc6TVm52BwZD07Ki6tIvf</td>
      <td>0</td>
      <td>0.703</td>
      <td>0.240</td>
      <td>152427</td>
      <td>0.326</td>
      <td>0.000</td>
      <td>C#</td>
      <td>0.0985</td>
      <td>-12.178</td>
      <td>Major</td>
      <td>0.0395</td>
      <td>171.758</td>
      <td>4/4</td>
      <td>0.227</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Movie</td>
      <td>Fabien Nataf</td>
      <td>Ouverture</td>
      <td>0IuslXpMROHdEPvSl1fTQK</td>
      <td>4</td>
      <td>0.950</td>
      <td>0.331</td>
      <td>82625</td>
      <td>0.225</td>
      <td>0.123</td>
      <td>F</td>
      <td>0.2020</td>
      <td>-21.150</td>
      <td>Major</td>
      <td>0.0456</td>
      <td>140.576</td>
      <td>4/4</td>
      <td>0.390</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Read in 6 needed columns into data frame
slapability_data = spotify_data[["genre", "popularity", "danceability", "energy", "loudness","tempo"]]
slapability_data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>genre</th>
      <th>popularity</th>
      <th>danceability</th>
      <th>energy</th>
      <th>loudness</th>
      <th>tempo</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Movie</td>
      <td>0</td>
      <td>0.389</td>
      <td>0.9100</td>
      <td>-1.828</td>
      <td>166.969</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Movie</td>
      <td>1</td>
      <td>0.590</td>
      <td>0.7370</td>
      <td>-5.559</td>
      <td>174.003</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Movie</td>
      <td>3</td>
      <td>0.663</td>
      <td>0.1310</td>
      <td>-13.879</td>
      <td>99.488</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Movie</td>
      <td>0</td>
      <td>0.240</td>
      <td>0.3260</td>
      <td>-12.178</td>
      <td>171.758</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Movie</td>
      <td>4</td>
      <td>0.331</td>
      <td>0.2250</td>
      <td>-21.150</td>
      <td>140.576</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Movie</td>
      <td>0</td>
      <td>0.578</td>
      <td>0.0948</td>
      <td>-14.970</td>
      <td>87.479</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Movie</td>
      <td>2</td>
      <td>0.703</td>
      <td>0.2700</td>
      <td>-12.675</td>
      <td>82.873</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Movie</td>
      <td>15</td>
      <td>0.416</td>
      <td>0.2690</td>
      <td>-8.949</td>
      <td>96.827</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Movie</td>
      <td>0</td>
      <td>0.734</td>
      <td>0.4810</td>
      <td>-7.725</td>
      <td>125.080</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Movie</td>
      <td>10</td>
      <td>0.598</td>
      <td>0.7050</td>
      <td>-7.790</td>
      <td>137.496</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Movie</td>
      <td>0</td>
      <td>0.191</td>
      <td>0.1450</td>
      <td>-17.239</td>
      <td>85.225</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Movie</td>
      <td>2</td>
      <td>0.230</td>
      <td>0.1330</td>
      <td>-19.051</td>
      <td>91.739</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Movie</td>
      <td>4</td>
      <td>0.680</td>
      <td>0.6000</td>
      <td>-7.702</td>
      <td>110.026</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Movie</td>
      <td>3</td>
      <td>0.758</td>
      <td>0.2640</td>
      <td>-11.966</td>
      <td>110.068</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Movie</td>
      <td>0</td>
      <td>0.400</td>
      <td>0.1740</td>
      <td>-13.869</td>
      <td>115.022</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Movie</td>
      <td>0</td>
      <td>0.588</td>
      <td>0.4050</td>
      <td>-15.550</td>
      <td>83.560</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Movie</td>
      <td>0</td>
      <td>0.625</td>
      <td>0.2370</td>
      <td>-16.655</td>
      <td>108.508</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Movie</td>
      <td>3</td>
      <td>0.744</td>
      <td>0.9530</td>
      <td>-4.986</td>
      <td>129.959</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Movie</td>
      <td>1</td>
      <td>0.451</td>
      <td>0.4910</td>
      <td>-8.730</td>
      <td>71.633</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Movie</td>
      <td>4</td>
      <td>0.522</td>
      <td>0.7480</td>
      <td>-5.203</td>
      <td>184.063</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Movie</td>
      <td>1</td>
      <td>0.362</td>
      <td>0.4360</td>
      <td>-6.971</td>
      <td>79.542</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Movie</td>
      <td>1</td>
      <td>0.855</td>
      <td>0.5040</td>
      <td>-12.359</td>
      <td>128.052</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Movie</td>
      <td>8</td>
      <td>0.704</td>
      <td>0.8040</td>
      <td>-6.699</td>
      <td>127.999</td>
    </tr>
    <tr>
      <th>23</th>
      <td>Movie</td>
      <td>0</td>
      <td>0.707</td>
      <td>0.6960</td>
      <td>-10.472</td>
      <td>114.752</td>
    </tr>
    <tr>
      <th>24</th>
      <td>Movie</td>
      <td>5</td>
      <td>0.526</td>
      <td>0.2980</td>
      <td>-7.287</td>
      <td>156.350</td>
    </tr>
    <tr>
      <th>25</th>
      <td>Movie</td>
      <td>0</td>
      <td>0.711</td>
      <td>0.2180</td>
      <td>-22.212</td>
      <td>125.521</td>
    </tr>
    <tr>
      <th>26</th>
      <td>Movie</td>
      <td>0</td>
      <td>0.636</td>
      <td>0.5940</td>
      <td>-7.943</td>
      <td>111.361</td>
    </tr>
    <tr>
      <th>27</th>
      <td>Movie</td>
      <td>0</td>
      <td>0.594</td>
      <td>0.4810</td>
      <td>-7.678</td>
      <td>136.182</td>
    </tr>
    <tr>
      <th>28</th>
      <td>Movie</td>
      <td>2</td>
      <td>0.835</td>
      <td>0.3840</td>
      <td>-15.629</td>
      <td>126.892</td>
    </tr>
    <tr>
      <th>29</th>
      <td>Movie</td>
      <td>0</td>
      <td>0.695</td>
      <td>0.5000</td>
      <td>-10.217</td>
      <td>112.965</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>232695</th>
      <td>Soul</td>
      <td>37</td>
      <td>0.646</td>
      <td>0.8270</td>
      <td>-3.651</td>
      <td>103.044</td>
    </tr>
    <tr>
      <th>232696</th>
      <td>Soul</td>
      <td>40</td>
      <td>0.711</td>
      <td>0.6970</td>
      <td>-7.788</td>
      <td>99.987</td>
    </tr>
    <tr>
      <th>232697</th>
      <td>Soul</td>
      <td>41</td>
      <td>0.888</td>
      <td>0.5270</td>
      <td>-8.293</td>
      <td>123.028</td>
    </tr>
    <tr>
      <th>232698</th>
      <td>Soul</td>
      <td>41</td>
      <td>0.767</td>
      <td>0.4940</td>
      <td>-5.349</td>
      <td>96.003</td>
    </tr>
    <tr>
      <th>232699</th>
      <td>Soul</td>
      <td>39</td>
      <td>0.597</td>
      <td>0.8590</td>
      <td>-6.343</td>
      <td>172.968</td>
    </tr>
    <tr>
      <th>232700</th>
      <td>Soul</td>
      <td>32</td>
      <td>0.550</td>
      <td>0.6180</td>
      <td>-11.138</td>
      <td>110.738</td>
    </tr>
    <tr>
      <th>232701</th>
      <td>Soul</td>
      <td>35</td>
      <td>0.600</td>
      <td>0.3950</td>
      <td>-6.868</td>
      <td>143.827</td>
    </tr>
    <tr>
      <th>232702</th>
      <td>Soul</td>
      <td>46</td>
      <td>0.505</td>
      <td>0.4630</td>
      <td>-10.652</td>
      <td>105.660</td>
    </tr>
    <tr>
      <th>232703</th>
      <td>Soul</td>
      <td>41</td>
      <td>0.563</td>
      <td>0.6200</td>
      <td>-6.292</td>
      <td>82.059</td>
    </tr>
    <tr>
      <th>232704</th>
      <td>Soul</td>
      <td>40</td>
      <td>0.875</td>
      <td>0.8300</td>
      <td>-4.222</td>
      <td>114.024</td>
    </tr>
    <tr>
      <th>232705</th>
      <td>Soul</td>
      <td>40</td>
      <td>0.380</td>
      <td>0.1600</td>
      <td>-17.717</td>
      <td>179.284</td>
    </tr>
    <tr>
      <th>232706</th>
      <td>Soul</td>
      <td>37</td>
      <td>0.752</td>
      <td>0.5840</td>
      <td>-8.388</td>
      <td>118.054</td>
    </tr>
    <tr>
      <th>232707</th>
      <td>Soul</td>
      <td>45</td>
      <td>0.502</td>
      <td>0.2990</td>
      <td>-8.956</td>
      <td>116.845</td>
    </tr>
    <tr>
      <th>232708</th>
      <td>Soul</td>
      <td>41</td>
      <td>0.754</td>
      <td>0.4790</td>
      <td>-8.305</td>
      <td>129.971</td>
    </tr>
    <tr>
      <th>232709</th>
      <td>Soul</td>
      <td>42</td>
      <td>0.837</td>
      <td>0.5410</td>
      <td>-5.068</td>
      <td>103.971</td>
    </tr>
    <tr>
      <th>232710</th>
      <td>Soul</td>
      <td>42</td>
      <td>0.764</td>
      <td>0.4480</td>
      <td>-14.135</td>
      <td>116.096</td>
    </tr>
    <tr>
      <th>232711</th>
      <td>Soul</td>
      <td>41</td>
      <td>0.764</td>
      <td>0.7570</td>
      <td>-7.049</td>
      <td>115.016</td>
    </tr>
    <tr>
      <th>232712</th>
      <td>Soul</td>
      <td>38</td>
      <td>0.630</td>
      <td>0.6550</td>
      <td>-6.708</td>
      <td>102.009</td>
    </tr>
    <tr>
      <th>232713</th>
      <td>Soul</td>
      <td>39</td>
      <td>0.659</td>
      <td>0.6440</td>
      <td>-4.510</td>
      <td>143.887</td>
    </tr>
    <tr>
      <th>232714</th>
      <td>Soul</td>
      <td>39</td>
      <td>0.611</td>
      <td>0.4930</td>
      <td>-9.297</td>
      <td>115.920</td>
    </tr>
    <tr>
      <th>232715</th>
      <td>Soul</td>
      <td>42</td>
      <td>0.394</td>
      <td>0.3460</td>
      <td>-13.617</td>
      <td>90.831</td>
    </tr>
    <tr>
      <th>232716</th>
      <td>Soul</td>
      <td>39</td>
      <td>0.736</td>
      <td>0.7010</td>
      <td>-4.345</td>
      <td>99.991</td>
    </tr>
    <tr>
      <th>232717</th>
      <td>Soul</td>
      <td>43</td>
      <td>0.802</td>
      <td>0.5160</td>
      <td>-9.014</td>
      <td>175.666</td>
    </tr>
    <tr>
      <th>232718</th>
      <td>Soul</td>
      <td>43</td>
      <td>0.294</td>
      <td>0.7390</td>
      <td>-7.167</td>
      <td>176.402</td>
    </tr>
    <tr>
      <th>232719</th>
      <td>Soul</td>
      <td>32</td>
      <td>0.423</td>
      <td>0.3370</td>
      <td>-13.092</td>
      <td>80.023</td>
    </tr>
    <tr>
      <th>232720</th>
      <td>Soul</td>
      <td>39</td>
      <td>0.687</td>
      <td>0.7140</td>
      <td>-10.626</td>
      <td>115.542</td>
    </tr>
    <tr>
      <th>232721</th>
      <td>Soul</td>
      <td>38</td>
      <td>0.785</td>
      <td>0.6830</td>
      <td>-6.944</td>
      <td>113.830</td>
    </tr>
    <tr>
      <th>232722</th>
      <td>Soul</td>
      <td>47</td>
      <td>0.517</td>
      <td>0.4190</td>
      <td>-8.282</td>
      <td>84.135</td>
    </tr>
    <tr>
      <th>232723</th>
      <td>Soul</td>
      <td>44</td>
      <td>0.745</td>
      <td>0.7040</td>
      <td>-7.137</td>
      <td>100.031</td>
    </tr>
    <tr>
      <th>232724</th>
      <td>Soul</td>
      <td>35</td>
      <td>0.758</td>
      <td>0.4700</td>
      <td>-6.708</td>
      <td>113.897</td>
    </tr>
  </tbody>
</table>
<p>232725 rows × 6 columns</p>
</div>




```python
#Group by genre
grouped_genre = slapability_data.groupby(['genre'])
grouped_genre

#Obtain averages of popularity, danceability, energy, loudness, tempo
avg_groups = grouped_genre.mean()
```


```python
#Get data with loc
top_genres_data = avg_groups.loc[["Pop", "Rap", "Rock","Hip-Hop", "Indie", "Dance", "R&B", "Alternative"],
                                    ["popularity","danceability", "energy", "loudness","tempo"]]
#Display
top_genres_data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>popularity</th>
      <th>danceability</th>
      <th>energy</th>
      <th>loudness</th>
      <th>tempo</th>
    </tr>
    <tr>
      <th>genre</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Pop</th>
      <td>66.590667</td>
      <td>0.640236</td>
      <td>0.642208</td>
      <td>-6.495423</td>
      <td>121.175844</td>
    </tr>
    <tr>
      <th>Rap</th>
      <td>60.533795</td>
      <td>0.697244</td>
      <td>0.650520</td>
      <td>-6.669916</td>
      <td>121.100808</td>
    </tr>
    <tr>
      <th>Rock</th>
      <td>59.619392</td>
      <td>0.538292</td>
      <td>0.683670</td>
      <td>-7.285875</td>
      <td>122.629630</td>
    </tr>
    <tr>
      <th>Hip-Hop</th>
      <td>58.423131</td>
      <td>0.718808</td>
      <td>0.643275</td>
      <td>-6.860286</td>
      <td>120.791039</td>
    </tr>
    <tr>
      <th>Indie</th>
      <td>54.701561</td>
      <td>0.566821</td>
      <td>0.581002</td>
      <td>-7.915142</td>
      <td>119.290814</td>
    </tr>
    <tr>
      <th>Dance</th>
      <td>57.275256</td>
      <td>0.638191</td>
      <td>0.698067</td>
      <td>-6.054241</td>
      <td>120.795919</td>
    </tr>
    <tr>
      <th>R&amp;B</th>
      <td>52.308719</td>
      <td>0.642125</td>
      <td>0.564248</td>
      <td>-7.597064</td>
      <td>116.373834</td>
    </tr>
    <tr>
      <th>Alternative</th>
      <td>50.213430</td>
      <td>0.541898</td>
      <td>0.711519</td>
      <td>-6.540803</td>
      <td>122.534485</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Generate and calculate 'partybility column'
partybility = top_genres_data["popularity"] + top_genres_data["danceability"] + top_genres_data["energy"] + top_genres_data["loudness"] + top_genres_data["tempo"]
top_genres_data["partybality"] = partybility

#display
top_genres_data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>popularity</th>
      <th>danceability</th>
      <th>energy</th>
      <th>loudness</th>
      <th>tempo</th>
      <th>partybality</th>
    </tr>
    <tr>
      <th>genre</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Pop</th>
      <td>66.590667</td>
      <td>0.640236</td>
      <td>0.642208</td>
      <td>-6.495423</td>
      <td>121.175844</td>
      <td>182.553533</td>
    </tr>
    <tr>
      <th>Rap</th>
      <td>60.533795</td>
      <td>0.697244</td>
      <td>0.650520</td>
      <td>-6.669916</td>
      <td>121.100808</td>
      <td>176.312452</td>
    </tr>
    <tr>
      <th>Rock</th>
      <td>59.619392</td>
      <td>0.538292</td>
      <td>0.683670</td>
      <td>-7.285875</td>
      <td>122.629630</td>
      <td>176.185108</td>
    </tr>
    <tr>
      <th>Hip-Hop</th>
      <td>58.423131</td>
      <td>0.718808</td>
      <td>0.643275</td>
      <td>-6.860286</td>
      <td>120.791039</td>
      <td>173.715966</td>
    </tr>
    <tr>
      <th>Indie</th>
      <td>54.701561</td>
      <td>0.566821</td>
      <td>0.581002</td>
      <td>-7.915142</td>
      <td>119.290814</td>
      <td>167.225056</td>
    </tr>
    <tr>
      <th>Dance</th>
      <td>57.275256</td>
      <td>0.638191</td>
      <td>0.698067</td>
      <td>-6.054241</td>
      <td>120.795919</td>
      <td>173.353191</td>
    </tr>
    <tr>
      <th>R&amp;B</th>
      <td>52.308719</td>
      <td>0.642125</td>
      <td>0.564248</td>
      <td>-7.597064</td>
      <td>116.373834</td>
      <td>162.291862</td>
    </tr>
    <tr>
      <th>Alternative</th>
      <td>50.213430</td>
      <td>0.541898</td>
      <td>0.711519</td>
      <td>-6.540803</td>
      <td>122.534485</td>
      <td>167.460530</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Build popularity date frame

#Square popularity data to highlight the genre's differences
popularity_df = top_genres_data['popularity']**2
popularity_df = popularity_df.sort_values(ascending=False)
popularity_chart = popularity_df.plot(kind='bar', stacked=True, color='purple')

#Set the xlabel and ylabel using class methods
popularity_chart.set_xlabel("Genre")
popularity_chart.set_ylabel("Popularity Score")
popularity_chart.set_title("Genre Popularity")

#Save and display chart
plt.show()
plt.tight_layout()
```


![png](Spoitfy_Analysis_Draft_files/Spoitfy_Analysis_Draft_38_0.png)



    <Figure size 432x288 with 0 Axes>


Popularity: Let’s start with analyzing each genre’s popularity score. The popularity of a track is a value between 0 and 100, with 100 being the most popular. The popularity is calculated by an algorithm and is based, in the most part, on the total number of plays the track has had and how recent those plays are. Generally speaking, songs that are being played a lot now will have a higher popularity than songs that were played a lot in the past. The top 8 genres average popularity score ranged from 50 to 65. When we were analyzing each Genre’s average popularity, we decided to square each genre’s average popularity score to highlight the differences. As you can see, unsurprisingly, Pop is the most popular genre, followed by Rap and Rock.


```python
#Build danceability data frame

#Square danceability data to highlight the genre's differences
danceability_df = top_genres_data['danceability']**2
danceability_df = danceability_df.sort_values(ascending=False)
danceability_chart = danceability_df.plot(kind='bar', stacked=True, color='gold')

#Set the xlabel and ylabel using class methods
danceability_chart.set_xlabel("Genre")
danceability_chart.set_ylabel("Danceability Score")
danceability_chart.set_title("Genre Danceability")

#Save and display chart
plt.show()
plt.tight_layout()
```


![png](Spoitfy_Analysis_Draft_files/Spoitfy_Analysis_Draft_40_0.png)



    <Figure size 432x288 with 0 Axes>


Danceability: Moving on to danceability. Danceability describes how suitable a track is for dancing based on a combination of musical elements including tempo, rhythm stability, beat strength, and overall regularity. A value of 0.0 is least danceable and 1.0 is most danceable. The top 8 genres average danceability score ranged from 0.53 to 0.72. Just as in our popularity analysis, we squared each genre’s average popularity score to highlight the differences. Hip-Hop is the genre with the highest danceability, followed by Rap and R&B. It is interesting that the ‘Dance’ genre danceability score was in the middle of the pack. This could be because Dance music tends to lack rhythm stability and overall regularity, which are key elements of the danceability score. 



```python
#Build energy data frame

#Square energy data to highlight the genre's differences
energy_df = top_genres_data['energy']**2
energy_df = energy_df.sort_values(ascending=False)
energy_chart = energy_df.plot(kind='bar', stacked=True, color='orange')

#Set the xlabel and ylabel using class methods
energy_chart.set_xlabel("Genre")
energy_chart.set_ylabel("Energy Score")
energy_chart.set_title("Genre Energy")

#Save and display chart
plt.show()
plt.tight_layout()
```


![png](Spoitfy_Analysis_Draft_files/Spoitfy_Analysis_Draft_42_0.png)



    <Figure size 432x288 with 0 Axes>


Energy: Now on the energy analysis, like the danceability score, the energy score is a measure from 0.0 to 1.0 and represents a perceptual measure of intensity and activity. Perceptual features contributing to this attribute include dynamic range, timbre, and perceived loudness. For example, a death metal song would have a high energy score, while a Bach prelude scores low on the scale. The top 8 genres average energy score ranged from 0.56 to 0.71. We also squared each genre’s average energy score to highlight the differences. Alternative is the genre with the highest energy score, followed by Dance and Rock. This make sense because Alternative and Rock music tend to have wide dynamic ranges and high timbre. It also makes sense that the Dance genre had the second highest energy score because high perceived loudness is a significant element of Dance music.  



```python
#Build loudness data frame

#Do not square value because loudness is in decibals
loudness_df = top_genres_data['loudness'].sort_values()
loudness_chart = loudness_df.plot(kind='bar', stacked=True, color='pink')

#Set the xlabel and ylabel using class methods
loudness_chart.set_xlabel("Genre")
loudness_chart.set_ylabel("Loudness (decibals)")
loudness_chart.set_title("Genre Loudness")

#Save and display chart
plt.show()
plt.tight_layout()
```


![png](Spoitfy_Analysis_Draft_files/Spoitfy_Analysis_Draft_44_0.png)



    <Figure size 432x288 with 0 Axes>


Loudness: We also analyzed each genre’s loudness score in decibels. The overall loudness of a track in decibels (dB) directly correlates with track’s average amplitude. Amplitude measures the physical strength of sound. The closer a track’s is to 0 db, the louder it is. The top 8 genres average loudness ranged from -7.9 (db) to -6 (db). We did not square each genre’s average loudness score because this attribute is objectively measured in decibels, instead of being algorithmically calculated by Spotify. Dance is the loudest genre. This make sense because, as mentioned before, dance music has high perceived loudness. Conversely, Indie is the least loud genre due to it having a low perceived loudness. 



```python
#Build tempo data frame

#Do not square value because tempo is in BPM

tempo_df = top_genres_data['tempo'].sort_values(ascending = False) 
tempo_chart = tempo_df.plot(kind='bar', stacked=True, color='purple')

# Set the xlabel and ylabel using class methods
tempo_chart.set_xlabel("Genre")
tempo_chart.set_ylabel("Tempo (BPM)")
tempo_chart.set_title("Genre Tempo")

plt.show()
plt.tight_layout()
tempo_df
```


![png](Spoitfy_Analysis_Draft_files/Spoitfy_Analysis_Draft_46_0.png)





    genre
    Rock           122.629630
    Alternative    122.534485
    Pop            121.175844
    Rap            121.100808
    Dance          120.795919
    Hip-Hop        120.791039
    Indie          119.290814
    R&B            116.373834
    Name: tempo, dtype: float64




    <Figure size 432x288 with 0 Axes>


Tempo: Here, we looked at each Genre’s average tempo. Tempo is the speed or pace of a given track and derives directly from the track’s average beat duration. Tempo is measured in BPM (beats per minute). So, the higher the BPM, the faster a song’s pace. Musically, if a piece is 120-160 BPM, it is considered Allegro, which means the piece is fast and cheerful. The top 8 genres average tempo did not vary significantly from each other. The top 8 were all around 120 BPM. So, it is safe to say that these genres generally have upbeat, fast-paced tracks, which are ideal for a party setting. 



```python
#Build partybility data frame
#Divide by 10 and square the partybality values to highlight the difference. 
partybality_df = top_genres_data['partybality']/10
partybality_adjusted = partybality_df**2
partybality_adjusted = partybality_adjusted.sort_values(ascending = False) 
partybality_chart = partybality_adjusted.plot(kind='bar', stacked=True, color='green')

#Set the xlabel and ylabel using class methods
partybality_chart.set_xlabel("Genre")
partybality_chart.set_ylabel("Partybility Score")
partybality_chart.set_title("Genre Partybility")

#Save and display chart
plt.show()
plt.tight_layout()
```


![png](Spoitfy_Analysis_Draft_files/Spoitfy_Analysis_Draft_48_0.png)



    <Figure size 432x288 with 0 Axes>


# Partybility Conclusion 

Finally, we have the results after adding all the attributes’ average scores to calculate each genre’s partibility. We divided each genre’s partibility popularity score by 10 and then squared the results to highlight the differences. We decided to divide the score by 10 so that the squared results wouldn’t be too large. From this analysis, we determined that Pop is the best genre in a party setting, followed by Rap and rock. The results were very similar to the popularity score results. Both had Pop, Rap, and Rock at the top of the pack while Indie, R&B and Alternative were at the bottom. We can conclude that in a party setting, it is safe to go with the most popular genres at the time. 


# BILLBOARD HOT-100 (TOP SONGS RANK THROUGH TIME 2014-2019)

Using Billboard's Hot-100 chart we have created three graphs that exhibit a songs rank through time. With a clean dataset already provided, we used a function to find songs that were ranked No.1, groupby the songs to 'artist' and 'title' and plotted multiple songs into a graph that ranged from 2014-2019. Our goal here was to see which songs stayed at No.1 the longest and which genre dominated the charts in relation to popularity. 


```python
#Dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import dates as plt_dates
```


```python
#Read in CSV
bb1_csv = '../Resources/billboard_1_year.csv'
bb3_csv = '../Resources/billboard_3_year.csv'
bb5_csv = '../Resources/billboard_5_year.csv'
bb1_df = pd.read_csv(bb1_csv)
bb3_df = pd.read_csv(bb3_csv)
bb5_df = pd.read_csv(bb5_csv)
```


```python
#Defining a function to look for songs ranked #1
def single(df):
    try:
        if (df['rank'].isin([1]).any()):
            return df
    except KeyError:
        return None
```


```python
#Transform date column to number (specifically unix time for readable graph)
bb1_df['date'] = bb1_df['date'].transform(plt_dates.datestr2num)

#Groupby artist and title in order to determine groups
gb_songtitle = bb1_df.groupby(['artist','title'])
wasrankone_df = gb_songtitle.apply(single)
wasrankone_df = wasrankone_df.dropna()
wasrankone_df = wasrankone_df.groupby(['artist','title'])

wasrankone_df.size()
```




    artist                               title                                        
    Ariana Grande                        7 Rings                                          31
                                         Thank U, Next                                    28
    Billie Eilish                        Bad Guy                                          21
    Drake                                In My Feelings                                   14
    Halsey                               Without Me                                       46
    Jonas Brothers                       Sucker                                           25
    Lady Gaga & Bradley Cooper           Shallow                                          45
    Lil Nas X Featuring Billy Ray Cyrus  Old Town Road                                    25
    Maroon 5 Featuring Cardi B           Girls Like You                                   39
    Post Malone & Swae Lee               Sunflower (Spider-Man: Into The Spider-Verse)    44
    Shawn Mendes & Camila Cabello        Senorita                                          9
    Travis Scott                         Sicko Mode                                       49
    dtype: int64




```python
#Plotting songs for Top Songs in Aug 2018 - Sep 2019
plt.figure(figsize=(20,10))
for key, df in wasrankone_df:
    plt.plot(df['date'], df['rank'], label=key)
    
plt.legend(fontsize='small')
plt.gca().invert_yaxis()
plt.title('Top Songs: August 2018 - September 2019', fontsize='xx-large')
plt.xticks([], [])
plt.ylabel('Rank')

#Save and display chart
plt.savefig("../Resources/topsongsaug18_sep19.png", bbox_inches='tight')
plt.show()
```


![png](Spoitfy_Analysis_Draft_files/Spoitfy_Analysis_Draft_55_0.png)


ANALYSIS: In this graph we observe 12 songs and its movement from its starting position on the Hot-100 chart until it exits the chart. We see that most of the top songs stay between rank 1 and high 40s. For Drake's 'In My Feelings' the song had a strong start but plummeted from its No.1 spot pretty quickly. As for 'Old Town Road,' the song entered the billboard charts from a low ranking and quickly reached No.1 and remained there for 24 weeks. 'Old Town Road' is considered a mix of two genres country and hip-hop.


```python
# 3 Year dataset (extract specific dates)
bb3_df = bb3_df[(bb3_df['date'] > '2016-09-01') & (bb3_df['date'] < '2018-08-05')]
bb3_df = pd.DataFrame(bb3_df)
bb3_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>artist</th>
      <th>title</th>
      <th>lastPos</th>
      <th>rank</th>
      <th>weeks</th>
      <th>date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5600</th>
      <td>0</td>
      <td>Drake</td>
      <td>In My Feelings</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>2018-08-04</td>
    </tr>
    <tr>
      <th>5601</th>
      <td>1</td>
      <td>Cardi B, Bad Bunny &amp; J Balvin</td>
      <td>I Like It</td>
      <td>2</td>
      <td>2</td>
      <td>16</td>
      <td>2018-08-04</td>
    </tr>
    <tr>
      <th>5602</th>
      <td>2</td>
      <td>Maroon 5 Featuring Cardi B</td>
      <td>Girls Like You</td>
      <td>3</td>
      <td>3</td>
      <td>9</td>
      <td>2018-08-04</td>
    </tr>
    <tr>
      <th>5603</th>
      <td>3</td>
      <td>6ix9ine Featuring Nicki Minaj &amp; Murda Beatz</td>
      <td>FEFE</td>
      <td>0</td>
      <td>4</td>
      <td>1</td>
      <td>2018-08-04</td>
    </tr>
    <tr>
      <th>5604</th>
      <td>4</td>
      <td>Post Malone</td>
      <td>Better Now</td>
      <td>7</td>
      <td>5</td>
      <td>13</td>
      <td>2018-08-04</td>
    </tr>
    <tr>
      <th>5605</th>
      <td>5</td>
      <td>Drake</td>
      <td>Nice For What</td>
      <td>4</td>
      <td>6</td>
      <td>16</td>
      <td>2018-08-04</td>
    </tr>
    <tr>
      <th>5606</th>
      <td>6</td>
      <td>Ella Mai</td>
      <td>Boo'd Up</td>
      <td>5</td>
      <td>7</td>
      <td>17</td>
      <td>2018-08-04</td>
    </tr>
    <tr>
      <th>5607</th>
      <td>7</td>
      <td>Juice WRLD</td>
      <td>Lucid Dreams</td>
      <td>6</td>
      <td>8</td>
      <td>11</td>
      <td>2018-08-04</td>
    </tr>
    <tr>
      <th>5608</th>
      <td>8</td>
      <td>Post Malone Featuring Ty Dolla $ign</td>
      <td>Psycho</td>
      <td>9</td>
      <td>9</td>
      <td>22</td>
      <td>2018-08-04</td>
    </tr>
    <tr>
      <th>5609</th>
      <td>9</td>
      <td>Tyga Featuring Offset</td>
      <td>Taste</td>
      <td>13</td>
      <td>10</td>
      <td>9</td>
      <td>2018-08-04</td>
    </tr>
    <tr>
      <th>5610</th>
      <td>10</td>
      <td>Ariana Grande</td>
      <td>No Tears Left To Cry</td>
      <td>8</td>
      <td>11</td>
      <td>14</td>
      <td>2018-08-04</td>
    </tr>
    <tr>
      <th>5611</th>
      <td>11</td>
      <td>Taylor Swift</td>
      <td>Delicate</td>
      <td>12</td>
      <td>12</td>
      <td>20</td>
      <td>2018-08-04</td>
    </tr>
    <tr>
      <th>5612</th>
      <td>12</td>
      <td>Drake</td>
      <td>God's Plan</td>
      <td>10</td>
      <td>13</td>
      <td>27</td>
      <td>2018-08-04</td>
    </tr>
    <tr>
      <th>5613</th>
      <td>13</td>
      <td>Lil Baby &amp; Drake</td>
      <td>Yes Indeed</td>
      <td>16</td>
      <td>14</td>
      <td>11</td>
      <td>2018-08-04</td>
    </tr>
    <tr>
      <th>5614</th>
      <td>14</td>
      <td>Zedd, Maren Morris &amp; Grey</td>
      <td>The Middle</td>
      <td>14</td>
      <td>15</td>
      <td>26</td>
      <td>2018-08-04</td>
    </tr>
    <tr>
      <th>5615</th>
      <td>15</td>
      <td>Marshmello &amp; Anne-Marie</td>
      <td>Friends</td>
      <td>18</td>
      <td>16</td>
      <td>24</td>
      <td>2018-08-04</td>
    </tr>
    <tr>
      <th>5616</th>
      <td>16</td>
      <td>Ed Sheeran</td>
      <td>Perfect</td>
      <td>19</td>
      <td>17</td>
      <td>48</td>
      <td>2018-08-04</td>
    </tr>
    <tr>
      <th>5617</th>
      <td>17</td>
      <td>Khalid &amp; Normani</td>
      <td>Love Lies</td>
      <td>21</td>
      <td>18</td>
      <td>23</td>
      <td>2018-08-04</td>
    </tr>
    <tr>
      <th>5618</th>
      <td>18</td>
      <td>XXXTENTACION</td>
      <td>Sad!</td>
      <td>15</td>
      <td>19</td>
      <td>21</td>
      <td>2018-08-04</td>
    </tr>
    <tr>
      <th>5619</th>
      <td>19</td>
      <td>Bebe Rexha &amp; Florida Georgia Line</td>
      <td>Meant To Be</td>
      <td>17</td>
      <td>20</td>
      <td>40</td>
      <td>2018-08-04</td>
    </tr>
    <tr>
      <th>5620</th>
      <td>20</td>
      <td>Ariana Grande</td>
      <td>God Is A Woman</td>
      <td>11</td>
      <td>21</td>
      <td>2</td>
      <td>2018-08-04</td>
    </tr>
    <tr>
      <th>5621</th>
      <td>21</td>
      <td>Bazzi</td>
      <td>Mine</td>
      <td>22</td>
      <td>22</td>
      <td>27</td>
      <td>2018-08-04</td>
    </tr>
    <tr>
      <th>5622</th>
      <td>22</td>
      <td>Shawn Mendes</td>
      <td>In My Blood</td>
      <td>23</td>
      <td>23</td>
      <td>19</td>
      <td>2018-08-04</td>
    </tr>
    <tr>
      <th>5623</th>
      <td>23</td>
      <td>Dan + Shay</td>
      <td>Tequila</td>
      <td>25</td>
      <td>24</td>
      <td>20</td>
      <td>2018-08-04</td>
    </tr>
    <tr>
      <th>5624</th>
      <td>24</td>
      <td>5 Seconds Of Summer</td>
      <td>Youngblood</td>
      <td>32</td>
      <td>25</td>
      <td>8</td>
      <td>2018-08-04</td>
    </tr>
    <tr>
      <th>5625</th>
      <td>25</td>
      <td>Selena Gomez</td>
      <td>Back To You</td>
      <td>27</td>
      <td>26</td>
      <td>11</td>
      <td>2018-08-04</td>
    </tr>
    <tr>
      <th>5626</th>
      <td>26</td>
      <td>Calvin Harris &amp; Dua Lipa</td>
      <td>One Kiss</td>
      <td>26</td>
      <td>27</td>
      <td>16</td>
      <td>2018-08-04</td>
    </tr>
    <tr>
      <th>5627</th>
      <td>27</td>
      <td>YG Featuring 2 Chainz, Big Sean &amp; Nicki Minaj</td>
      <td>Big Bank</td>
      <td>39</td>
      <td>28</td>
      <td>8</td>
      <td>2018-08-04</td>
    </tr>
    <tr>
      <th>5628</th>
      <td>28</td>
      <td>Drake</td>
      <td>Nonstop</td>
      <td>20</td>
      <td>29</td>
      <td>4</td>
      <td>2018-08-04</td>
    </tr>
    <tr>
      <th>5629</th>
      <td>29</td>
      <td>The Carters</td>
      <td>Apes**t</td>
      <td>28</td>
      <td>30</td>
      <td>6</td>
      <td>2018-08-04</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>15570</th>
      <td>70</td>
      <td>Kanye West</td>
      <td>Father Stretch My Hands Pt. 1</td>
      <td>74</td>
      <td>71</td>
      <td>16</td>
      <td>2016-09-10</td>
    </tr>
    <tr>
      <th>15571</th>
      <td>71</td>
      <td>French Montana Featuring Drake</td>
      <td>No Shopping</td>
      <td>67</td>
      <td>72</td>
      <td>6</td>
      <td>2016-09-10</td>
    </tr>
    <tr>
      <th>15572</th>
      <td>72</td>
      <td>Billy Currington</td>
      <td>It Don't Hurt Like It Used To</td>
      <td>75</td>
      <td>73</td>
      <td>5</td>
      <td>2016-09-10</td>
    </tr>
    <tr>
      <th>15573</th>
      <td>73</td>
      <td>Eric Church</td>
      <td>Record Year</td>
      <td>64</td>
      <td>74</td>
      <td>19</td>
      <td>2016-09-10</td>
    </tr>
    <tr>
      <th>15574</th>
      <td>74</td>
      <td>Nicky Jam</td>
      <td>With You Tonight / Hasta El Amanecer</td>
      <td>73</td>
      <td>75</td>
      <td>14</td>
      <td>2016-09-10</td>
    </tr>
    <tr>
      <th>15575</th>
      <td>75</td>
      <td>Rob $tone Featuring J. Davi$ &amp; Spooks</td>
      <td>Chill Bill</td>
      <td>80</td>
      <td>76</td>
      <td>4</td>
      <td>2016-09-10</td>
    </tr>
    <tr>
      <th>15576</th>
      <td>76</td>
      <td>Selena Gomez</td>
      <td>Kill Em With Kindness</td>
      <td>68</td>
      <td>77</td>
      <td>13</td>
      <td>2016-09-10</td>
    </tr>
    <tr>
      <th>15577</th>
      <td>77</td>
      <td>PARTYNEXTDOOR Featuring Drake</td>
      <td>Come And See Me</td>
      <td>65</td>
      <td>78</td>
      <td>8</td>
      <td>2016-09-10</td>
    </tr>
    <tr>
      <th>15578</th>
      <td>78</td>
      <td>Frank Ocean</td>
      <td>Nikes</td>
      <td>0</td>
      <td>79</td>
      <td>1</td>
      <td>2016-09-10</td>
    </tr>
    <tr>
      <th>15579</th>
      <td>79</td>
      <td>Frank Ocean</td>
      <td>Ivy</td>
      <td>0</td>
      <td>80</td>
      <td>1</td>
      <td>2016-09-10</td>
    </tr>
    <tr>
      <th>15580</th>
      <td>80</td>
      <td>Justin Moore</td>
      <td>You Look Like I Need A Drink</td>
      <td>79</td>
      <td>81</td>
      <td>3</td>
      <td>2016-09-10</td>
    </tr>
    <tr>
      <th>15581</th>
      <td>81</td>
      <td>Young Thug And Travis Scott Featuring Quavo</td>
      <td>Pick Up The Phone</td>
      <td>90</td>
      <td>82</td>
      <td>2</td>
      <td>2016-09-10</td>
    </tr>
    <tr>
      <th>15582</th>
      <td>82</td>
      <td>Tucker Beathard</td>
      <td>Rock On</td>
      <td>98</td>
      <td>83</td>
      <td>2</td>
      <td>2016-09-10</td>
    </tr>
    <tr>
      <th>15583</th>
      <td>83</td>
      <td>Frank Ocean</td>
      <td>Pink + White</td>
      <td>0</td>
      <td>84</td>
      <td>1</td>
      <td>2016-09-10</td>
    </tr>
    <tr>
      <th>15584</th>
      <td>84</td>
      <td>Carrie Underwood</td>
      <td>Church Bells</td>
      <td>78</td>
      <td>85</td>
      <td>16</td>
      <td>2016-09-10</td>
    </tr>
    <tr>
      <th>15585</th>
      <td>85</td>
      <td>Chance The Rapper Featuring Lil Wayne &amp; 2 Chainz</td>
      <td>No Problem</td>
      <td>87</td>
      <td>86</td>
      <td>14</td>
      <td>2016-09-10</td>
    </tr>
    <tr>
      <th>15586</th>
      <td>86</td>
      <td>Lil Yachty</td>
      <td>1 Night</td>
      <td>94</td>
      <td>87</td>
      <td>3</td>
      <td>2016-09-10</td>
    </tr>
    <tr>
      <th>15587</th>
      <td>87</td>
      <td>Tove Lo</td>
      <td>Cool Girl</td>
      <td>0</td>
      <td>88</td>
      <td>2</td>
      <td>2016-09-10</td>
    </tr>
    <tr>
      <th>15588</th>
      <td>88</td>
      <td>DJ Khaled Featuring Nicki Minaj, Chris Brown, ...</td>
      <td>Do You Mind</td>
      <td>91</td>
      <td>89</td>
      <td>3</td>
      <td>2016-09-10</td>
    </tr>
    <tr>
      <th>15589</th>
      <td>89</td>
      <td>French Montana Featuring Kodak Black</td>
      <td>Lockjaw</td>
      <td>83</td>
      <td>90</td>
      <td>8</td>
      <td>2016-09-10</td>
    </tr>
    <tr>
      <th>15590</th>
      <td>90</td>
      <td>Rae Sremmurd</td>
      <td>Look Alive</td>
      <td>76</td>
      <td>91</td>
      <td>2</td>
      <td>2016-09-10</td>
    </tr>
    <tr>
      <th>15591</th>
      <td>91</td>
      <td>Enrique Iglesias Featuring Wisin</td>
      <td>Duele El Corazon</td>
      <td>88</td>
      <td>92</td>
      <td>7</td>
      <td>2016-09-10</td>
    </tr>
    <tr>
      <th>15592</th>
      <td>92</td>
      <td>DJ Drama Featuring Chris Brown, Skeme &amp; Lyquin</td>
      <td>Wishing</td>
      <td>95</td>
      <td>93</td>
      <td>2</td>
      <td>2016-09-10</td>
    </tr>
    <tr>
      <th>15593</th>
      <td>93</td>
      <td>DJ ESCO Featuring Future &amp; Lil Uzi Vert</td>
      <td>Too Much Sauce</td>
      <td>0</td>
      <td>94</td>
      <td>1</td>
      <td>2016-09-10</td>
    </tr>
    <tr>
      <th>15594</th>
      <td>94</td>
      <td>Shawn Mendes</td>
      <td>Mercy</td>
      <td>0</td>
      <td>95</td>
      <td>1</td>
      <td>2016-09-10</td>
    </tr>
    <tr>
      <th>15595</th>
      <td>95</td>
      <td>Frank Ocean</td>
      <td>Solo</td>
      <td>0</td>
      <td>96</td>
      <td>1</td>
      <td>2016-09-10</td>
    </tr>
    <tr>
      <th>15596</th>
      <td>96</td>
      <td>Lil Uzi Vert</td>
      <td>You Was Right</td>
      <td>92</td>
      <td>97</td>
      <td>7</td>
      <td>2016-09-10</td>
    </tr>
    <tr>
      <th>15597</th>
      <td>97</td>
      <td>Frank Ocean</td>
      <td>Nights</td>
      <td>0</td>
      <td>98</td>
      <td>1</td>
      <td>2016-09-10</td>
    </tr>
    <tr>
      <th>15598</th>
      <td>98</td>
      <td>William Michael Morgan</td>
      <td>I Met A Girl</td>
      <td>0</td>
      <td>99</td>
      <td>1</td>
      <td>2016-09-10</td>
    </tr>
    <tr>
      <th>15599</th>
      <td>99</td>
      <td>Luke Bryan</td>
      <td>Move</td>
      <td>0</td>
      <td>100</td>
      <td>1</td>
      <td>2016-09-10</td>
    </tr>
  </tbody>
</table>
<p>10000 rows × 7 columns</p>
</div>




```python
#Defining a function to look for songs ranked #1
def single(df):
    try:
        if (df['rank'].isin([1]).any()):
            return df
    except KeyError:
        return None
```


```python
#Transform date column to number (specifically unix time for readable graph)
bb3_df['date'] = bb3_df['date'].transform(plt_dates.datestr2num)

#Groupby artist and title in order to determine groups
gb_songtitle = bb3_df.groupby(['artist','title'])
rankone_df = gb_songtitle.apply(single)
rankone_df = rankone_df.dropna()
rankone_df = rankone_df.groupby(['artist','title'])

rankone_df.size()
```




    artist                                                                   title                     
    Bruno Mars                                                               That's What I Like            51
    Camila Cabello Featuring Young Thug                                      Havana                        45
    Cardi B                                                                  Bodak Yellow (Money Moves)    34
    Cardi B, Bad Bunny & J Balvin                                            I Like It                     16
    Childish Gambino                                                         This Is America               12
    DJ Khaled Featuring Justin Bieber, Quavo, Chance The Rapper & Lil Wayne  I'm The One                   22
    Drake                                                                    God's Plan                    27
                                                                             In My Feelings                 4
                                                                             Nice For What                 16
    Ed Sheeran                                                               Perfect                       47
                                                                             Shape Of You                  58
    Kendrick Lamar                                                           Humble.                       37
    Luis Fonsi & Daddy Yankee Featuring Justin Bieber                        Despacito                     51
    Migos Featuring Lil Uzi Vert                                             Bad And Boujee                36
    Post Malone Featuring 21 Savage                                          Rockstar                      40
    Post Malone Featuring Ty Dolla $ign                                      Psycho                        22
    Rae Sremmurd Featuring Gucci Mane                                        Black Beatles                 27
    Taylor Swift                                                             Look What You Made Me Do      19
    The Chainsmokers Featuring Halsey                                        Closer                        49
    The Weeknd Featuring Daft Punk                                           Starboy                       30
    XXXTENTACION                                                             Sad!                          21
    dtype: int64




```python
#Plotting songs for Top Songs in Aug 2016 - Sep 2018
plt.figure(figsize=(20,10))
for key, df in rankone_df:
    plt.plot(df['date'], df['rank'], label=key)

plt.legend(fontsize='small', loc='center left', bbox_to_anchor=(1, 0.5))
plt.gca().invert_yaxis()
plt.title('Top Songs: August 2018 - September 2016', fontsize='xx-large')
plt.xticks([], [])
plt.ylabel('Rank')

#Save and display chart
plt.savefig("../Resources/topsongsaug18_sep16.png", bbox_inches='tight')
plt.show()
```


![png](Spoitfy_Analysis_Draft_files/Spoitfy_Analysis_Draft_60_0.png)


ANALYSIS: This graph displays the top songs for the course of 2 years. Due to its longer year range, there are more top songs that can be observed. Most of these songs are categorized in the pop, rap and hip-hop genre (particularly in this graph most songs are either rap/hip-hop). 'Despacito'(genre: latin pop) stayed at No.1 the longest compared to other songs. Ed Sheeran's 'Perfect', began in the low 40s, plummeted to high 90s but suddenly climbed back up and stayed at the No.1 spot for 6 weeks.


```python
#5 Year Dataset (extract specific dates)
bb5_df = bb5_df[(bb5_df['date'] > '2014-09-05') & (bb5_df['date'] < '2016-08-10')]
bb5_df = pd.DataFrame(bb5_df)
bb5_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>artist</th>
      <th>title</th>
      <th>lastPos</th>
      <th>rank</th>
      <th>weeks</th>
      <th>date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>16000</th>
      <td>0</td>
      <td>Sia Featuring Sean Paul</td>
      <td>Cheap Thrills</td>
      <td>3</td>
      <td>1</td>
      <td>23</td>
      <td>2016-08-06</td>
    </tr>
    <tr>
      <th>16001</th>
      <td>1</td>
      <td>Drake Featuring WizKid &amp; Kyla</td>
      <td>One Dance</td>
      <td>1</td>
      <td>2</td>
      <td>16</td>
      <td>2016-08-06</td>
    </tr>
    <tr>
      <th>16002</th>
      <td>2</td>
      <td>Calvin Harris Featuring Rihanna</td>
      <td>This Is What You Came For</td>
      <td>4</td>
      <td>3</td>
      <td>12</td>
      <td>2016-08-06</td>
    </tr>
    <tr>
      <th>16003</th>
      <td>3</td>
      <td>Justin Timberlake</td>
      <td>Can't Stop The Feeling!</td>
      <td>2</td>
      <td>4</td>
      <td>11</td>
      <td>2016-08-06</td>
    </tr>
    <tr>
      <th>16004</th>
      <td>4</td>
      <td>The Chainsmokers Featuring Daya</td>
      <td>Don't Let Me Down</td>
      <td>5</td>
      <td>5</td>
      <td>23</td>
      <td>2016-08-06</td>
    </tr>
    <tr>
      <th>16005</th>
      <td>5</td>
      <td>twenty one pilots</td>
      <td>Ride</td>
      <td>6</td>
      <td>6</td>
      <td>19</td>
      <td>2016-08-06</td>
    </tr>
    <tr>
      <th>16006</th>
      <td>6</td>
      <td>Rihanna</td>
      <td>Needed Me</td>
      <td>7</td>
      <td>7</td>
      <td>25</td>
      <td>2016-08-06</td>
    </tr>
    <tr>
      <th>16007</th>
      <td>7</td>
      <td>Kent Jones</td>
      <td>Don't Mind</td>
      <td>9</td>
      <td>8</td>
      <td>12</td>
      <td>2016-08-06</td>
    </tr>
    <tr>
      <th>16008</th>
      <td>8</td>
      <td>Desiigner</td>
      <td>Panda</td>
      <td>8</td>
      <td>9</td>
      <td>22</td>
      <td>2016-08-06</td>
    </tr>
    <tr>
      <th>16009</th>
      <td>9</td>
      <td>Adele</td>
      <td>Send My Love (To Your New Lover)</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
      <td>2016-08-06</td>
    </tr>
    <tr>
      <th>16010</th>
      <td>10</td>
      <td>Katy Perry</td>
      <td>Rise</td>
      <td>0</td>
      <td>11</td>
      <td>1</td>
      <td>2016-08-06</td>
    </tr>
    <tr>
      <th>16011</th>
      <td>11</td>
      <td>P!nk</td>
      <td>Just Like Fire</td>
      <td>11</td>
      <td>12</td>
      <td>14</td>
      <td>2016-08-06</td>
    </tr>
    <tr>
      <th>16012</th>
      <td>12</td>
      <td>Fifth Harmony Featuring Ty Dolla $ign</td>
      <td>Work From Home</td>
      <td>12</td>
      <td>13</td>
      <td>21</td>
      <td>2016-08-06</td>
    </tr>
    <tr>
      <th>16013</th>
      <td>13</td>
      <td>Meghan Trainor</td>
      <td>Me Too</td>
      <td>15</td>
      <td>14</td>
      <td>10</td>
      <td>2016-08-06</td>
    </tr>
    <tr>
      <th>16014</th>
      <td>14</td>
      <td>Florida Georgia Line</td>
      <td>H.O.L.Y.</td>
      <td>14</td>
      <td>15</td>
      <td>12</td>
      <td>2016-08-06</td>
    </tr>
    <tr>
      <th>16015</th>
      <td>15</td>
      <td>Mike Posner</td>
      <td>I Took A Pill In Ibiza</td>
      <td>13</td>
      <td>16</td>
      <td>27</td>
      <td>2016-08-06</td>
    </tr>
    <tr>
      <th>16016</th>
      <td>16</td>
      <td>Britney Spears Featuring G-Eazy</td>
      <td>Make Me...</td>
      <td>0</td>
      <td>17</td>
      <td>1</td>
      <td>2016-08-06</td>
    </tr>
    <tr>
      <th>16017</th>
      <td>17</td>
      <td>Drake</td>
      <td>Controlla</td>
      <td>16</td>
      <td>18</td>
      <td>12</td>
      <td>2016-08-06</td>
    </tr>
    <tr>
      <th>16018</th>
      <td>18</td>
      <td>DJ Khaled Featuring Drake</td>
      <td>For Free</td>
      <td>17</td>
      <td>19</td>
      <td>7</td>
      <td>2016-08-06</td>
    </tr>
    <tr>
      <th>16019</th>
      <td>19</td>
      <td>Drake Featuring Rihanna</td>
      <td>Too Good</td>
      <td>19</td>
      <td>20</td>
      <td>12</td>
      <td>2016-08-06</td>
    </tr>
    <tr>
      <th>16020</th>
      <td>20</td>
      <td>Shawn Mendes</td>
      <td>Treat You Better</td>
      <td>20</td>
      <td>21</td>
      <td>7</td>
      <td>2016-08-06</td>
    </tr>
    <tr>
      <th>16021</th>
      <td>21</td>
      <td>twenty one pilots</td>
      <td>Heathens</td>
      <td>23</td>
      <td>22</td>
      <td>5</td>
      <td>2016-08-06</td>
    </tr>
    <tr>
      <th>16022</th>
      <td>22</td>
      <td>Rihanna Featuring Drake</td>
      <td>Work</td>
      <td>21</td>
      <td>23</td>
      <td>26</td>
      <td>2016-08-06</td>
    </tr>
    <tr>
      <th>16023</th>
      <td>23</td>
      <td>Ariana Grande</td>
      <td>Into You</td>
      <td>33</td>
      <td>24</td>
      <td>10</td>
      <td>2016-08-06</td>
    </tr>
    <tr>
      <th>16024</th>
      <td>24</td>
      <td>Flume Featuring Kai</td>
      <td>Never Be Like You</td>
      <td>26</td>
      <td>25</td>
      <td>16</td>
      <td>2016-08-06</td>
    </tr>
    <tr>
      <th>16025</th>
      <td>25</td>
      <td>Lukas Graham</td>
      <td>7 Years</td>
      <td>22</td>
      <td>26</td>
      <td>27</td>
      <td>2016-08-06</td>
    </tr>
    <tr>
      <th>16026</th>
      <td>26</td>
      <td>Charlie Puth Featuring Selena Gomez</td>
      <td>We Don't Talk Anymore</td>
      <td>35</td>
      <td>27</td>
      <td>7</td>
      <td>2016-08-06</td>
    </tr>
    <tr>
      <th>16027</th>
      <td>27</td>
      <td>Fifth Harmony Featuring Fetty Wap</td>
      <td>All In My Head (Flex)</td>
      <td>29</td>
      <td>28</td>
      <td>5</td>
      <td>2016-08-06</td>
    </tr>
    <tr>
      <th>16028</th>
      <td>28</td>
      <td>twenty one pilots</td>
      <td>Stressed Out</td>
      <td>25</td>
      <td>29</td>
      <td>44</td>
      <td>2016-08-06</td>
    </tr>
    <tr>
      <th>16029</th>
      <td>29</td>
      <td>DJ Khaled Featuring Jay Z &amp; Future</td>
      <td>I Got The Keys</td>
      <td>34</td>
      <td>30</td>
      <td>4</td>
      <td>2016-08-06</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>25970</th>
      <td>70</td>
      <td>Childish Gambino</td>
      <td>V. 3005</td>
      <td>75</td>
      <td>71</td>
      <td>13</td>
      <td>2014-09-13</td>
    </tr>
    <tr>
      <th>25971</th>
      <td>71</td>
      <td>Migos</td>
      <td>Fight Night</td>
      <td>69</td>
      <td>72</td>
      <td>8</td>
      <td>2014-09-13</td>
    </tr>
    <tr>
      <th>25972</th>
      <td>72</td>
      <td>Brad Paisley</td>
      <td>River Bank</td>
      <td>65</td>
      <td>73</td>
      <td>17</td>
      <td>2014-09-13</td>
    </tr>
    <tr>
      <th>25973</th>
      <td>73</td>
      <td>Iggy Azalea</td>
      <td>Work</td>
      <td>73</td>
      <td>74</td>
      <td>18</td>
      <td>2014-09-13</td>
    </tr>
    <tr>
      <th>25974</th>
      <td>74</td>
      <td>Maddie &amp; Tae</td>
      <td>Girl In A Country Song</td>
      <td>74</td>
      <td>75</td>
      <td>4</td>
      <td>2014-09-13</td>
    </tr>
    <tr>
      <th>25975</th>
      <td>75</td>
      <td>Joe Nichols</td>
      <td>Yeah</td>
      <td>70</td>
      <td>76</td>
      <td>19</td>
      <td>2014-09-13</td>
    </tr>
    <tr>
      <th>25976</th>
      <td>76</td>
      <td>Blake Shelton</td>
      <td>Neon Light</td>
      <td>81</td>
      <td>77</td>
      <td>2</td>
      <td>2014-09-13</td>
    </tr>
    <tr>
      <th>25977</th>
      <td>77</td>
      <td>Little Big Town</td>
      <td>Day Drinking</td>
      <td>82</td>
      <td>78</td>
      <td>6</td>
      <td>2014-09-13</td>
    </tr>
    <tr>
      <th>25978</th>
      <td>78</td>
      <td>Michael Jackson &amp; Justin Timberlake</td>
      <td>Love Never Felt So Good</td>
      <td>76</td>
      <td>79</td>
      <td>18</td>
      <td>2014-09-13</td>
    </tr>
    <tr>
      <th>25979</th>
      <td>79</td>
      <td>Mr. Probz</td>
      <td>Waves</td>
      <td>88</td>
      <td>80</td>
      <td>2</td>
      <td>2014-09-13</td>
    </tr>
    <tr>
      <th>25980</th>
      <td>80</td>
      <td>Kid Ink Featuring Chris Brown</td>
      <td>Main Chick</td>
      <td>71</td>
      <td>81</td>
      <td>16</td>
      <td>2014-09-13</td>
    </tr>
    <tr>
      <th>25981</th>
      <td>81</td>
      <td>Beyonce Featuring Nicki Minaj Or Chimamanda Ng...</td>
      <td>***Flawless</td>
      <td>0</td>
      <td>82</td>
      <td>1</td>
      <td>2014-09-13</td>
    </tr>
    <tr>
      <th>25982</th>
      <td>82</td>
      <td>Rich Homie Quan Featuring Problem</td>
      <td>Walk Thru</td>
      <td>99</td>
      <td>83</td>
      <td>2</td>
      <td>2014-09-13</td>
    </tr>
    <tr>
      <th>25983</th>
      <td>83</td>
      <td>Hozier</td>
      <td>Take Me To Church</td>
      <td>87</td>
      <td>84</td>
      <td>3</td>
      <td>2014-09-13</td>
    </tr>
    <tr>
      <th>25984</th>
      <td>84</td>
      <td>Rita Ora</td>
      <td>I Will Never Let You Down</td>
      <td>77</td>
      <td>85</td>
      <td>5</td>
      <td>2014-09-13</td>
    </tr>
    <tr>
      <th>25985</th>
      <td>85</td>
      <td>Tyga Featuring Young Thug</td>
      <td>Hookah</td>
      <td>85</td>
      <td>86</td>
      <td>3</td>
      <td>2014-09-13</td>
    </tr>
    <tr>
      <th>25986</th>
      <td>86</td>
      <td>Eli Young Band</td>
      <td>Dust</td>
      <td>90</td>
      <td>87</td>
      <td>6</td>
      <td>2014-09-13</td>
    </tr>
    <tr>
      <th>25987</th>
      <td>87</td>
      <td>Eric Church</td>
      <td>Cold One</td>
      <td>89</td>
      <td>88</td>
      <td>4</td>
      <td>2014-09-13</td>
    </tr>
    <tr>
      <th>25988</th>
      <td>88</td>
      <td>Usher Featuring Nicki Minaj</td>
      <td>She Came To Give It To You</td>
      <td>0</td>
      <td>89</td>
      <td>1</td>
      <td>2014-09-13</td>
    </tr>
    <tr>
      <th>25989</th>
      <td>89</td>
      <td>Billy Currington</td>
      <td>We Are Tonight</td>
      <td>84</td>
      <td>90</td>
      <td>19</td>
      <td>2014-09-13</td>
    </tr>
    <tr>
      <th>25990</th>
      <td>90</td>
      <td>Keith Urban</td>
      <td>Somewhere In My Car</td>
      <td>0</td>
      <td>91</td>
      <td>1</td>
      <td>2014-09-13</td>
    </tr>
    <tr>
      <th>25991</th>
      <td>91</td>
      <td>The Swon Brothers</td>
      <td>Later On</td>
      <td>97</td>
      <td>92</td>
      <td>3</td>
      <td>2014-09-13</td>
    </tr>
    <tr>
      <th>25992</th>
      <td>92</td>
      <td>Steve Aoki, Chris Lake &amp; Tujamo Featuring Kid Ink</td>
      <td>Delirious (Boneless)</td>
      <td>96</td>
      <td>93</td>
      <td>2</td>
      <td>2014-09-13</td>
    </tr>
    <tr>
      <th>25993</th>
      <td>93</td>
      <td>Ariana Grande Featuring Big Sean</td>
      <td>Best Mistake</td>
      <td>0</td>
      <td>94</td>
      <td>2</td>
      <td>2014-09-13</td>
    </tr>
    <tr>
      <th>25994</th>
      <td>94</td>
      <td>Maroon 5</td>
      <td>It Was Always You</td>
      <td>0</td>
      <td>95</td>
      <td>2</td>
      <td>2014-09-13</td>
    </tr>
    <tr>
      <th>25995</th>
      <td>95</td>
      <td>Usher</td>
      <td>Good Kisser</td>
      <td>91</td>
      <td>96</td>
      <td>17</td>
      <td>2014-09-13</td>
    </tr>
    <tr>
      <th>25996</th>
      <td>96</td>
      <td>TeeFLii Featuring 2 Chainz</td>
      <td>24 Hours</td>
      <td>0</td>
      <td>97</td>
      <td>1</td>
      <td>2014-09-13</td>
    </tr>
    <tr>
      <th>25997</th>
      <td>97</td>
      <td>Chris Young</td>
      <td>Who I Am With You</td>
      <td>95</td>
      <td>98</td>
      <td>19</td>
      <td>2014-09-13</td>
    </tr>
    <tr>
      <th>25998</th>
      <td>98</td>
      <td>Trey Songz</td>
      <td>Foreign</td>
      <td>98</td>
      <td>99</td>
      <td>9</td>
      <td>2014-09-13</td>
    </tr>
    <tr>
      <th>25999</th>
      <td>99</td>
      <td>Pharrell Williams</td>
      <td>Come Get It Bae</td>
      <td>93</td>
      <td>100</td>
      <td>15</td>
      <td>2014-09-13</td>
    </tr>
  </tbody>
</table>
<p>10000 rows × 7 columns</p>
</div>




```python
#Defining a function to look for songs ranked #1
def single(df):
    try:
        if (df['rank'].isin([1]).any()):
            return df
    except KeyError:
        return None

```


```python
#Transform date column to number (specifically unix time for readable graph)
bb5_df['date'] = bb5_df['date'].transform(plt_dates.datestr2num)

#Groupby artist and title in order to determine groups
gb_songtitle = bb5_df.groupby(['artist','title'])
rankone5_df = gb_songtitle.apply(single)
rankone5_df = rankone5_df.dropna()
rankone5_df = rankone5_df.groupby(['artist','title'])

rankone5_df.size()
```




    artist                                 title                  
    Adele                                  Hello                      26
    Desiigner                              Panda                      22
    Drake Featuring WizKid & Kyla          One Dance                  16
    Justin Bieber                          Love Yourself              36
                                           Sorry                      39
                                           What Do You Mean?          31
    Justin Timberlake                      Can't Stop The Feeling!    11
    Mark Ronson Featuring Bruno Mars       Uptown Funk!               56
    Meghan Trainor                         All About That Bass        40
    OMI                                    Cheerleader                35
    Rihanna Featuring Drake                Work                       26
    Sia Featuring Sean Paul                Cheap Thrills              23
    Taylor Swift                           Blank Space                36
                                           Shake It Off               49
    Taylor Swift Featuring Kendrick Lamar  Bad Blood                  25
    The Weeknd                             Can't Feel My Face         41
                                           The Hills                  48
    Wiz Khalifa Featuring Charlie Puth     See You Again              52
    Zayn                                   Pillowtalk                 24
    dtype: int64




```python
#Plotting songs for Top Songs in Aug 2014 - Sep 2016
plt.figure(figsize=(20,10))
for key, df in rankone5_df:
    plt.plot(df['date'], df['rank'], label=key)
    
plt.legend(fontsize='small', loc='center left', bbox_to_anchor=(1, 0.5))
plt.gca().invert_yaxis()
plt.title('Top Songs: August 2016 - September 2014', fontsize='xx-large')
plt.xticks([], [])
plt.ylabel('Rank')

#Save and display chart
plt.savefig("../Resources/topsongsaug16_sep14.png", bbox_inches='tight')
plt.show()
```


![png](Spoitfy_Analysis_Draft_files/Spoitfy_Analysis_Draft_65_0.png)


ANALYSIS: In this graph we see again another shift in top charting genres. There are more hip-hop and pop songs in this graph rather than the previous one that leaned more towards rap and hip-hop. Zayn's 'Pillowtalk' debuted at No.1 but quickly descended after one week. However, 'Pillowtalk' is not within the usual pop/rap/hip-hop genre. It is categorized as Alternative R&B and electronic. 'Uptown Funk' was the longest song in the No.1 spot even with its mix of funk and pop genre. It got replaced with 'See You Again' which is categorized as a mix of the pop and rap genre.

CONCLUSION: From the 3 graphs obtained from Billboard's Hot-100 chart, it is evident that Pop, Hip-Hop and Rap are the most dominating genres throughout 2014-2019. However, it is undeniable that Pop is the top genre for all 5 years. This confirms our spotify popular genre analysis that exhibit Pop as the leading genre with a 20.9% score. Hip-hop and Rap follow behind with 20% and 20.5%.

# Billboard Analysis - Average Length of Rank 1

Using the `billboard.py` wrapper, we consolidated 1 year, 3 years, and 5 years worth of weekly "Hot 100" charts into csv's containing each chart's top 100 songs and their rankings. Since each chart is ordered by, or can be grouped by, date, we will be making pycharts that dig deep into the highly sought after Rank 1. In this iPython Notebook, we will be analyzing the distribution of the lengths of time that songs are ranked 1.

Combined below are three plots: a distribution for a song's ranking as number 1 over one year and five years by weeks and a histogram describing the normality of the 5 year plot.

The distribution in the first plot describes the 12 songs most popular in the past year and shows us how popular those 12 songs were relative to each other. The most popular song of the year, "Old Town Road" by Lil Nas X, was the most popular song of the past year. Outside reading shows that this song broke records as it stayed number 1 on the charts for 19 straight weeks. However, as a song that broke records, it is a clear outlier in the first year.

The distribution in the second plot describes the past 5 years in a similar way, but with much more data. With 51 data points, we can see the clear bundling of songs around 5 weeks, the mean of the data. Note the difference in the means between 1-year data and 5-year data, around 15%. The 5-year data was definitely has less spread than the 1-year data, but that decreased spread raised the mean by a small percentage. Further statistical analysis can reveal which points are outliers, but for that we need to test for normality.

The last plot tests for normality over the past five years by binning each result by week. We expect to see a bell curve set up around 5 weeks. However, we can clearly see our data is not normal. The orange line indictes an exponential regression to fit the data, and though it fits the best, it does not accurately describe our data. Normality in events tends to be the result of randomness or through slight perturbations in initial conditions. Our data is skewed because it is biased. Song popularity is complex, and what is lasting is dependent on the people listening. 

# 1-year and 5-year Scatter plots and 5-year Histogram


```python
# Dependencies

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.optimize import curve_fit
import numpy as np

%matplotlib inline

# Import billboard data

charts_1 = pd.read_csv("./Resources/billboard_1_year.csv") # 1 Year data
charts_3 = pd.read_csv("./Resources/billboard_3_year.csv") # 3 Year data
charts_5 = pd.read_csv("./Resources/billboard_5_year.csv") # 5 Year data

# Function Definitions

def exp_reg(x,a,b):
    return a*np.exp(-b*x)
```


```python
### Plotting ###

fig = plt.figure(figsize=(12,9))
gs = fig.add_gridspec(3,3)
ax1 = fig.add_subplot(gs[0,:])
ax2 = fig.add_subplot(gs[1,:])
ax3 = fig.add_subplot(gs[2,1])

## ax1 - year 1 demo ##

    # Group-by rank, then get group for rank 1 ("1")
charts_1_gb = charts_1.groupby("rank")
rankone_1 = charts_1_gb.get_group(1)

    # Define group-by for artist and title for rankone_1
rankone_1_gb = rankone_1.groupby(['artist', 'title'])

    # Calculate the amount of weeks each artist-title pair has via count()
rankone_1_frac = rankone_1_gb.count().reset_index().drop(columns=['Unnamed: 0'])

    # Mean and Std Deviation
weeks_bar = rankone_1_frac['weeks'].mean()
sigma = rankone_1_frac['weeks'].std()

ax1.hlines(y=weeks_bar,xmin=0, xmax=11, ls="dotted") # Mean
ax1.fill_between(x=np.arange(12), y1=weeks_bar-2*sigma, y2=weeks_bar+2*sigma, color='blue', alpha=.10)
ax1.fill_between(x=np.arange(12), y1=weeks_bar-sigma, y2=weeks_bar+sigma, color='blue', alpha=.25)

    # Labels and Plots
labels = []
for t in rankone_1_frac.itertuples():
    ax1.scatter(x=[t[0]], y=[t[4]], edgecolor='k')
    labels.append(f"{t[2]}")

labels[9] = 'Sunflower'

ax1.set_xlim((-1,12))
ax1.set_xticklabels([])
ax1.grid(axis='x', alpha=.5, color='k')

ax1.set_ylabel("Weeks in Rank 1", fontsize="x-large")
ax1.set_xlabel(f"Song Index ($\mu = {weeks_bar:.3f}, \sigma = {sigma:.3f}$)", fontsize='x-large')

## ax2 - 5 year demo ##

    # Group-by rank, pull 1
rankone_5 = charts_5.groupby('rank').get_group(1)

    # Group-by (artist,title): count weeks
rankone_5 = rankone_5.groupby(['artist', 'title']).count()

    # x_axis values
x5 = np.arange(rankone_5['weeks'].size)

    # y_axis values
percents_5 = rankone_5['weeks'].to_numpy()

    # Mean and std deviation
rank_mu = rankone_5['weeks'].mean()
rank_sigma = rankone_5['weeks'].std()

    # Standard deviation
ax2.hlines(y=rank_mu ,xmin=0, xmax=np.amax(x5), ls="dotted")
ax2.fill_between(x=x5, y1=rank_mu-2*rank_sigma, y2=rank_mu+2*rank_sigma, color='blue', alpha=.10)
ax2.fill_between(x=x5, y1=rank_mu-rank_sigma, y2=rank_mu+rank_sigma, color='blue', alpha=.25)

    # Plotting
ax2.scatter(x5, percents_5, edgecolor='k', label='Year 4-5', color='gold', s=50)

ax2.set_xlabel(f'Song Index ($\mu = {rank_mu:.3f}$, $\sigma = {rank_sigma:.3f}$)', size='x-large')
ax2.set_ylabel('Weeks in Rank 1', size='x-large')
ax2.set_xticks([])
ax2.grid(axis='x', alpha=.5, color='k')

## ax3 - histogram demo ##

    # Histogram
n, bins, patches = ax3.hist(percents_5, bins=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], edgecolor='k')
    
    # Regression
x = np.arange(1,20,1)
popt, pcov = curve_fit(exp_reg, x, n)

ax3.plot(x, exp_reg(x, *popt))
ax3.set_xlabel("Weeks (1 per Bin)", fontsize='x-large')
ax3.set_ylabel("Number of Songs", fontsize='x-large')

# Final plt.show() and savefig()
ax1.set_title("Distribution of Songs in Rank 1 by Year", fontsize='xx-large')
plt.tight_layout()

plt.savefig('./rank1_distribution.png')
plt.show()
```


![png](Spoitfy_Analysis_Draft_files/Spoitfy_Analysis_Draft_73_0.png)


# Conlcusions


Our analysis of our Spotify Dataset and Billboard Data left us with 4 lasting impressions:
We cannot base a song’s popularity based on acoustic features.
Our classification of Partyability generated some promise in using acoustic features to categorize songs.
Billboard ‘Hot-100’ charts tell us Pop, Hip-Hop, and Rap were the most popular genres in the past 5 years.
On average a song is ranked 1 for 5 weeks. Most songs last 1-2 weeks ranked 1. Few songs are popular for more than couple months. 
