import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from wordcloud import ImageColorGenerator
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from queue import PriorityQueue as pq
import string
import re

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

st.write("""
# An Analytic Approach to Popular Movie Statistics
""")

# Load datasets
movie_ratings = pd.read_csv("datasets/filmtv_movies - ENG.csv")
netflix_titles = pd.read_csv("datasets/netflix_titles_nov_2019.csv")

# Merge datasets
merged = movie_ratings.merge(netflix_titles, left_on=['title','year'], right_on=['title','release_year'])
merged_copy = merged.copy()
merged_copy = merged_copy.drop(["filmtv_id", "country_x","actors","directors","total_votes","show_id","director","cast","country_y","date_added","release_year","duration_y"], axis=1)

# Process categories
category_cols = []
for cat in list(set(merged['listed_in'])):
    cats = cat.strip("'").split(",")
    for ca in cats:
        if ca not in category_cols:
            category_cols.append(ca.strip())
category_cols = [cat.strip() for cat in category_cols]
category_set = list(set(category_cols))

data = merged.copy()
for col in category_set:
    data[col] = data['listed_in'].apply(lambda x: 1 if col in x else 0)

numerical_columns = data.select_dtypes(include=np.number).columns.tolist()

# Wordcloud for genres
st.write("### What genres are represented the most?")
text = " ".join(i for i in merged.listed_in)
wordcloud = WordCloud(background_color="white").generate(text)
fig, ax = plt.subplots()
ax.imshow(wordcloud, interpolation='bilinear')
ax.axis("off")
st.pyplot(fig)

# Average ratings by genre
genre_cat = st.selectbox('Genre', category_set)
numerical_column = 'avg_vote'

output = data.groupby(genre_cat)[numerical_column].mean().reset_index()
output[genre_cat] = output[genre_cat].apply(lambda x: "Selected genre" if x == 1 else "Other genre")

st.write("### Let's see the average ratings for a specific genre")

genre_vs_col = alt.Chart(output).mark_bar().encode(
    x=str(genre_cat),
    y=numerical_column,
    tooltip=["avg_vote"]
).properties(
    width=650,
    height=500,
    title=f"Average Ratings of {genre_cat} Categorized Films vs Non {genre_cat} Categorized Films"
).interactive()

st.altair_chart(genre_vs_col)

# Critic vs public ratings
cr_pv = alt.Chart(data).mark_circle().encode(
    alt.X('critics_vote', bin=True, scale=alt.Scale(zero=False)),
    alt.Y('public_vote', bin=True),
    size='count()',
    color='genre',
    tooltip=['genre','critics_vote','public_vote','count()']
).properties(
    title="How do Critic Ratings Compare to Public Ratings?",
    width=200,
    height=650
).interactive()

st.altair_chart(cr_pv, use_container_width=True)
st.write("#### It seems like critic and public ratings tend to be fairly similar!")

# Relationship between chosen features
st.write("### Let's look at the relationship between any two chosen features")
x1 = st.selectbox('X1', data.columns, index=2)
y1 = st.selectbox('Y1', [i for i in data.columns if i != x1], index=3)

year_avg_vote = alt.Chart(data).mark_point().encode(
    alt.X(x1, scale=alt.Scale(zero=False)),
    y=y1,
    color='genre',
    tooltip=['title','avg_vote']
).properties(
    title=f"{x1} vs {y1}",
    width=650,
    height=500
).interactive()

st.altair_chart(year_avg_vote)

# Count of records by genre and year
df = data.groupby(by=["year", "genre"]).size().reset_index(name="counts")
genre_year = alt.Chart(data).mark_bar().encode(
    x='year',
    y='count(genre)',
    color='genre',
    tooltip=['year','genre','count()']
).properties(
    title="Count of Records by Genre and Year",
    width=650,
    height=500
).interactive()

st.altair_chart(genre_year)

# Movie Recommendations
st.markdown("# Movie Recommendations")
st.sidebar.markdown("# Movie recommendations")

text = open("datasets/review_ratings.csv", encoding="utf-8")
output = open("datasets/res.txt", "w", encoding="utf-8")
text.readline()  # remove header

for row in text:
    row_details = row.split('^')
    title, year, genre, duration = row_details[1].strip(' ,"\''), row_details[2].strip(' ,"\''), row_details[3].strip(' ,"\''), row_details[4].strip(' ,"\''')
    critic_rate, pub_rate = row_details[6].strip(' ,"\''), row_details[7].strip(' ,"\''')
    desc, notes, listed_in, comment = row_details[8].strip(' ,"\''), row_details[9].strip(' ,"\''), row_details[16].strip(' ,"\''), row_details[17].strip(' ,"\''')
    alpha_rate = row_details[16].strip(" ,"\'")

    combined_desc = desc + ' ' + notes + ' ' + listed_in + ' ' + comment + ' ' + title + ' ' + genre
    formatted_str = title + '\t' + str(year) + '\t' + str(critic_rate) + '\t' + str(pub_rate) + '\t' + combined_desc + '\n'
    output.write(formatted_str)

output.close()
text.close()

punc = string.punctuation
films = {}
text = open("datasets/review_ratings.csv", encoding="utf-8")
text.readline()  # remove header

for row in text:
    row_details = row.split('^')
    title, year, genre, duration = row_details[1].strip(' ,"\''), row_details[2].strip(' ,"\''), row_details[3].strip(' ,"\''), row_details[4].strip(' ,"\''')
    critic_rate, pub_rate = row_details[6].strip(' ,"\''), row_details[7].strip(' ,"\''')
    desc, notes, listed_in, comment = row_details[8].strip(' ,"\''), row_details[9].strip(' ,"\''), row_details[16].strip(' ,"\''), row_details[17].strip(' ,"\''')
    alpha_rate = row_details[16].strip(' ,"\'')

    combined_desc = desc + ' ' + notes + ' ' + listed_in + ' ' + comment + ' ' + title + ' ' + genre

    valid_words = ""
    for word in combined_desc:
        if word == " ":
            valid_words += word
        elif word in punc:
            pass
        elif word.isalpha() == False:
            pass
        else:
            valid_words += word
    films[(title, year)] = valid_words

text.close()

st.write("There are", len(films), "films in this dataset")
films_df = pd.DataFrame.from_dict(films, orient="index")
films_df.columns = ['desc']

st.dataframe(films_df.head(5))

# Remove line breaks
def remove_linebreaks(input):
    text = re.compile(r'\n')
    return text.sub(r' ', input)

films_df["desc"] = films_df["desc"].apply(remove_linebreaks)

# Tokenize words
films_df["desc"] = films_df["desc"].apply(word_tokenize)

# Remove stopwords
def remove_stopwords(input1):
    words = []
    for word in input1:
        if word not in stopwords.words('english'):
            words.append(word)
    return words

films_df["desc"] = films_df["desc"].apply(remove_stopwords)

# Lemmatization
lem = WordNetLemmatizer()
def lemma_wordnet(input):
    return [lem.lemmatize(w) for w in input]

films_df["desc"] = films_df["desc"].apply(lemma_wordnet)

def combine_text(input):
    combined = ' '.join(input)
    return combined

films_df["desc"] = films_df["desc"].apply(combine_text)

# Train a TF-IDF vectorizer
vectorizer = TfidfVectorizer(max_features=1800, lowercase=True, stop_words='english', ngram_range=(1,2))

# Matrix of all features (TF-IDF matrix)
X = vectorizer.fit_transform(films_df["desc"])

# Train NMF model to get topics
nmf_model = NMF(n_components=10)
nmf_model.fit(X)

# Top words associated with topics
for index, topic in enumerate(nmf_model.components_):
    st.write(f'The top 10 words for topic #{index}')
    st.write([vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-10:]])

topic_results = nmf_model.transform(X)

films_df["Topic"] = topic_results.argmax(axis=1)

st.write(films_df.head(5))

# Define the recommendation function
def movie_recommender(title, year):
    title, year = title.lower(), year.lower()
    movie = films_df.loc[(title, year)]
    movie_topic = movie['Topic']
    recommendation = films_df[films_df['Topic'] == movie_topic].sort_values('pub_rate', ascending=False)
    return recommendation

# Select a movie to get recommendations
title = st.text_input("Enter a movie title:")
year = st.text_input("Enter the movie's year of release:")

if st.button("Get Recommendations"):
    recommendations = movie_recommender(title, year)
    st.write("Recommendations:")
    st.dataframe(recommendations)

# Visualizing movie data
desc_length = films_df['desc'].apply(len)
critics_rating = data['critics_vote']
public_rating = data['public_vote']

plt.figure(figsize=(10, 6))
plt.hist(desc_length, bins=50, alpha=0.5, label='Description Length')
plt.hist(critics_rating, bins=50, alpha=0.5, label='Critics Rating')
plt.hist(public_rating, bins=50, alpha=0.5, label='Public Rating')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.legend(loc='upper right')
plt.title('Distribution of Description Length, Critics Rating, and Public Rating')
st.pyplot(plt)
