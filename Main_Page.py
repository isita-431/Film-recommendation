from pickle import encode_long
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from wordcloud import ImageColorGenerator
import nltk
import os
import string
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.metrics.pairwise import linear_kernel
from queue import PriorityQueue as pq

# Define a custom directory for NLTK data
nltk_data_dir = 'nltk_data'
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir)

nltk.data.path.append(nltk_data_dir)

# Download necessary NLTK data
nltk.download('punkt', download_dir=nltk_data_dir)
nltk.download('stopwords', download_dir=nltk_data_dir)
nltk.download('wordnet', download_dir=nltk_data_dir)
nltk.download('omw-1.4', download_dir=nltk_data_dir)

st.write("""
# An Analytic Approach to Popular Movie Statistics
""")

movie_ratings = pd.read_csv("datasets/filmtv_movies - ENG.csv")
netflix_titles = pd.read_csv("datasets/netflix_titles_nov_2019.csv")

merged = movie_ratings.merge(netflix_titles, left_on=['title', 'year'], right_on=['title', 'release_year'])
merged_copy = merged.copy()
merged_copy = merged_copy.drop(["filmtv_id", "country_x", "actors", "directors", "total_votes", "show_id", "director", "cast", "country_y", "date_added", "release_year", "duration_y"], axis=1)

category_cols = []
for cat in list(set(merged['listed_in'])):
    cats = cat.strip("'").split(",")
    for ca in cats:
        if ca in category_cols:
            pass
        else:
            category_cols.append(ca.strip())

category_cols = [cat.strip() for cat in category_cols]
category_set = list(set(category_cols))

data = merged.copy()
for col in category_set:
    data[col] = data['listed_in'].apply(lambda x: 1 if col in x else 0)

numerical_columns = data.select_dtypes(include=np.number).columns.tolist()

st.write("### What genres are represented the most? ")
text = " ".join(i for i in merged.listed_in)
wordcloud = WordCloud(background_color="white").generate(text)
fig, ax = plt.subplots()
ax.imshow(wordcloud, interpolation='bilinear')
ax.axis("off")
st.pyplot(fig)

genre_cat = st.selectbox('Genre', category_set)
numerical_column = 'avg_vote'  # st.selectbox('Select a numeric column', numerical_columns)

output = data.groupby(genre_cat)[numerical_column].mean()
output = output.reset_index()  # can we rename the first column?
output[genre_cat] = output[genre_cat].apply(lambda x: "Selected genre" if x == 1 else "Other genre")

st.write("### Let's see the average ratings for a specific genre")

genre_vs_col = alt.Chart(output).mark_bar().encode(
    x=str(genre_cat),
    y=numerical_column,
    tooltip=["avg_vote"]
).properties(
    width=650,
    height=500,
    title="Average Ratings of " + str(genre_cat) + " Categorized Films vs Non " + str(genre_cat) + " Categorized Films"
).interactive()

st.altair_chart(genre_vs_col)

cr_pv = alt.Chart(data).mark_circle().encode(
    alt.X('critics_vote', bin=True, scale=alt.Scale(zero=False)),
    alt.Y('public_vote', bin=True),
    size='count()',
    color='genre',
    tooltip=['genre', 'critics_vote', 'public_vote', 'count()']
).properties(
    title="How do Critic Ratings Compare to Public Ratings?",
    width=200,
    height=650
).interactive()

st.altair_chart(cr_pv, use_container_width=True)
st.write("#### It seems like critic and public ratings tend to be fairly similar!")

st.write("### Let's look at the relationship between any two chosen features")
x1 = st.selectbox('X1', data.columns, index=2)
y1 = st.selectbox('Y1', [i for i in data.columns if i != x1], index=3)

year_avg_vote = alt.Chart(data).mark_point().encode(
    alt.X(x1, scale=alt.Scale(zero=False)),
    y=y1,
    color='genre',
    tooltip=['title', 'avg_vote']
).properties(
    title=str(x1) + str(" vs ") + str(y1),
    width=650,
    height=500
).interactive()

st.altair_chart(year_avg_vote)

df = data.groupby(by=["year", "genre"]).size().reset_index(name="counts")
genre_year = alt.Chart(data).mark_bar().encode(
    x='year',
    y='count(genre)',
    color='genre',
    tooltip=['year', 'genre', 'count()']
).properties(
    title="Count of Records by Genre and Year",
    width=650,
    height=500
).interactive()

st.altair_chart(genre_year)

st.markdown("# Movie Recommendations")
st.sidebar.markdown("# Movie recommendations")

text = open("datasets/review_ratings.csv", encoding="utf-8")
output = open("datasets/res.txt", "w", encoding="utf-8")

text.readline()  # remove header
for row in text:
    row_details = row.split('^')
    title, year, genre, duration = row_details[1].strip(' ,"\''), row_details[2].strip(' ,"\''), row_details[3].strip(' ,"\''), row_details[4].strip(' ,"\'')
    critic_rate, pub_rate = row_details[6].strip(' ,"\''), row_details[7].strip(' ,"\'')
    desc, notes, listed_in, comment = row_details[8].strip(' ,"\''), row_details[9].strip(' ,"\''), row_details[16].strip(' ,"\''), row_details[17].strip(' ,"\'')
    alpha_rate = row_details[16].strip(' ,"\'')

    combined_desc = desc + ' ' + notes + ' ' + listed_in + ' ' + comment + ' ' + title + ' ' + genre
    formatted_str = title + '\t' + str(year) + ('\t') + str(critic_rate) + '\t' + str(pub_rate) + '\t' + combined_desc + '\n'
    output.write(formatted_str)

output.close()
text.close()

punc = string.punctuation
films = {}
text = open("datasets/review_ratings.csv", encoding="utf-8")
text.readline()  # remove header
for row in text:
    row_details = row.split('^')
    title, year, genre, duration = row_details[1].strip(' ,"\''), row_details[2].strip(' ,"\''), row_details[3].strip(' ,"\''), row_details[4].strip(' ,"\'')
    critic_rate, pub_rate = row_details[6].strip(' ,"\''), row_details[7].strip(' ,"\'')
    desc, notes, listed_in, comment = row_details[8].strip(' ,"\''), row_details[9].strip(' ,"\''), row_details[16].strip(' ,"\''), row_details[17].strip(' ,"\'')
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
nltk.download('punkt')
films_df["desc"] = films_df["desc"].apply(word_tokenize)

# Remove stopwords
nltk.download('stopwords')
def remove_stopwords(input1):
    words = []
    for word in input1:
        if word not in stopwords.words('english'):
            words.append(word)
    return words
films_df["desc"] = films_df["desc"].apply(remove_stopwords)

# Lemmatization
nltk.download('wordnet')
nltk.download('omw-1.4')
lem = WordNetLemmatizer()
def lemma_wordnet(input):
    return [lem.lemmatize(w) for w in input]
films_df["desc"] = films_df["desc"].apply(lemma_wordnet)

def combine_text(input):
    combined = ' '.join(input)
    return combined
films_df["desc"] = films_df["desc"].apply(combine_text)

# TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(films_df["desc"])

num_topics = 5
nmf = NMF(n_components=num_topics, random_state=1).fit(tfidf_matrix)

def top_keywords_per_topic(model, feature_names, n_words=10):
    keywords = {}
    for topic_idx, topic in enumerate(model.components_):
        top_keywords_idx = topic.argsort()[-n_words:][::-1]
        top_keywords = [feature_names[i] for i in top_keywords_idx]
        keywords[topic_idx] = top_keywords
    return keywords

feature_names = tfidf_vectorizer.get_feature_names_out()
top_keywords = top_keywords_per_topic(nmf, feature_names)

for topic, keywords in top_keywords.items():
    st.write(f"**Topic {topic + 1}:** {', '.join(keywords)}")

topic_assignments = nmf.transform(tfidf_matrix)
films_df['topic'] = np.argmax(topic_assignments, axis=1)

st.write("### Movie Recommendations")
user_input = st.text_input("Enter movie description for recommendation")

def recommend_movies(input_desc, films_df, nmf_model, tfidf_vectorizer):
    input_vec = tfidf_vectorizer.transform([input_desc])
    topic_dist = nmf_model.transform(input_vec)
    topic = np.argmax(topic_dist)
    st.write(f"Recommended Topic: {topic + 1}")

    # Find movies in the same topic
    recommendations = films_df[films_df['topic'] == topic].sort_values(by='desc')
    return recommendations

if user_input:
    recommendations = recommend_movies(user_input, films_df, nmf, tfidf_vectorizer)
    st.dataframe(recommendations[['desc']].head(10))
