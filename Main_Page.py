
from pickle import encode_long
import streamlit as st

import pandas as pd
import numpy as np
import altair as alt
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from wordcloud import ImageColorGenerator


st.write("""
# An Analytic Approach to Popular Movie Statistics
""")



movie_ratings = pd.read_csv("datasets/filmtv_movies - ENG.csv")

netflix_titles = pd.read_csv("datasets/netflix_titles_nov_2019.csv")

merged = movie_ratings.merge(netflix_titles, left_on=['title','year'], right_on=['title','release_year'])

merged_copy = merged.copy()
merged_copy = merged_copy.drop(["filmtv_id", "country_x","actors","directors","total_votes","show_id","director","cast","country_y","date_added","release_year","duration_y"], axis=1)

category_cols = []
for cat in list(set(merged['listed_in'])):
    cats = cat.strip("'").split(",")
    for ca in cats:
        if ca in category_cols:
            pass
        else:
            category_cols.append(ca.strip())

category_cols=[cat.strip() for cat in category_cols]
# category_cols.sort()
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
numerical_column = 'avg_vote'#st.selectbox('Select a numeric column', numerical_columns)

output = data.groupby(genre_cat)[numerical_column].mean()
output = output.reset_index() # can we rename the first column? 
output[genre_cat] = output[genre_cat].apply(lambda x: "Selected genre" if x==1 else "Other genre") 


st.write("### Let's see the average ratings for a specific genre")

genre_vs_col = alt.Chart(output).mark_bar().encode(
    x=str(genre_cat),
    y=numerical_column,
    tooltip = ["avg_vote"]
).properties(
    width=650,
    height=500,
    title="Average Ratings of "+str(genre_cat)+" Categorized Films vs Non "+str(genre_cat) + " Categorized Films"
).interactive()

st.altair_chart(genre_vs_col)


cr_pv = alt.Chart(data).mark_circle().encode(
    alt.X('critics_vote', bin=True, scale=alt.Scale(zero=False)),
    alt.Y('public_vote', bin=True),
    size='count()',
    color='genre',
    # color=alt.Color('genre', legend=alt.Legend(
    #     orient='none',
    #     legendX=520, legendY=0,
    #     direction='vertical',
    #     titleAnchor='middle')),
    tooltip=['genre','critics_vote','public_vote','count()']
).properties(
    title= "How do Critic Ratings Compare to Public Ratings?",
    width=200,
    height=650
).interactive()

st.altair_chart(cr_pv, use_container_width=True)
st.write("#### It seems like critic and public ratings tend to be fairly similar! ")

st.write("### Let's look at the relationship between any two chosen features")
x1 = st.selectbox('X1', data.columns, index=2)

y1 =  st.selectbox('Y1',[i for i in  data.columns if i!=x1],index=3)


year_avg_vote = alt.Chart(data).mark_point().encode(
    alt.X(x1,
        scale=alt.Scale(zero=False)
    ),
    # x=x1,
    y=y1,
    color='genre',
    tooltip=['title','avg_vote']
).properties(
    title= str(x1) + str(" vs ") + str(y1),
    width=650,
    height=500
).interactive()

st.altair_chart(year_avg_vote)
#st.write("#### Older movies in this dataset tend to have a varied runtime compared to recently released movies, with a majority of them running from 70 to 140 minutes ")





df = data.groupby(by=["year", "genre"]).size().reset_index(name="counts")
genre_year = alt.Chart(data).mark_bar().encode(
    x='year',
    y='count(genre)',
    color = 'genre',
    tooltip=['year','genre','count()']
).properties(
    title="Count of Records by Genre and Year",
    width=650,
    height=500
).interactive()

st.altair_chart(genre_year)

import pandas as pd 
import nltk
import numpy as np
import string
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.metrics.pairwise import linear_kernel
from queue import PriorityQueue as pq

import streamlit as st

st.markdown("# Movie Recommendations")
st.sidebar.markdown("# Movie recommendations")

text = open("datasets/review_ratings.csv",encoding="utf-8")
output = open("datasets/res.txt","w",encoding="utf-8")
# print(text)
text.readline() # remove header
print(text.readline())
for row in text: 
    row_details = row.split('^')
    title, year, genre, duration = row_details[1].strip(' ,"\''), row_details[2].strip(' ,"\''), row_details[3].strip(' ,"\''), row_details[4].strip(' ,"\'')
    critic_rate, pub_rate = row_details[6].strip(' ,"\''), row_details[7].strip(' ,"\'')
    desc, notes, listed_in, comment = row_details[8].strip(' ,"\''),row_details[9].strip(' ,"\''),row_details[16].strip(' ,"\''),row_details[17].strip(' ,"\'')
    alpha_rate = row_details[16].strip(' ,"\'')

    
    combined_desc = desc+' '+ notes+' '+listed_in+' ' +comment+' ' +title+' '+genre
    formatted_str = title+'\t'+str(year)+('\t')+str(critic_rate)+'\t'+str(pub_rate)+'\t'+combined_desc+'\n'
    output.write(formatted_str)
    # print(desc)

output.close()
text.close()

punc = string.punctuation
films = {}
text = open("datasets/review_ratings.csv",encoding="utf-8")
# print(text)
text.readline() # remove header
for row in text: 
    row_details = row.split('^')
    title, year, genre, duration = row_details[1].strip(' ,"\''), row_details[2].strip(' ,"\''), row_details[3].strip(' ,"\''), row_details[4].strip(' ,"\'')
    critic_rate, pub_rate = row_details[6].strip(' ,"\''), row_details[7].strip(' ,"\'')
    desc, notes, listed_in, comment = row_details[8].strip(' ,"\''),row_details[9].strip(' ,"\''),row_details[16].strip(' ,"\''),row_details[17].strip(' ,"\'')
    alpha_rate = row_details[16].strip(' ,"\'')

    combined_desc = desc+' '+ notes+' '+listed_in+' ' +comment+' ' +title+' '+genre

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
            
    # formatted_str = title+'\t'+str(year)+('\t')+str(critic_rate)+'\t'+str(pub_rate)+'\t'+combined_desc+'\n'

text.close()
# films

st.write("There are",len(films),"films in this dataset")
films_df = pd.DataFrame.from_dict(films,orient="index")
films_df.columns = ['desc']

st.dataframe(films_df.head(5))

# Remove line breaks 
def remove_linebreaks(input):
    text = re.compile(r'\n')
    return text.sub(r' ',input)

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

# Train a TF-IDF vectorizer 
vectorizer = TfidfVectorizer(max_features=1800, lowercase=True, stop_words='english', ngram_range=(1,2)) #unigrams and bigrams

# features 
tf_idf_output = vectorizer.fit_transform(films_df["desc"])
vocab = np.array(vectorizer.get_feature_names())

input_keyword = st.text_input("Enter the keywords relating a movie you want to watch", 'boy romance girl crush love')

st.write("Applying NMF")

num_topics = st.number_input("Pick no of topics", 50)

num_topics = int(num_topics)
nmf = NMF(n_components = num_topics, solver = "mu").fit(tf_idf_output)
W = nmf.fit_transform(tf_idf_output)
H = nmf.components_

num_top_words = st.number_input("Pick no of top words", 20)

num_top_words = int(num_top_words)
def show_topics(a):
    top_words = lambda t: [vocab[i] for i in np.argsort(t)[:-num_top_words-1:-1]]
    topic_words = ([top_words(t) for t in a])
    return [' '.join(t) for t in topic_words]
topics = show_topics(H)
i = 1

topics_set_list = []
for row in topics: 
    topic = row.split()
    #print("Topic",i,":",set(topic))
    topics_set_list.append(set(topic))
    i += 1

st.write("### Here are some example topics that we specifically found within our dataset")
for i in range(3): 
    st.write("Topic",i+1,":",topics_set_list[i])

st.write("Applying jaccard distance algorithm")

def jaccard_distance(topic_set, desc_set):
    intersect = len(topic_set.intersection(desc_set))
    union = len(topic_set.union(desc_set))
    return 1-intersect/union
    # return len(topic_set.intersection(desc_set))

input_keywords_to_set = set(input_keyword.split())
best_topic_score = 1
best_topics = None
for topic in topics_set_list: 
    # print(topic, jaccard_distance(topic, input_keywords_to_set))
    if jaccard_distance(topic, input_keywords_to_set) < best_topic_score: 
        # print(jaccard_distance(topic, input_keywords_to_set),best_topic_score)
        best_topic_score = jaccard_distance(topic, input_keywords_to_set) 
        best_topics = topic

st.write("Best similar topics", best_topics)

pq_best_titles = pq()
# max_d = 0
for k, v in films.items(): 
    jd = jaccard_distance(best_topics, set(v.split()))
    pq_best_titles.put((jd, k))

num_recs = 0
out_df = pd.DataFrame()
#out_df.columns = ['Title', 'Release Year']
title = []
year = [] 
while pq_best_titles.empty() == False and num_recs < 5:
    #print("Title:",pq_best_titles.get()[1][0],"\tRelease Year:",pq_best_titles.get()[1][1])
    #curr = {'Title': pq_best_titles.get()[1][0], 'Release Year' : pq_best_titles.get()[1][1] }
    title.append(pq_best_titles.get()[1][0])
    year.append(pq_best_titles.get()[1][1])
    num_recs += 1 
out_df['Title'] = title
out_df['Release Year'] = year
st.table(out_df)

    
