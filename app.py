import streamlit as st
import feedparser
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Function to fetch news
def fetch_news(url):
    feed = feedparser.parse(url)
    news_items = []
    for entry in feed.entries:
        title = entry.title if 'title' in entry else 'No Title Available'
        summary = entry.summary if 'summary' in entry else entry.description if 'description' in entry else 'No Summary Available'
        link = entry.link if 'link' in entry else 'No Link Available'
        news_items.append({'title': title, 'summary': summary, 'link': link})
    return news_items

# Parse the CNN RSS feed
url = 'http://rss.cnn.com/rss/cnn_topstories.rss'
news_articles = fetch_news(url)

# Convert to DataFrame
df = pd.DataFrame(news_articles)

# Setup TF-IDF Vectorizer
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df['summary'])

# Streamlit app interface
st.title('News Recommender System')
selected_title = st.selectbox('Select a news article:', df['title'])

# Recommendation function
def get_recommendations(title, df, tfidf_matrix):
    index = df[df['title'] == title].index[0]
    cosine_sim = cosine_similarity(tfidf_matrix[index:index+1], tfidf_matrix).flatten()
    indices = cosine_sim.argsort()[-3:-1][::-1]  # Get top 2 indices
    recommended_titles = df.iloc[indices]
    return recommended_titles

# Show recommendations
if selected_title:
    recommendations = get_recommendations(selected_title, df, tfidf_matrix)
    st.subheader('Recommended Articles:')
    for index, row in recommendations.iterrows():
        st.write(f"[{row['title']}]({row['link']})")