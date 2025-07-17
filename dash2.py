# dashboard_final_complete.py

import streamlit as st
import pandas as pd
import re
from collections import Counter, defaultdict
import spacy
import numpy as np
import google.generativeai as genai
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import networkx as nx
import plotly.graph_objects as go
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from nltk.corpus import stopwords

# --- Page Configuration ---
st.set_page_config(
    page_title="Project Drishti | Geopolitical Analysis",
    page_icon="🇮🇳",
    layout="wide"
)

# --- ⚙️ YOUR GEMINI API KEY ---
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY_HERE"

# --- Caching Functions for Performance ---
@st.cache_data
def load_and_process_data(file_path):
    """Loads, cleans, and performs initial analysis on the dataset."""
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        return None
    
    df['full_text'] = df['title'].astype(str) + ' ' + df['text'].astype(str)
    df['created_utc'] = pd.to_datetime(df['created_utc'])

    sia = SentimentIntensityAnalyzer()
    df['sentiment_score'] = df['full_text'].astype(str).apply(lambda x: sia.polarity_scores(x)['compound'])
    
    def get_sentiment_label(score):
        if score >= 0.05: return 'Positive'
        elif score <= -0.05: return 'Negative'
        else: return 'Neutral'
    df['sentiment_label'] = df['sentiment_score'].apply(get_sentiment_label)
    
    def clean_text(text):
        text = str(text).lower()
        text = re.sub(r'https://\S+|www\.\S+', '', text)
        text = re.sub(r'<.*?>|\[.*?\]|&[a-z]+;', '', text)
        text = re.sub(r'[^a-z\s]', '', text)
        return text
    df['cleaned_text'] = df['full_text'].apply(clean_text)
    
    return df

@st.cache_resource
def load_spacy_model():
    """Loads the spaCy model once."""
    return spacy.load("en_core_web_sm")

nlp = load_spacy_model()

# --- All Analysis Helper Functions ---
stop_words = list(stopwords.words('english'))
stop_words.extend(['army', 'indian', 'pla', 'people', 'liberation', 'china', 'chinese', 'would', 'like', 'get', 'it', 'also', 'us', 'one'])

def get_top_items(corpus, item_type='ner', n=10):
    if corpus.dropna().empty: return []
    if item_type == 'ner':
        all_items = []
        for text in corpus.dropna():
            doc = nlp(text)
            for ent in doc.ents:
                if ent.label_ in ['GPE', 'ORG'] and ent.text.lower() not in stop_words and len(ent.text) > 2:
                    all_items.append(ent.text.title())
        return Counter(all_items).most_common(n)
    elif item_type == 'ngram':
        try:
            vectorizer = CountVectorizer(ngram_range=(2, 2), stop_words=stop_words).fit(corpus.dropna())
            bag_of_words = vectorizer.transform(corpus.dropna())
            sum_words = bag_of_words.sum(axis=0)
            words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
            return sorted(words_freq, key=lambda x: x[1], reverse=True)[:n]
        except ValueError: return []
    return []

def get_topic_label_from_gemini(keywords, api_key):
    if not api_key or api_key == "YOUR_GEMINI_API_KEY_HERE": return "AI Labeling Disabled"
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = f"You are a geopolitical analyst. Based on these keywords from an online discussion topic, create a short, descriptive title (4-5 words max).\n\nKeywords: {', '.join(keywords)}\n\nDescriptive Title:"
        response = model.generate_content(prompt)
        return response.text.strip().replace("*", "")
    except Exception: return "[Gemini API Error]"

def analyze_topic_intensity(df_group):
    if df_group.dropna(subset=['cleaned_text']).empty: return pd.DataFrame()
    n_topics = 5
    vectorizer = CountVectorizer(max_df=0.9, min_df=5, stop_words=stop_words)
    try:
        doc_term_matrix = vectorizer.fit_transform(df_group['cleaned_text'].dropna())
    except ValueError: return pd.DataFrame()
    if doc_term_matrix.shape[0] < n_topics: return pd.DataFrame(columns=['Post Count', 'score', 'num_comments', 'Avg. Sentiment'])
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(doc_term_matrix)
    df_group_copy = df_group.dropna(subset=['cleaned_text']).copy()
    topic_predictions = lda.transform(doc_term_matrix)
    df_group_copy['topic'] = np.argmax(topic_predictions, axis=1)
    topic_labels = {}
    feature_names = vectorizer.get_feature_names_out()
    for topic_idx, topic in enumerate(lda.components_):
        top_words = [feature_names[i] for i in topic.argsort()[:-8 - 1:-1]]
        gemini_label = get_topic_label_from_gemini(top_words, GEMINI_API_KEY)
        topic_labels[topic_idx] = f"{gemini_label} ({', '.join(top_words[:3])}...)"
    df_group_copy['topic_label'] = df_group_copy['topic'].map(topic_labels)
    frequency = df_group_copy['topic_label'].value_counts().rename("Post Count")
    engagement = df_group_copy.groupby('topic_label')[['score', 'num_comments']].mean().round(1)
    sentiment = df_group_copy.groupby('topic_label')['sentiment_score'].mean().rename("Avg. Sentiment").round(2)
    dashboard = pd.concat([frequency, engagement, sentiment], axis=1)
    return dashboard.sort_values(by="Post Count", ascending=False)

def generate_network_html(filtered_data, edge_threshold):
    if filtered_data.empty: return None
    co_occurrences, entity_counts = defaultdict(int), Counter()
    sample_size = min(len(filtered_data), 1500)
    for doc in nlp.pipe(filtered_data['full_text'].sample(sample_size, random_state=1), disable=["parser"]):
        entities = list(set([ent.text.lower() for ent in doc.ents if ent.label_ in ['GPE', 'ORG'] and len(ent.text) > 3]))
        for entity in entities: entity_counts[entity] += 1
        for i in range(len(entities)):
            for j in range(i + 1, len(entities)):
                co_occurrences[tuple(sorted((entities[i], entities[j])))] += 1
    G = nx.Graph()
    top_entities = {entity for entity, count in entity_counts.most_common(100)}
    for entity in top_entities: G.add_node(entity, size=entity_counts[entity], title=f"{entity.title()}\nMentions: {entity_counts[entity]}")
    for pair, weight in co_occurrences.items():
        if pair[0] in G and pair[1] in G and weight > edge_threshold: G.add_edge(pair[0], pair[1], weight=weight)
    if len(G.nodes()) < 2: return None
    pos = nx.spring_layout(G, k=0.8, iterations=50)
    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]; x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None]); edge_y.extend([y0, y1, None])
    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.5, color='#888'), hoverinfo='none', mode='lines')
    node_x, node_y, node_text, node_size = [], [], [], []
    for node in G.nodes():
        x, y = pos[node]; node_x.append(x); node_y.append(y)
        node_text.append(G.nodes[node].get('title', '')); node_size.append(G.nodes[node].get('size', 1) * 3)
    node_trace = go.Scatter(x=node_x, y=node_y, mode='markers', hoverinfo='text', text=node_text,
                            marker=dict(showscale=True, colorscale='YlGnBu', reversescale=True, color=[], size=node_size,
                                        colorbar=dict(thickness=15, title='Node Connections'), line_width=2))
    node_adjacencies = [len(adj[1]) for adj in G.adjacency()]
    node_trace.marker.color = node_adjacencies
    fig = go.Figure(data=[edge_trace, node_trace],
                 layout=go.Layout(title='<br>Interactive Network Graph of Co-mentioned Entities', showlegend=False, hovermode='closest',
                                  margin=dict(b=20,l=5,r=5,t=40),
                                  xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                  yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))
    return fig

# --- Main App ---
st.title("🇮🇳 Project Drishti: Geopolitical Narrative Dashboard")
df_master = load_and_process_data('reddit_data.csv')

if df_master is None:
    st.error("Dataset 'reddit_data.csv' not found. Please run the `reddit_collector.py` script first.")
else:
    # --- Sidebar Filters ---
    st.sidebar.header("Dashboard Filters")
    group_options = ['Indian Military', 'PLA (Chinese Military)']
    selected_groups = st.sidebar.multiselect("Select Groups:", options=group_options, default=group_options)
    min_date, max_date = df_master['created_utc'].min().date(), df_master['created_utc'].max().date()
    date_range = st.sidebar.date_input("Select Date Range:", value=(min_date, max_date), min_value=min_date, max_value=max_date)

    # --- Filtering Logic ---
    start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    # Keyword lists for group assignment
    indian_keywords = ['indian army', 'iaf', 'indian navy', 'indian military']
    pla_keywords = ['pla', 'chinese military', 'plaaf', 'plan']
    def assign_group(text):
        text_lower = text.lower()
        is_indian = any(keyword in text_lower for keyword in indian_keywords)
        is_pla = any(keyword in text_lower for keyword in pla_keywords)
        if is_indian: return 'Indian Military'
        if is_pla: return 'PLA (Chinese Military)'
        return 'Other'
    df_master['group'] = df_master['full_text'].apply(assign_group)
    filtered_df = df_master[(df_master['created_utc'] >= start_date) & (df_master['created_utc'] <= end_date) & (df_master['group'].isin(selected_groups))]

    st.header("Analysis Overview")
    st.markdown(f"*{len(filtered_df):,} posts from {start_date.strftime('%d-%b-%Y')} to {end_date.strftime('%d-%b-%Y')}*")

    # --- Create All Four Tabs ---
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Sentiment", "💬 Phrases & Entities", "☁️ Word Clouds", "📑 Topic Dashboard", "🕸️ Network Graph"])

    # Filtered data for each group
    ia_df = filtered_df[filtered_df['group'] == 'Indian Military']
    pla_df = filtered_df[filtered_df['group'] == 'PLA (Chinese Military)']

    with tab1:
        st.subheader("Sentiment Distribution by Group")
        if not filtered_df.empty:
            fig, ax = plt.subplots(figsize=(10, 6))
            sentiment_counts = filtered_df.groupby('group')['sentiment_label'].value_counts(normalize=True).unstack(fill_value=0).mul(100)
            sentiment_counts.plot(kind='bar', ax=ax, colormap='viridis')
            ax.set_ylabel("Percentage of Posts (%)"); ax.set_xlabel(""); ax.tick_params(axis='x', rotation=0)
            st.pyplot(fig)
        else: st.warning("No data for sentiment chart.")

    with tab2:
        st.subheader("Top Phrases and Entities Comparison")
        def plot_top_items(ax, data, title, palette):
            if data:
                item_df = pd.DataFrame(data, columns=['item', 'count'])
                sns.barplot(x='count', y='item', data=item_df, ax=ax, palette=palette, hue='item', legend=False)
                ax.set_title(title, fontsize=14)
            else:
                ax.text(0.5, 0.5, 'No data to display', ha='center', va='center'); ax.set_title(title, fontsize=14)

        st.markdown("#### Top Phrases (Bigrams)")
        fig_ph, (ax1_ph, ax2_ph) = plt.subplots(1, 2, figsize=(20, 10))
        plot_top_items(ax1_ph, get_top_items(ia_df['cleaned_text'], 'ngram'), 'Indian Military Discussions', 'Blues_r')
        plot_top_items(ax2_ph, get_top_items(pla_df['cleaned_text'], 'ngram'), 'PLA Discussions', 'Oranges_r')
        st.pyplot(fig_ph)
        
        st.markdown("---")
        st.markdown("#### Top Entities (Countries & Orgs)")
        fig_ent, (ax1_ent, ax2_ent) = plt.subplots(1, 2, figsize=(20, 10))
        plot_top_items(ax1_ent, get_top_items(ia_df['full_text'], 'ner'), 'Indian Military Discussions', 'Greens_r')
        plot_top_items(ax2_ent, get_top_items(pla_df['full_text'], 'ner'), 'PLA Discussions', 'Reds_r')
        st.pyplot(fig_ent)

    with tab3:
        st.subheader("Word Clouds")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Indian Military Discussions")
            if not ia_df.empty:
                ia_text = ' '.join(ia_df['cleaned_text'].dropna())
                wordcloud = WordCloud(width=800, height=600, background_color='white', colormap='Greens').generate(ia_text)
                st.image(wordcloud.to_array())
        with col2:
            st.markdown("#### PLA Discussions")
            if not pla_df.empty:
                pla_text = ' '.join(pla_df['cleaned_text'].dropna())
                wordcloud = WordCloud(width=800, height=600, background_color='black', colormap='Reds').generate(pla_text)
                st.image(wordcloud.to_array())
    
    with tab4:
        st.subheader("AI-Powered Topic Dashboard")
        st.markdown("This dashboard identifies themes in the conversation and measures their frequency, engagement, and sentiment.")
        if 'Indian Military' in selected_groups:
            if not ia_df.empty:
                st.markdown("---"); st.markdown("#### Indian Military Discussions")
                with st.spinner('Analyzing Indian Military topics with Gemini...'):
                    st.dataframe(analyze_topic_intensity(ia_df))
            else: st.warning("No 'Indian Military' posts found.")
        if 'PLA (Chinese Military)' in selected_groups:
            if not pla_df.empty:
                st.markdown("---"); st.markdown("#### PLA Discussions")
                with st.spinner('Analyzing PLA topics with Gemini...'):
                    st.dataframe(analyze_topic_intensity(pla_df))
            else: st.warning("No 'PLA (Chinese Military)' posts found.")

    with tab5:
        st.subheader("Entity Co-occurrence Network")
        edge_threshold = st.slider("Connection Strength:", min_value=1, max_value=50, value=15, step=1)
        st.markdown("This graph shows which countries & organizations are mentioned together. Larger nodes are more frequent, and thicker lines indicate stronger connections.")
        with st.spinner("Generating network graph..."):
            network_fig = generate_network_html(filtered_df, edge_threshold)
            if network_fig:
                st.plotly_chart(network_fig, use_container_width=True)
            else:
                st.warning("Not enough connections to build a network graph with the current filters and connection strength.")