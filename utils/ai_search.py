"""
AI-powered search and NLP utilities for NASA Bioscience Dashboard
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

class AISearch:
    """AI-powered semantic search for publications"""
    
    def __init__(self):
        self.vectorizer = None
        self.document_vectors = None
        self.documents = None
        
    def initialize(self, publications):
        """
        Initialize the search engine with publications data
        """
        if publications.empty:
            return False
        
        # Combine relevant text fields
        self.documents = publications.copy()
        
        # Create search corpus (title + findings/abstract)
        search_text = publications['title'].fillna('')
        
        if 'findings' in publications.columns:
            search_text = search_text + ' ' + publications['findings'].fillna('')
        if 'abstract' in publications.columns:
            search_text = search_text + ' ' + publications['abstract'].fillna('')
        
        # Create TF-IDF vectors
        try:
            self.vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2)
            )
            self.document_vectors = self.vectorizer.fit_transform(search_text)
            return True
        except Exception as e:
            st.error(f"Error initializing search: {str(e)}")
            return False
    
    def semantic_search(self, query, top_k=10):
        """
        Perform semantic search using cosine similarity
        
        Args:
            query: Search query string
            top_k: Number of top results to return
            
        Returns:
            DataFrame of top matching publications
        """
        if self.vectorizer is None:
            return pd.DataFrame()
        
        try:
            # Vectorize query
            query_vector = self.vectorizer.transform([query])
            
            # Calculate similarities
            similarities = cosine_similarity(query_vector, self.document_vectors)[0]
            
            # Get top results
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            # Filter out zero similarity results
            top_indices = [idx for idx in top_indices if similarities[idx] > 0]
            
            # Add similarity scores to results
            results = self.documents.iloc[top_indices].copy()
            results['relevance_score'] = [similarities[idx] for idx in top_indices]
            
            return results
            
        except Exception as e:
            st.error(f"Search error: {str(e)}")
            return pd.DataFrame()
    
    def extract_keywords(self, text, top_n=5):
        """
        Extract key terms from text using TF-IDF
        
        Args:
            text: Input text
            top_n: Number of keywords to extract
            
        Returns:
            List of keywords
        """
        if self.vectorizer is None:
            return []
        
        try:
            vector = self.vectorizer.transform([text])
            feature_names = self.vectorizer.get_feature_names_out()
            
            # Get top features by TF-IDF score
            scores = vector.toarray()[0]
            top_indices = np.argsort(scores)[::-1][:top_n]
            
            keywords = [feature_names[idx] for idx in top_indices if scores[idx] > 0]
            return keywords
            
        except Exception as e:
            return []
    
    def find_similar_publications(self, pub_id, top_k=5):
        """
        Find publications similar to a given publication
        
        Args:
            pub_id: ID or index of the publication
            top_k: Number of similar publications to return
            
        Returns:
            DataFrame of similar publications
        """
        if self.document_vectors is None:
            return pd.DataFrame()
        
        try:
            # Get vector for the publication
            pub_vector = self.document_vectors[pub_id]
            
            # Calculate similarities
            similarities = cosine_similarity(pub_vector, self.document_vectors)[0]
            
            # Get top results (excluding the publication itself)
            top_indices = np.argsort(similarities)[::-1][1:top_k+1]
            
            results = self.documents.iloc[top_indices].copy()
            results['similarity'] = [similarities[idx] for idx in top_indices]
            
            return results
            
        except Exception as e:
            st.error(f"Error finding similar publications: {str(e)}")
            return pd.DataFrame()
    
    def get_topic_clusters(self, n_clusters=5):
        """
        Identify topic clusters in publications
        
        Args:
            n_clusters: Number of clusters to create
            
        Returns:
            DataFrame with cluster assignments
        """
        if self.document_vectors is None:
            return pd.DataFrame()
        
        try:
            from sklearn.cluster import KMeans
            
            # Perform clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(self.document_vectors.toarray())
            
            # Add cluster labels
            results = self.documents.copy()
            results['cluster'] = clusters
            
            return results
            
        except ImportError:
            st.warning("KMeans clustering requires scikit-learn")
            return pd.DataFrame()
        except Exception as e:
            st.error(f"Clustering error: {str(e)}")
            return pd.DataFrame()
    
    def summarize_text(self, text, max_sentences=3):
        """
        Simple extractive summarization
        
        Args:
            text: Input text to summarize
            max_sentences: Maximum sentences in summary
            
        Returns:
            Summary string
        """
        if not text or len(text) < 100:
            return text
        
        # Split into sentences
        sentences = text.split('. ')
        
        if len(sentences) <= max_sentences:
            return text
        
        try:
            # Vectorize sentences
            sentence_vectors = self.vectorizer.transform(sentences)
            
            # Calculate average similarity to all sentences
            avg_similarities = cosine_similarity(sentence_vectors).mean(axis=1)
            
            # Get top sentences
            top_indices = np.argsort(avg_similarities)[::-1][:max_sentences]
            top_indices = sorted(top_indices)  # Keep order
            
            summary = '. '.join([sentences[idx] for idx in top_indices])
            return summary + '.'
            
        except Exception:
            # Fallback: just return first few sentences
            return '. '.join(sentences[:max_sentences]) + '.'


# Advanced search filters
def advanced_filter(publications, filters):
    """
    Apply advanced filters to publications
    
    Args:
        publications: DataFrame of publications
        filters: Dictionary of filter criteria
        
    Returns:
        Filtered DataFrame
    """
    filtered = publications.copy()
    
    # Year range
    if 'year_min' in filters and 'year' in filtered.columns:
        filtered = filtered[filtered['year'] >= filters['year_min']]
    if 'year_max' in filters and 'year' in filtered.columns:
        filtered = filtered[filtered['year'] <= filters['year_max']]
    
    # Citation threshold
    if 'min_citations' in filters and 'citations' in filtered.columns:
        filtered = filtered[filtered['citations'] >= filters['min_citations']]
    
    # Confidence threshold
    if 'min_confidence' in filters and 'confidence' in filtered.columns:
        filtered = filtered[filtered['confidence'] >= filters['min_confidence']]
    
    # Multiple organisms
    if 'organisms' in filters and 'organism' in filtered.columns:
        if filters['organisms']:
            filtered = filtered[filtered['organism'].isin(filters['organisms'])]
    
    # Multiple areas
    if 'areas' in filters and 'area' in filtered.columns:
        if filters['areas']:
            filtered = filtered[filtered['area'].isin(filters['areas'])]
    
    return filtered


# Convenience function
def create_search_engine(publications):
    """
    Create and initialize search engine
    
    Args:
        publications: DataFrame of publications
        
    Returns:
        AISearch instance
    """
    search = AISearch()
    search.initialize(publications)
    return search