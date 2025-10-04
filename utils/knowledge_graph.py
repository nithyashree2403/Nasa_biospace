"""
Knowledge graph utilities for NASA Bioscience Dashboard
"""

import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import streamlit as st
from collections import Counter

class KnowledgeGraph:
    """Build and visualize knowledge graphs from publications"""
    
    def __init__(self):
        self.graph = nx.Graph()
        
    def build_from_publications(self, publications):
        """
        Build knowledge graph from publications data
        
        Args:
            publications: DataFrame with publications
        """
        self.graph.clear()
        
        if publications.empty:
            return
        
        # Add nodes for each unique area, organism, topic
        areas = publications['area'].unique() if 'area' in publications.columns else []
        organisms = publications['organism'].unique() if 'organism' in publications.columns else []
        topics = publications['topic'].unique() if 'topic' in publications.columns else []
        
        # Add nodes with types
        for area in areas:
            self.graph.add_node(area, type='area', color='#3b82f6', size=20)
        
        for organism in organisms:
            self.graph.add_node(organism, type='organism', color='#10b981', size=15)
        
        for topic in topics:
            self.graph.add_node(topic, type='topic', color='#f59e0b', size=15)
        
        # Add edges based on co-occurrence in publications
        for _, pub in publications.iterrows():
            area = pub.get('area', None)
            organism = pub.get('organism', None)
            topic = pub.get('topic', None)
            
            # Connect area to organism
            if area and organism:
                if self.graph.has_edge(area, organism):
                    self.graph[area][organism]['weight'] += 1
                else:
                    self.graph.add_edge(area, organism, weight=1)
            
            # Connect area to topic
            if area and topic:
                if self.graph.has_edge(area, topic):
                    self.graph[area][topic]['weight'] += 1
                else:
                    self.graph.add_edge(area, topic, weight=1)
            
            # Connect organism to topic
            if organism and topic:
                if self.graph.has_edge(organism, topic):
                    self.graph[organism][topic]['weight'] += 1
                else:
                    self.graph.add_edge(organism, topic, weight=1)
    
    def get_central_concepts(self, top_n=10):
        """
        Get most central concepts using betweenness centrality
        
        Args:
            top_n: Number of top concepts to return
            
        Returns:
            List of (concept, centrality) tuples
        """
        if len(self.graph.nodes) == 0:
            return []
        
        centrality = nx.betweenness_centrality(self.graph)
        top_concepts = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        return top_concepts
    
    def find_research_bridges(self):
        """
        Find concepts that bridge different research areas
        
        Returns:
            List of bridge concepts
        """
        if len(self.graph.nodes) == 0:
            return []
        
        # Find articulation points (nodes whose removal disconnects the graph)
        try:
            bridges = list(nx.articulation_points(self.graph))
            return bridges
        except:
            return []
    
    def get_connected_components(self):
        """
        Get separate research clusters
        
        Returns:
            List of connected components
        """
        if len(self.graph.nodes) == 0:
            return []
        
        components = list(nx.connected_components(self.graph))
        return components
    
    def create_interactive_plot(self, width=800, height=600):
        """
        Create interactive Plotly visualization of the knowledge graph
        
        Args:
            width: Plot width in pixels
            height: Plot height in pixels
            
        Returns:
            Plotly figure
        """
        if len(self.graph.nodes) == 0:
            # Return empty figure
            fig = go.Figure()
            fig.add_annotation(
                text="No data available for knowledge graph",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color="white")
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                height=height
            )
            return fig
        
        # Use spring layout for positioning
        pos = nx.spring_layout(self.graph, k=2, iterations=50)
        
        # Create edge traces
        edge_trace = []
        for edge in self.graph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            
            weight = self.graph[edge[0]][edge[1]].get('weight', 1)
            
            edge_trace.append(go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode='lines',
                line=dict(width=weight * 0.5, color='rgba(255,255,255,0.2)'),
                hoverinfo='none',
                showlegend=False
            ))
        
        # Create node traces by type
        node_traces = {}
        
        for node in self.graph.nodes():
            node_data = self.graph.nodes[node]
            node_type = node_data.get('type', 'unknown')
            color = node_data.get('color', '#ffffff')
            size = node_data.get('size', 10)
            
            if node_type not in node_traces:
                node_traces[node_type] = {
                    'x': [], 'y': [], 'text': [],
                    'color': color, 'size': size
                }
            
            x, y = pos[node]
            node_traces[node_type]['x'].append(x)
            node_traces[node_type]['y'].append(y)
            node_traces[node_type]['text'].append(node)
        
        # Create figure
        fig = go.Figure()
        
        # Add edges
        for trace in edge_trace:
            fig.add_trace(trace)
        
        # Add nodes
        for node_type, data in node_traces.items():
            fig.add_trace(go.Scatter(
                x=data['x'],
                y=data['y'],
                mode='markers+text',
                name=node_type.capitalize(),
                text=data['text'],
                textposition='top center',
                marker=dict(
                    size=data['size'],
                    color=data['color'],
                    line=dict(width=2, color='white')
                ),
                hovertext=data['text'],
                hoverinfo='text'
            ))
        
        # Update layout
        fig.update_layout(
            title='Research Knowledge Graph',
            showlegend=True,
            hovermode='closest',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            width=width,
            height=height
        )
        
        return fig
    
    def export_to_gephi(self, filename='knowledge_graph.gexf'):
        """
        Export graph to GEXF format for Gephi
        
        Args:
            filename: Output filename
        """
        nx.write_gexf(self.graph, filename)
        return filename
    
    def get_graph_statistics(self):
        """
        Get basic statistics about the knowledge graph
        
        Returns:
            Dictionary of statistics
        """
        if len(self.graph.nodes) == 0:
            return {
                'num_nodes': 0,
                'num_edges': 0,
                'density': 0,
                'avg_degree': 0
            }
        
        stats = {
            'num_nodes': self.graph.number_of_nodes(),
            'num_edges': self.graph.number_of_edges(),
            'density': nx.density(self.graph),
            'avg_degree': sum(dict(self.graph.degree()).values()) / self.graph.number_of_nodes() if self.graph.number_of_nodes() > 0 else 0,
        }
        
        # Add clustering coefficient if graph is not empty
        try:
            stats['clustering_coefficient'] = nx.average_clustering(self.graph)
        except:
            stats['clustering_coefficient'] = 0
        
        return stats


def create_collaboration_network(publications):
    """
    Create network of research collaborations based on shared topics/areas
    
    Args:
        publications: DataFrame with publications
        
    Returns:
        NetworkX graph
    """
    G = nx.Graph()
    
    if publications.empty or 'area' not in publications.columns:
        return G
    
    # Count co-occurrences
    areas = publications['area'].value_counts()
    
    for area, count in areas.items():
        G.add_node(area, weight=count)
    
    # Add edges for areas that appear together in studies
    for area1 in areas.index:
        for area2 in areas.index:
            if area1 < area2:  # Avoid duplicates
                # Count how many publications have both areas
                # This is simplified - you'd need actual co-occurrence data
                G.add_edge(area1, area2, weight=1)
    
    return G


def analyze_research_trends(publications):
    """
    Analyze trends in research topics over time
    
    Args:
        publications: DataFrame with publications
        
    Returns:
        DataFrame with trend analysis
    """
    if publications.empty or 'year' not in publications.columns:
        return pd.DataFrame()
    
    # Group by year and topic/area
    if 'topic' in publications.columns:
        trends = publications.groupby(['year', 'topic']).size().reset_index(name='count')
    elif 'area' in publications.columns:
        trends = publications.groupby(['year', 'area']).size().reset_index(name='count')
    else:
        trends = publications.groupby('year').size().reset_index(name='count')
    
    return trends