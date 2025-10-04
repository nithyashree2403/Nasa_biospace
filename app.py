"""
NASA Bioscience Research Explorer - Main Application
Integrated with utility modules for AI search and knowledge graphs
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# Import custom utilities
from utils.data_loader import DataLoader
from utils.ai_search import create_search_engine
from utils.knowledge_graph import KnowledgeGraph

# Page configuration
st.set_page_config(
    page_title="NASA Bioscience Research Explorer",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 50%, #0f172a 100%);
    }
    .stMetric {
        background-color: rgba(59, 130, 246, 0.1);
        padding: 15px;
        border-radius: 10px;
        border: 1px solid rgba(59, 130, 246, 0.3);
    }
    h1, h2, h3 {
        color: #60a5fa;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(0, 0, 0, 0.2);
        border-radius: 8px;
        padding: 10px 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2563eb;
    }
</style>
""", unsafe_allow_html=True)

# Initialize data loader
@st.cache_resource
def initialize_data():
    """Initialize and load all data"""
    loader = DataLoader('data/publications.csv')
    publications = loader.load_publications()
    
    # Generate derived data
    maturity = loader.generate_maturity_data(publications)
    gaps = loader.generate_knowledge_gaps(publications)
    risks = loader.generate_mission_risks(publications)
    timeline = loader.generate_timeline_data(publications)
    stats = loader.get_summary_stats(publications)
    
    # Initialize AI search
    search_engine = create_search_engine(publications)
    
    # Initialize knowledge graph
    kg = KnowledgeGraph()
    kg.build_from_publications(publications)
    
    return publications, maturity, gaps, risks, timeline, stats, search_engine, kg

# Load data
try:
    publications_df, maturity_df, gaps_df, risks_df, timeline_df, stats, search_engine, kg = initialize_data()
    data_loaded = True
except Exception as e:
    st.error(f"Error loading data: {str(e)}")
    st.info("Please ensure publications.csv is in the data/ folder")
    data_loaded = False
    publications_df = pd.DataFrame()
    stats = {'total_pubs': 0, 'research_areas': 0, 'avg_citations': 0, 'high_confidence': 0}

# Header
col1, col2 = st.columns([3, 1])
with col1:
    st.title("🚀 NASA Bioscience Research Explorer")
    st.caption(f"{stats['total_pubs']} Publications • AI-Powered Insights")
with col2:
    st.metric("Last Updated", "Jan 2025", delta="Current")

# Sidebar
with st.sidebar:
    st.header("📊 Quick Stats")
    st.metric("Total Studies", stats['total_pubs'])
    st.metric("Research Areas", stats['research_areas'])
    if stats['avg_citations'] > 0:
        st.metric("Avg Citations", stats['avg_citations'])
    
    st.markdown("---")
    
    st.header("🔍 Filters")
    
    # Global filters (if data loaded)
    if data_loaded and not publications_df.empty:
        if 'year' in publications_df.columns:
            year_min = int(publications_df['year'].min())
            year_max = int(publications_df['year'].max())
            year_range = st.slider(
                "Year Range",
                year_min, year_max,
                (year_min, year_max)
            )
        else:
            year_range = None
        
        if 'organism' in publications_df.columns:
            organisms = ['All'] + sorted(publications_df['organism'].dropna().unique().tolist())
        else:
            organisms = ['All']
        
        if 'missionType' in publications_df.columns:
            missions = ['All'] + sorted(publications_df['missionType'].dropna().unique().tolist())
        else:
            missions = ['All']
    else:
        year_range = None
        organisms = ['All']
        missions = ['All']
    
    st.markdown("---")
    
    st.header("ℹ️ About")
    st.write("This dashboard analyzes NASA bioscience publications to provide insights for space exploration.")
    
    st.markdown("---")
    
    st.header("📚 Data Sources")
    st.write("- NASA OSDR")
    st.write("- GeneLab")
    st.write("- Space Life Sciences Library")
    
    st.caption("Built with Streamlit • Data updated Jan 2025")

# Main content tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview", 
    "🔍 Smart Search", 
    "⚠️ Knowledge Gaps", 
    "🎯 Mission Planner",
    "🕸️ Knowledge Graph"
])

# TAB 1: OVERVIEW
with tab1:
    if not data_loaded:
        st.warning("⚠️ No data loaded. Please add publications.csv to the data/ folder.")
    else:
        # Key Metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Publications", stats['total_pubs'])
        with col2:
            st.metric("Research Areas", stats['research_areas'])
        with col3:
            st.metric("High Confidence", f"{stats['high_confidence']}%")
        with col4:
            gap_count = len(gaps_df) if not gaps_df.empty else 0
            st.metric("Critical Gaps", gap_count)
        
        st.markdown("---")
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🔬 Research Maturity by Area")
            if not maturity_df.empty:
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    name='Maturity %',
                    x=maturity_df['area'],
                    y=maturity_df['maturity'],
                    marker_color='#3b82f6'
                ))
                fig.add_trace(go.Bar(
                    name='Knowledge Gaps %',
                    x=maturity_df['area'],
                    y=maturity_df['gaps'],
                    marker_color='#f59e0b'
                ))
                fig.update_layout(
                    barmode='group',
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='white',
                    height=400,
                    xaxis_tickangle=-45
                )
                st.plotly_chart(fig, config={'displayModeBar': False})
            else:
                st.info("No maturity data available")
        
        with col2:
            st.subheader("📈 Publications by Year")
            if not timeline_df.empty:
                fig = px.line(timeline_df, x='year', y='publications', markers=True)
                fig.update_traces(line_color='#8b5cf6', line_width=3, marker_size=10)
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='white',
                    height=400
                )
                st.plotly_chart(fig, config={'displayModeBar': False})
            else:
                st.info("No timeline data available")
        
        # High Impact Studies
        st.subheader("🎯 High-Impact Recent Studies")
        if not publications_df.empty:
            display_pubs = publications_df.head(5)
            for idx, pub in display_pubs.iterrows():
                # Handle both 'title' and 'Title' column names
                title = pub.get('title', pub.get('Title', 'Untitled'))
                year = pub.get('year', 'N/A')
                
                with st.expander(f"**{title}** ({year})"):
                    cols = st.columns(4)
                    
                    with cols[0]:
                        st.metric("Area", pub.get('area', 'N/A'))
                    with cols[1]:
                        citations = pub.get('citations', 0)
                        st.metric("Citations", citations if pd.notna(citations) else 'N/A')
                    with cols[2]:
                        conf = pub.get('confidence', 0)
                        if pd.notna(conf) and conf > 0:
                            st.metric("Confidence", f"{int(conf*100)}%")
                        else:
                            st.metric("Confidence", "N/A")
                    with cols[3]:
                        st.metric("Mission", pub.get('missionType', 'N/A'))
                    
                    findings = pub.get('findings', pub.get('abstract', 'No findings available'))
                    st.write(f"**Key Finding:** {findings}")
                    
                    # Add link if available
                    if 'Link' in pub and pd.notna(pub['Link']):
                        st.markdown(f"[📄 View Publication]({pub['Link']})")
        else:
            st.info("No publications to display")

# TAB 2: SMART SEARCH
with tab2:
    st.subheader("🔍 AI-Powered Research Search")
    
    if not data_loaded:
        st.warning("⚠️ Please load data first")
    else:
        # Search interface
        col1, col2, col3 = st.columns([3, 1, 1])
        with col1:
            search_query = st.text_input(
                "Search publications",
                placeholder="e.g., 'bone density Mars mission', 'plant growth microgravity'"
            )
        with col2:
            organism_filter = st.selectbox("Organism", organisms)
        with col3:
            mission_filter = st.selectbox("Mission", missions)
        
        # Perform search
        if search_query and search_engine:
            # Use AI semantic search
            with st.spinner("Searching..."):
                results = search_engine.semantic_search(search_query, top_k=20)
                
                # Apply additional filters
                if organism_filter != 'All' and 'organism' in results.columns:
                    results = results[results['organism'] == organism_filter]
                if mission_filter != 'All' and 'missionType' in results.columns:
                    results = results[results['missionType'] == mission_filter]
        else:
            # Show all publications with filters
            loader = DataLoader()
            results = loader.filter_publications(
                publications_df,
                organism=organism_filter,
                mission=mission_filter
            )
        
        st.info(f"Found {len(results)} publications")
        
        # Display results
        if not results.empty:
            for idx, pub in results.iterrows():
                with st.container():
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        # Handle both 'title' and 'Title' column names
                        title = pub.get('title', pub.get('Title', 'Untitled'))
                        st.markdown(f"### {title}")
                        
                        # Tags
                        tags = []
                        if 'organism' in pub and pd.notna(pub['organism']):
                            tags.append(f"🧬 {pub['organism']}")
                        if 'missionType' in pub and pd.notna(pub['missionType']):
                            tags.append(f"🚀 {pub['missionType']}")
                        if 'year' in pub and pd.notna(pub['year']):
                            tags.append(f"📅 {int(pub['year'])}")
                        
                        st.caption(" | ".join(tags))
                        
                        # Findings
                        findings = pub.get('findings', pub.get('abstract', 'No description available'))
                        if findings != 'No description available':
                            st.write(findings[:300] + "..." if len(str(findings)) > 300 else findings)
                        
                        # Metrics
                        metrics = []
                        if 'citations' in pub and pd.notna(pub['citations']):
                            metrics.append(f"📚 {int(pub['citations'])} citations")
                        if 'relevance_score' in pub:
                            metrics.append(f"Relevance: {pub['relevance_score']:.2f}")
                        if 'confidence' in pub and pd.notna(pub['confidence']):
                            metrics.append(f"Confidence: {int(pub['confidence']*100)}%")
                        
                        if metrics:
                            st.caption(" • ".join(metrics))
                        
                        # Add link if available
                        if 'Link' in pub and pd.notna(pub['Link']):
                            st.markdown(f"[📄 View Publication]({pub['Link']})")
                    
                    with col2:
                        if 'confidence' in pub and pd.notna(pub['confidence']):
                            st.metric("Confidence", f"{int(pub['confidence']*100)}%")
                    
                    st.markdown("---")
        else:
            st.info("No results found. Try adjusting your search query or filters.")

# TAB 3: KNOWLEDGE GAPS
with tab3:
    st.subheader("⚠️ Critical Knowledge Gaps")
    st.write("Areas requiring additional research for safe Moon and Mars missions")
    
    if not data_loaded or gaps_df.empty:
        st.info("No knowledge gap data available")
    else:
        # Display gaps
        for idx, gap in gaps_df.iterrows():
            priority_color = {
                "Critical": "🔴",
                "High": "🟠",
                "Medium": "🟡"
            }
            icon = priority_color.get(gap['priority'], "⚪")
            
            with st.expander(f"{icon} {gap['gap']} - **{gap['priority']} Priority**"):
                st.write(f"**Related Studies:** {gap['studies']}")
                st.write("**Why this matters:** This knowledge gap represents a critical unknown that could impact mission safety and success. Additional research is needed to develop effective countermeasures.")
        
        st.markdown("---")
        
        # Gap Analysis Chart
        st.subheader("📊 Research Coverage Analysis")
        if not maturity_df.empty:
            fig = px.scatter(
                maturity_df, 
                x='studies', 
                y='maturity',
                size='gaps', 
                color='area', 
                hover_name='area',
                labels={'studies': 'Number of Studies', 'maturity': 'Research Maturity %'},
                size_max=60
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white',
                height=500
            )
            st.plotly_chart(fig, config={'displayModeBar': False})

# TAB 4: MISSION PLANNER
with tab4:
    st.subheader("🎯 Mission Risk Assessment Tool")
    st.write("Compare biological risks for Moon vs Mars missions based on aggregated research")
    
    if not data_loaded or risks_df.empty:
        st.info("No mission risk data available")
    else:
        # Risk comparison
        for idx, risk in risks_df.iterrows():
            st.markdown(f"#### {risk['risk']}")
            st.caption(f"**Countermeasure:** {risk['countermeasure']}")
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Moon Mission**")
                st.progress(risk['moon'] / 100)
                st.caption(f"{risk['moon']}% risk level")
            with col2:
                st.write("**Mars Mission**")
                st.progress(risk['mars'] / 100)
                st.caption(f"{risk['mars']}% risk level")
            st.markdown("---")
        
        # Radar Chart
        st.subheader("📡 Risk Profile Comparison")
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=risks_df['moon'].tolist() + [risks_df['moon'].iloc[0]],
            theta=risks_df['risk'].tolist() + [risks_df['risk'].iloc[0]],
            fill='toself',
            name='Moon Mission',
            line_color='#3b82f6'
        ))
        fig.add_trace(go.Scatterpolar(
            r=risks_df['mars'].tolist() + [risks_df['mars'].iloc[0]],
            theta=risks_df['risk'].tolist() + [risks_df['risk'].iloc[0]],
            fill='toself',
            name='Mars Mission',
            line_color='#ef4444'
        ))
        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 100], gridcolor='rgba(255,255,255,0.2)'),
                angularaxis=dict(gridcolor='rgba(255,255,255,0.2)')
            ),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            height=600
        )
        st.plotly_chart(fig, config={'displayModeBar': False})

# TAB 5: KNOWLEDGE GRAPH
with tab5:
    st.subheader("🕸️ Research Knowledge Graph")
    st.write("Explore relationships between research areas, organisms, and topics")
    
    if not data_loaded:
        st.info("No data available for knowledge graph")
    else:
        # Graph statistics
        stats_kg = kg.get_graph_statistics()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Nodes", stats_kg['num_nodes'])
        with col2:
            st.metric("Edges", stats_kg['num_edges'])
        with col3:
            st.metric("Density", f"{stats_kg['density']:.3f}")
        with col4:
            st.metric("Avg Degree", f"{stats_kg['avg_degree']:.1f}")
        
        st.markdown("---")
        
        # Visualize graph
        fig = kg.create_interactive_plot(width=1000, height=700)
        st.plotly_chart(fig, use_container_width=True)
        
        # Central concepts
        st.subheader("🎯 Most Central Research Concepts")
        central = kg.get_central_concepts(top_n=10)
        
        if central:
            central_df = pd.DataFrame(central, columns=['Concept', 'Centrality Score'])
            central_df['Centrality Score'] = central_df['Centrality Score'].round(3)
            st.dataframe(central_df)
        else:
            st.info("No central concepts identified")