import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(
    page_title="NASA Bioscience Research Explorer",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
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

# Mock data
@st.cache_data
def load_data():
    publications = pd.DataFrame([
        {"id": 1, "title": "Microgravity Effects on Bone Density in Long-Duration Spaceflight", "year": 2023, "organism": "Human", "topic": "Bone Health", "missionType": "Mars", "confidence": 0.92, "citations": 45, "findings": "15% bone density loss over 6 months", "area": "Musculoskeletal"},
        {"id": 2, "title": "Plant Growth Responses to Reduced Gravity Environments", "year": 2022, "organism": "Plants", "topic": "Plant Growth", "missionType": "Moon", "confidence": 0.88, "citations": 38, "findings": "Altered root orientation and growth rates", "area": "Botany"},
        {"id": 3, "title": "Cardiovascular Adaptation During Extended Microgravity Exposure", "year": 2023, "organism": "Human", "topic": "Cardiovascular", "missionType": "Mars", "confidence": 0.85, "citations": 52, "findings": "Cardiac atrophy and fluid redistribution", "area": "Cardiovascular"},
        {"id": 4, "title": "Radiation Effects on DNA Repair Mechanisms in Space", "year": 2021, "organism": "Human", "topic": "Radiation Biology", "missionType": "Mars", "confidence": 0.90, "citations": 67, "findings": "Increased DNA damage in GCR exposure", "area": "Radiation"},
        {"id": 5, "title": "Microbial Behavior in Closed-Loop Life Support Systems", "year": 2022, "organism": "Bacteria", "topic": "Microbiology", "missionType": "Moon", "confidence": 0.79, "citations": 29, "findings": "Biofilm formation accelerated in microgravity", "area": "Microbiology"},
        {"id": 6, "title": "Immune System Dysregulation in Spaceflight Conditions", "year": 2023, "organism": "Human", "topic": "Immunology", "missionType": "Mars", "confidence": 0.87, "citations": 41, "findings": "T-cell function suppression observed", "area": "Immunology"},
        {"id": 7, "title": "Muscle Atrophy Countermeasures: Exercise Protocols", "year": 2022, "organism": "Human", "topic": "Muscle Health", "missionType": "Mars", "confidence": 0.91, "citations": 55, "findings": "ARED reduces muscle loss by 40%", "area": "Musculoskeletal"},
        {"id": 8, "title": "Sleep Patterns and Circadian Rhythm Disruption", "year": 2021, "organism": "Human", "topic": "Neuroscience", "missionType": "Moon", "confidence": 0.83, "citations": 33, "findings": "Circadian misalignment in 78% of subjects", "area": "Neuroscience"},
        {"id": 9, "title": "Arabidopsis Gene Expression in Microgravity", "year": 2023, "organism": "Plants", "topic": "Plant Growth", "missionType": "Mars", "confidence": 0.86, "citations": 42, "findings": "1,200+ genes differentially expressed", "area": "Botany"},
        {"id": 10, "title": "Radiation Shielding: Biological Effectiveness Studies", "year": 2022, "organism": "Human", "topic": "Radiation Biology", "missionType": "Mars", "confidence": 0.89, "citations": 48, "findings": "Polyethylene reduces exposure by 30%", "area": "Radiation"},
    ])
    
    maturity = pd.DataFrame([
        {"area": "Musculoskeletal", "studies": 45, "maturity": 85, "gaps": 15},
        {"area": "Cardiovascular", "studies": 38, "maturity": 78, "gaps": 22},
        {"area": "Radiation", "studies": 52, "maturity": 72, "gaps": 28},
        {"area": "Botany", "studies": 29, "maturity": 65, "gaps": 35},
        {"area": "Microbiology", "studies": 34, "maturity": 68, "gaps": 32},
        {"area": "Immunology", "studies": 41, "maturity": 70, "gaps": 30},
        {"area": "Neuroscience", "studies": 25, "maturity": 62, "gaps": 38},
    ])
    
    gaps = pd.DataFrame([
        {"gap": "Long-term Mars radiation effects", "priority": "Critical", "studies": 12},
        {"gap": "Multi-generational plant growth", "priority": "High", "studies": 8},
        {"gap": "Microbial evolution in space", "priority": "Medium", "studies": 15},
        {"gap": "Pregnancy and development", "priority": "Critical", "studies": 3},
        {"gap": "Artificial gravity efficacy", "priority": "High", "studies": 18},
    ])
    
    risks = pd.DataFrame([
        {"risk": "Bone Loss", "moon": 45, "mars": 85, "countermeasure": "Exercise + Nutrition"},
        {"risk": "Radiation Exposure", "moon": 30, "mars": 90, "countermeasure": "Shielding + Monitoring"},
        {"risk": "Immune Suppression", "moon": 40, "mars": 75, "countermeasure": "Under Development"},
        {"risk": "Muscle Atrophy", "moon": 50, "mars": 80, "countermeasure": "ARED Protocol"},
        {"risk": "Vision Changes", "moon": 35, "mars": 70, "countermeasure": "Limited Options"},
        {"risk": "Food Production", "moon": 55, "mars": 85, "countermeasure": "Research Ongoing"},
    ])
    
    timeline = pd.DataFrame([
        {"year": "2019", "publications": 45},
        {"year": "2020", "publications": 52},
        {"year": "2021", "publications": 68},
        {"year": "2022", "publications": 89},
        {"year": "2023", "publications": 112},
        {"year": "2024", "publications": 128},
        {"year": "2025", "publications": 114},
    ])
    
    return publications, maturity, gaps, risks, timeline

publications_df, maturity_df, gaps_df, risks_df, timeline_df = load_data()

# Header
col1, col2 = st.columns([3, 1])
with col1:
    st.title("🚀 NASA Bioscience Research Explorer")
    st.caption("608 Publications • AI-Powered Insights")
with col2:
    st.metric("Last Updated", "Jan 2025", delta="Current")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🔍 Smart Search", "⚠️ Knowledge Gaps", "🎯 Mission Planner"])

# TAB 1: OVERVIEW
with tab1:
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Publications", "608", delta="127 areas")
    with col2:
        st.metric("Research Areas", "127", delta="Active")
    with col3:
        st.metric("High Confidence", "73%", delta="+5%")
    with col4:
        st.metric("Critical Gaps", "34", delta="-2")
    
    st.markdown("---")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔬 Research Maturity by Area")
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
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📈 Publications by Year")
        fig = px.line(timeline_df, x='year', y='publications', markers=True)
        fig.update_traces(line_color='#8b5cf6', line_width=3, marker_size=10)
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # High Impact Studies
    st.subheader("🎯 High-Impact Recent Studies")
    for _, pub in publications_df.head(5).iterrows():
        with st.expander(f"**{pub['title']}** ({pub['year']})"):
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Area", pub['area'])
            with col2:
                st.metric("Citations", pub['citations'])
            with col3:
                st.metric("Confidence", f"{int(pub['confidence']*100)}%")
            with col4:
                st.metric("Mission", pub['missionType'])
            st.write(f"**Key Finding:** {pub['findings']}")

# TAB 2: SMART SEARCH
with tab2:
    st.subheader("🔍 AI-Powered Research Search")
    
    # Search filters
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        search_query = st.text_input("Search publications", placeholder="e.g., 'bone density Mars mission', 'plant growth microgravity'")
    with col2:
        organism_filter = st.selectbox("Organism", ["All", "Human", "Plants", "Bacteria"])
    with col3:
        mission_filter = st.selectbox("Mission", ["All", "Moon", "Mars"])
    
    # Filter data
    filtered_df = publications_df.copy()
    if search_query:
        filtered_df = filtered_df[
            filtered_df['title'].str.contains(search_query, case=False, na=False) |
            filtered_df['findings'].str.contains(search_query, case=False, na=False)
        ]
    if organism_filter != "All":
        filtered_df = filtered_df[filtered_df['organism'] == organism_filter]
    if mission_filter != "All":
        filtered_df = filtered_df[filtered_df['missionType'] == mission_filter]
    
    st.info(f"Found {len(filtered_df)} publications")
    
    # Display results
    for _, pub in filtered_df.iterrows():
        with st.container():
            col1, col2 = st.columns([4, 1])
            with col1:
                st.markdown(f"### {pub['title']}")
                st.caption(f"🧬 {pub['organism']} | 🚀 {pub['missionType']} | 📅 {pub['year']}")
                st.write(pub['findings'])
                st.caption(f"📚 {pub['citations']} citations • Confidence: {int(pub['confidence']*100)}%")
            with col2:
                st.metric("Confidence", f"{int(pub['confidence']*100)}%")
            st.markdown("---")

# TAB 3: KNOWLEDGE GAPS
with tab3:
    st.subheader("⚠️ Critical Knowledge Gaps")
    st.write("Areas requiring additional research for safe Moon and Mars missions")
    
    # Display gaps
    for _, gap in gaps_df.iterrows():
        priority_color = {
            "Critical": "🔴",
            "High": "🟠",
            "Medium": "🟡"
        }
        with st.expander(f"{priority_color[gap['priority']]} {gap['gap']} - **{gap['priority']} Priority**"):
            st.write(f"**Related Studies:** {gap['studies']}")
            st.write("**Why this matters:** This knowledge gap represents a critical unknown that could impact mission safety and success. Additional research is needed to develop effective countermeasures.")
    
    st.markdown("---")
    
    # Gap Analysis Chart
    st.subheader("📊 Research Coverage Analysis")
    fig = px.scatter(maturity_df, x='studies', y='maturity', 
                     size='gaps', color='area', hover_name='area',
                     labels={'studies': 'Number of Studies', 'maturity': 'Research Maturity %'},
                     size_max=60)
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)

# TAB 4: MISSION PLANNER
with tab4:
    st.subheader("🎯 Mission Risk Assessment Tool")
    st.write("Compare biological risks for Moon vs Mars missions based on aggregated research")
    
    # Risk comparison
    for _, risk in risks_df.iterrows():
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
    st.plotly_chart(fig, use_container_width=True)

# Sidebar
with st.sidebar:
    st.header("About")
    st.write("This dashboard analyzes 608 NASA bioscience publications to provide insights for space exploration.")
    
    st.markdown("---")
    
    st.header("Quick Stats")
    st.metric("Total Studies", "608")
    st.metric("Research Areas", "7")
    st.metric("Avg Citations", "44")
    
    st.markdown("---")
    
    st.header("Data Sources")
    st.write("- NASA OSDR")
    st.write("- GeneLab")
    st.write("- Space Life Sciences Library")
    
    st.markdown("---")
    st.caption("Built with Streamlit • Data updated Jan 2025")