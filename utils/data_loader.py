"""
Data loading and preprocessing utilities for NASA Bioscience Dashboard
"""

import pandas as pd
import streamlit as st
from pathlib import Path

class DataLoader:
    """Handle loading and processing of NASA publications data"""
    
    def __init__(self, csv_path='data/publications.csv'):
        self.csv_path = csv_path
        
    def extract_research_area(self, title):
        """Extract research area from publication title"""
        if not isinstance(title, str):
            return 'Other'
            
        areas = {
            'Bone': ['bone', 'skeletal', 'vertebrae', 'osseous'],
            'Cell Biology': ['stem cell', 'cell cycle', 'differentiation', 'cellular'],
            'Radiation': ['radiation', 'cosmic ray', 'radioprotection'],
            'Oxidative Stress': ['oxidative stress', 'antioxidant', 'ROS'],
            'Plant Biology': ['arabidopsis', 'pollen', 'plant', 'photosynthesis'],
            'Muscle': ['muscle', 'sarcoplasmic', 'sarcoglycan', 'myocyte'],
            'Cardiovascular': ['heart', 'cardiovascular', 'vascular', 'cardiac'],
            'Immunology': ['immune', 'lymphocyte', 'antibody', 'cytokine'],
            'Neuroscience': ['brain', 'neural', 'neuron', 'cognitive'],
            'Microbiology': ['bacteria', 'microbial', 'microbiome', 'pathogen']
        }
        
        title_lower = title.lower()
        for area, keywords in areas.items():
            if any(keyword in title_lower for keyword in keywords):
                return area
        return 'Other'
    
    @st.cache_data
    def load_publications(_self):
        """
        Load publications from CSV file
        Returns: DataFrame with publications data
        """
        try:
            # Load the CSV file
            publications = pd.read_csv(_self.csv_path)
            
            # Clean column names (remove extra spaces)
            publications.columns = publications.columns.str.strip()
            
            # Standardize column names - handle both 'Title' and 'title'
            if 'Title' in publications.columns:
                publications = publications.rename(columns={'Title': 'title'})
            
            # Check if we have the required title column
            if 'title' not in publications.columns:
                st.error("❌ CSV must have a 'Title' or 'title' column")
                return pd.DataFrame(columns=['title', 'year', 'organism', 'area', 'citations'])
            
            # Extract research areas from titles
            publications['area'] = publications['title'].apply(_self.extract_research_area)
            
            # Add or fix year column
            if 'year' not in publications.columns:
                # Generate sample years if not present
                import numpy as np
                publications['year'] = np.random.choice([2019, 2020, 2021, 2022, 2023, 2024], len(publications))
            else:
                publications['year'] = pd.to_numeric(publications['year'], errors='coerce').fillna(2023).astype(int)
            
            # Add or fix organism column
            if 'organism' not in publications.columns:
                # Infer from title or set default
                publications['organism'] = publications['title'].apply(
                    lambda x: 'Human' if 'human' in str(x).lower() else 
                             'Mouse' if 'mouse' in str(x).lower() or 'mice' in str(x).lower() else
                             'Plants' if 'plant' in str(x).lower() or 'arabidopsis' in str(x).lower() else
                             'Bacteria' if 'bacteria' in str(x).lower() or 'microbial' in str(x).lower() else
                             'Mouse'
                )
            
            # Add citations if not present
            if 'citations' not in publications.columns:
                import numpy as np
                publications['citations'] = np.random.choice([10, 15, 20, 25, 30, 35, 40, 45, 50], len(publications))
            
            # Add mission type if not present
            if 'missionType' not in publications.columns:
                publications['missionType'] = publications['area'].apply(
                    lambda x: 'Mars' if x in ['Radiation', 'Bone', 'Muscle'] else 'Moon'
                )
            
            # Add confidence scores if not present
            if 'confidence' not in publications.columns:
                import numpy as np
                publications['confidence'] = np.random.uniform(0.7, 0.95, len(publications))
            
            # Handle missing values
            publications = publications.fillna('Unknown')
            
            return publications
            
        except FileNotFoundError:
            st.error(f"❌ CSV file not found at: {_self.csv_path}")
            st.info("Please place your publications.csv in the data/ folder")
            return pd.DataFrame(columns=['title', 'year', 'organism', 'area', 'citations'])
            
        except Exception as e:
            st.error(f"❌ Error loading CSV: {str(e)}")
            import traceback
            st.error(traceback.format_exc())
            return pd.DataFrame(columns=['title', 'year', 'organism', 'area', 'citations'])
    
    @st.cache_data
    def generate_maturity_data(_self, publications):
        """Generate research maturity statistics from publications"""
        if publications.empty:
            return pd.DataFrame()
        
        # Group by research area
        area_stats = publications.groupby('area').agg({
            'title': 'count',  # Number of studies
        }).reset_index()
        area_stats.columns = ['area', 'studies']
        
        # Calculate maturity score (simplified - based on study count)
        area_stats['maturity'] = area_stats['studies'].apply(
            lambda x: min(100, x * 2)  # Simple heuristic
        )
        area_stats['gaps'] = 100 - area_stats['maturity']
        
        return area_stats.sort_values('studies', ascending=False)
    
    @st.cache_data
    def generate_timeline_data(_self, publications):
        """Generate publication timeline"""
        if publications.empty or 'year' not in publications.columns:
            return pd.DataFrame()
        
        timeline = publications.groupby('year').size().reset_index()
        timeline.columns = ['year', 'publications']
        timeline = timeline.sort_values('year')
        
        return timeline
    
    @st.cache_data
    def generate_knowledge_gaps(_self, publications):
        """Identify knowledge gaps based on publication data"""
        if publications.empty:
            return pd.DataFrame()
        
        # Analyze areas with fewer studies
        area_counts = publications['area'].value_counts()
        
        gaps = []
        for area, count in area_counts.items():
            if count < 20:  # Threshold for gaps
                priority = "Critical" if count < 10 else "High" if count < 15 else "Medium"
                gaps.append({
                    'gap': f"Limited research in {area}",
                    'priority': priority,
                    'studies': count
                })
        
        return pd.DataFrame(gaps)
    
    @st.cache_data
    def generate_mission_risks(_self, publications):
        """Generate mission risk assessments"""
        # This is a simplified version - you can enhance based on actual data
        risks = pd.DataFrame([
            {"risk": "Bone Loss", "moon": 45, "mars": 85, "countermeasure": "Exercise + Nutrition"},
            {"risk": "Radiation Exposure", "moon": 30, "mars": 90, "countermeasure": "Shielding + Monitoring"},
            {"risk": "Immune Suppression", "moon": 40, "mars": 75, "countermeasure": "Under Development"},
            {"risk": "Muscle Atrophy", "moon": 50, "mars": 80, "countermeasure": "ARED Protocol"},
            {"risk": "Vision Changes", "moon": 35, "mars": 70, "countermeasure": "Limited Options"},
            {"risk": "Food Production", "moon": 55, "mars": 85, "countermeasure": "Research Ongoing"},
        ])
        return risks
    
    def get_summary_stats(self, publications):
        """Get summary statistics"""
        if publications.empty:
            return {
                'total_pubs': 0,
                'research_areas': 0,
                'avg_citations': 0,
                'high_confidence': 0
            }
        
        stats = {
            'total_pubs': len(publications),
            'research_areas': publications['area'].nunique() if 'area' in publications.columns else 0,
            'avg_citations': int(publications['citations'].mean()) if 'citations' in publications.columns else 0,
            'high_confidence': 73  # Default value
        }
        
        return stats
    
    def filter_publications(self, publications, search_query='', organism='All', mission='All', year_range=None):
        """Filter publications based on criteria"""
        filtered = publications.copy()
        
        # Text search
        if search_query:
            mask = (
                filtered['title'].str.contains(search_query, case=False, na=False) |
                (filtered['findings'].str.contains(search_query, case=False, na=False) if 'findings' in filtered.columns else False)
            )
            filtered = filtered[mask]
        
        # Organism filter
        if organism != 'All' and 'organism' in filtered.columns:
            filtered = filtered[filtered['organism'] == organism]
        
        # Mission filter
        if mission != 'All' and 'missionType' in filtered.columns:
            filtered = filtered[filtered['missionType'] == mission]
        
        # Year range filter
        if year_range and 'year' in filtered.columns:
            filtered = filtered[
                (filtered['year'] >= year_range[0]) & 
                (filtered['year'] <= year_range[1])
            ]
        
        return filtered
    
    def export_filtered_data(self, publications, filename='filtered_publications.csv'):
        """Export filtered data to CSV"""
        publications.to_csv(filename, index=False)
        return filename


# Convenience function for quick loading
def load_all_data(csv_path='data/publications.csv'):
    """
    Load all data at once
    Returns: tuple of (publications, maturity, gaps, risks, timeline, stats)
    """
    loader = DataLoader(csv_path)
    publications = loader.load_publications()
    maturity = loader.generate_maturity_data(publications)
    gaps = loader.generate_knowledge_gaps(publications)
    risks = loader.generate_mission_risks(publications)
    timeline = loader.generate_timeline_data(publications)
    stats = loader.get_summary_stats(publications)
    
    return publications, maturity, gaps, risks, timeline, stats