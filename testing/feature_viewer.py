"""
Feature Data Explorer - Streamlit App
Explore the engineered features from feature_engineer.py
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="NFL Features Explorer",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 NFL Feature Data Explorer")
st.markdown("Explore engineered features for ML model training")
st.markdown("---")

# Data loading functions
@st.cache_data
def get_available_files():
    """Get list of available feature files"""
    feature_dir = Path("../data/nfl/cleaned")
    if not feature_dir.exists():
        return []
    
    files = sorted(feature_dir.glob("features_*.parquet"))
    file_info = []
    
    for f in files:
        parts = f.stem.split('_')
        if len(parts) >= 4:
            season = int(parts[1])
            week = int(parts[3])
            file_info.append({
                'file': f,
                'season': season,
                'week': week,
                'label': f"Season {season} - Week {week}"
            })
    
    return file_info

@st.cache_data
def load_feature_data(filepath):
    """Load a specific feature file"""
    try:
        df = pd.read_parquet(filepath)
        return df
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return pd.DataFrame()

@st.cache_data
def load_all_data():
    """Load all feature files into one dataframe"""
    all_files = get_available_files()
    if not all_files:
        return pd.DataFrame()
    
    dfs = []
    for file_info in all_files:
        df = load_feature_data(file_info['file'])
        if not df.empty:
            dfs.append(df)
    
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return pd.DataFrame()

# Sidebar for file selection
with st.sidebar:
    st.header("📁 Data Selection")
    
    files = get_available_files()
    
    if not files:
        st.error("No feature files found in ./data/nfl/cleaned/")
        st.stop()
    
    st.success(f"Found {len(files)} feature files")
    
    # View mode selection
    view_mode = st.radio(
        "View Mode",
        ["Single Week", "All Data", "Player History"]
    )
    
    if view_mode == "Single Week":
        selected_file = st.selectbox(
            "Select Week",
            options=files,
            format_func=lambda x: x['label']
        )
        
        df = load_feature_data(selected_file['file'])
        st.info(f"Loaded {len(df)} players from {selected_file['label']}")
    
    elif view_mode == "All Data":
        df = load_all_data()
        st.info(f"Loaded {len(df)} total records")
    
    else:  # Player History
        df = load_all_data()
        st.info(f"Loaded {len(df)} total records")

# Main content area
if view_mode == "Single Week":
    st.header(f"📊 Week Analysis: {selected_file['label']}")
    
    # Basic stats
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Players", len(df))
    with col2:
        st.metric("With Sufficient Data", df['has_sufficient_data'].sum())
    with col3:
        st.metric("Positions", df['position'].nunique())
    with col4:
        st.metric("Teams", df['team'].nunique())
    
    st.markdown("---")
    
    # Position filter
    col1, col2 = st.columns([1, 3])
    with col1:
        positions = st.multiselect(
            "Filter by Position",
            options=sorted(df['position'].unique()),
            default=sorted(df['position'].unique())
        )
    
    with col2:
        search_name = st.text_input("Search Player Name", "")
    
    # Apply filters
    filtered_df = df[df['position'].isin(positions)]
    if search_name:
        filtered_df = filtered_df[
            filtered_df['player_name'].str.contains(search_name, case=False, na=False)
        ]
    
    # Display options
    st.markdown("### 📋 Feature Data")
    
    # Column selection
    feature_cols = [col for col in df.columns if 'rolling_avg' in col or 'trend' in col or 'rank' in col]
    base_cols = ['player_name', 'position', 'team', 'opponent_team', 'games_in_history', 'has_sufficient_data']
    
    col1, col2 = st.columns([1, 1])
    with col1:
        show_all_cols = st.checkbox("Show All Columns", value=False)
    with col2:
        only_sufficient_data = st.checkbox("Only Players with Sufficient Data", value=False)
    
    if only_sufficient_data:
        filtered_df = filtered_df[filtered_df['has_sufficient_data'] == True]
    
    if show_all_cols:
        display_df = filtered_df
    else:
        # Show selected columns based on position
        if len(filtered_df) > 0:
            # Get position-specific columns
            selected_cols = base_cols.copy()
            
            pos_counts = filtered_df['position'].value_counts()
            if len(pos_counts) == 1:  # Single position selected
                pos = pos_counts.index[0]
                if pos == 'QB':
                    selected_cols.extend(['rolling_avg_passing_yds', 'rolling_avg_passing_tds', 
                                         'rolling_avg_completions', 'opponent_pass_defense_rank'])
                elif pos == 'RB':
                    selected_cols.extend(['rolling_avg_rushing_yds', 'rolling_avg_rushing_tds',
                                         'rolling_avg_carries', 'carry_share_trend'])
                elif pos == 'WR':
                    selected_cols.extend(['rolling_avg_receiving_yds', 'rolling_avg_targets',
                                         'target_share_trend', 'rolling_avg_air_yards'])
                elif pos == 'TE':
                    selected_cols.extend(['rolling_avg_receiving_yds', 'rolling_avg_targets',
                                         'target_share_trend'])
                elif pos == 'K':
                    selected_cols.extend(['rolling_avg_fg_made', 'rolling_avg_fg_att',
                                         'rolling_avg_pat_made'])
            else:
                # Multiple positions - show fantasy points
                selected_cols.extend(['rolling_avg_fantasy_pts', 'rolling_avg_fantasy_ppr'])
            
            # Only include columns that exist
            selected_cols = [col for col in selected_cols if col in filtered_df.columns]
            display_df = filtered_df[selected_cols]
        else:
            display_df = filtered_df
    
    # Display the data
    st.dataframe(
        display_df.style.format(
            {col: '{:.2f}' for col in display_df.select_dtypes(include=[np.number]).columns}
        ),
        height=400,
        use_container_width=True
    )
    
    st.info(f"Showing {len(display_df)} players")
    
    # Data quality analysis
    st.markdown("---")
    st.markdown("### 📊 Data Quality Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Missing data heatmap
        st.markdown("#### Missing Data by Column")
        
        # Calculate missing percentages
        missing_pct = (filtered_df.isnull().sum() / len(filtered_df) * 100).sort_values(ascending=False)
        missing_df = pd.DataFrame({
            'Column': missing_pct.index[:15],  # Top 15 columns with most missing
            'Missing %': missing_pct.values[:15]
        })
        
        if len(missing_df) > 0:
            fig = px.bar(missing_df, x='Missing %', y='Column', orientation='h',
                        title="Top 15 Columns with Missing Data")
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Games in history distribution
        st.markdown("#### Games in History Distribution")
        
        fig = px.histogram(filtered_df, x='games_in_history', 
                          title="Distribution of Historical Games Available")
        fig.update_layout(
            xaxis_title="Number of Games in History",
            yaxis_title="Player Count"
        )
        st.plotly_chart(fig, use_container_width=True)

elif view_mode == "All Data":
    st.header("📊 All Data Overview")
    
    # Overall stats
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", len(df))
    with col2:
        st.metric("Unique Players", df['player_id'].nunique())
    with col3:
        st.metric("Seasons", df['season'].nunique())
    with col4:
        st.metric("Weeks Covered", df.groupby(['season', 'week']).size().shape[0])
    
    st.markdown("---")
    
    # Data coverage heatmap
    st.markdown("### 📅 Data Coverage Heatmap")
    
    # Create pivot table for coverage
    coverage = df.groupby(['season', 'week'])['player_id'].count().reset_index()
    coverage_pivot = coverage.pivot(index='week', columns='season', values='player_id')
    
    fig = px.imshow(coverage_pivot, 
                    labels=dict(x="Season", y="Week", color="Player Count"),
                    title="Players per Week/Season",
                    color_continuous_scale="Viridis")
    st.plotly_chart(fig, use_container_width=True)
    
    # Position breakdown
    st.markdown("### 🏈 Position Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        pos_counts = df.groupby('position')['player_id'].nunique().sort_values(ascending=False)
        fig = px.bar(x=pos_counts.index, y=pos_counts.values,
                    labels={'x': 'Position', 'y': 'Unique Players'},
                    title="Players by Position")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        sufficient_data = df.groupby('position')['has_sufficient_data'].mean() * 100
        fig = px.bar(x=sufficient_data.index, y=sufficient_data.values,
                    labels={'x': 'Position', 'y': '% with Sufficient Data'},
                    title="Data Quality by Position")
        st.plotly_chart(fig, use_container_width=True)
    
    # Sample data
    st.markdown("### 📋 Sample Data")
    st.dataframe(df.head(100), use_container_width=True)

else:  # Player History mode
    st.header("👤 Player History Analysis")
    
    # Player selection
    all_players = sorted(df['player_name'].unique())
    selected_player = st.selectbox(
        "Select Player",
        options=all_players,
        index=0
    )
    
    # Get player data
    player_df = df[df['player_name'] == selected_player].sort_values(['season', 'week'])
    
    if len(player_df) == 0:
        st.warning("No data found for selected player")
    else:
        # Player info
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Position", player_df['position'].iloc[0])
        with col2:
            st.metric("Current Team", player_df['team'].iloc[-1])
        with col3:
            st.metric("Games", len(player_df))
        with col4:
            avg_sufficient = player_df['has_sufficient_data'].mean() * 100
            st.metric("% Sufficient Data", f"{avg_sufficient:.1f}%")
        
        st.markdown("---")
        
        # Feature trends
        st.markdown("### 📈 Feature Trends")
        
        # Create week labels
        player_df['week_label'] = player_df['season'].astype(str) + '-W' + player_df['week'].astype(str)
        
        # Get numeric columns
        numeric_cols = [col for col in player_df.columns 
                       if col.startswith('rolling_avg_') or col.endswith('_trend') or col.endswith('_rank')]
        numeric_cols = [col for col in numeric_cols if not player_df[col].isna().all()]
        
        if numeric_cols:
            selected_metric = st.selectbox(
                "Select Metric to Plot",
                options=numeric_cols,
                index=0
            )
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=player_df['week_label'],
                y=player_df[selected_metric],
                mode='lines+markers',
                name=selected_metric
            ))
            fig.update_layout(
                title=f"{selected_player} - {selected_metric}",
                xaxis_title="Week",
                yaxis_title=selected_metric
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Show raw data
        st.markdown("### 📋 Player Data")
        display_cols = ['week_label', 'team', 'opponent_team', 'games_in_history'] + numeric_cols[:5]
        st.dataframe(
            player_df[display_cols].style.format(
                {col: '{:.2f}' for col in numeric_cols[:5]}
            ),
            use_container_width=True
        )

# Footer
st.markdown("---")
st.markdown("🔍 **Feature Data Explorer** | Use this to verify feature engineering output before ML training")