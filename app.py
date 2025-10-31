"""
File: app.py

Streamlit dashboard for NFL player analytics
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

# Add src directory to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from nfl.nfl_pipeline import NFLDataPipeline
from nfl.column_mappings import COLUMN_DISPLAY_NAMES


# Page configuration
st.set_page_config(
    page_title="NFL Analytics Platform",
    page_icon="🏈",
    layout="wide"
)

# Initialize pipeline
@st.cache_resource
def get_pipeline():
    return NFLDataPipeline()

pipeline = get_pipeline()


# Helper function to load all data for a season
@st.cache_data
def load_season_data(season):
    """Load all weeks for a given season"""
    all_data = []
    
    for week in range(1, 19):  # Weeks 1-18
        if pipeline.check_file_exists(season, week):
            filepath = f"{pipeline.raw_dir}/player_stats_{season}_week_{week}.parquet"
            df = pd.read_parquet(filepath)
            all_data.append(df)
    
    if all_data:
        return pd.concat(all_data, ignore_index=True)
    return pd.DataFrame()


# Helper function to get available seasons
def get_available_seasons():
    """Get list of seasons with downloaded data"""
    parquet_files = list(Path(pipeline.raw_dir).glob("player_stats_*.parquet"))
    seasons = set()
    
    for file in parquet_files:
        parts = file.stem.split('_')
        if len(parts) >= 3:
            try:
                season = int(parts[2])
                seasons.add(season)
            except ValueError:
                continue
    
    return sorted(list(seasons), reverse=True)


# Main App
st.title("🏈 NFL Player Analytics Platform")
st.markdown("---")

# Sidebar for data management
with st.sidebar:
    st.header("⚙️ Data Management")
    
    # Show current data status
    last_season, last_week = pipeline.get_last_downloaded_week()
    st.info(f"**Latest Data:**\nSeason {last_season}, Week {last_week}")
    
    st.markdown("---")
    
    # Fetch Latest Data Button
    if st.button("📥 Fetch Latest Data", type="primary"):
        with st.spinner("Checking for new data..."):
            # Capture pipeline output
            import io
            from contextlib import redirect_stdout
            
            output_buffer = io.StringIO()
            
            # Track starting point
            start_season, start_week = pipeline.get_last_downloaded_week()
            
            # Run the incremental update logic
            current_season = start_season
            current_week = start_week + 1
            
            if current_week > 18:
                current_season += 1
                current_week = 1
            
            total_fetched = 0
            found_data = True
            
            status_placeholder = st.empty()
            
            while found_data and current_season <= 2025:
                if pipeline.check_file_exists(current_season, current_week):
                    current_week += 1
                    if current_week > 18:
                        current_season += 1
                        current_week = 1
                    continue
                
                try:
                    status_placeholder.write(f"Checking Season {current_season}, Week {current_week}...")
                    
                    with redirect_stdout(output_buffer):
                        result = pipeline.run_pipeline(
                            season=current_season, 
                            week=current_week, 
                            silent_check=True
                        )
                    
                    if result is None:
                        status_placeholder.write(f"Week {current_week} hasn't started yet.")
                        found_data = False
                    else:
                        total_fetched += 1
                        status_placeholder.success(f"✅ Downloaded Season {current_season}, Week {current_week}")
                        
                        current_week += 1
                        if current_week > 18:
                            current_season += 1
                            current_week = 1
                
                except Exception as e:
                    status_placeholder.error(f"Error: {str(e)}")
                    found_data = False
            
            # Final status
            if total_fetched > 0:
                st.success(f"✅ Downloaded {total_fetched} new week(s)!")
                st.cache_data.clear()  # Clear cache to reload new data
            else:
                st.info("✓ You're up to date! No new data available.")

st.markdown("---")

# Main content tabs
tab1, tab2 = st.tabs(["📊 Player Data Explorer", "📈 Performance Trends"])

# TAB 1: Player Data Explorer
with tab1:
    st.header("Player Data Explorer")
    
    # Filters in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        available_seasons = get_available_seasons()
        if available_seasons:
            default_season = 2025 if 2025 in available_seasons else available_seasons[0]
            selected_season = st.selectbox("Season", available_seasons, index=available_seasons.index(default_season))
        else:
            st.error("No data available. Please fetch data first.")
            st.stop()
    
    # Load data for selected season
    season_data = load_season_data(selected_season)
    
    # Get most recent week as default
    most_recent_week = season_data['week'].max() if not season_data.empty else 1
    
    with col4:
        available_weeks = sorted(season_data['week'].unique().tolist(), reverse=True)
        selected_week = st.selectbox("Week", available_weeks, index=0)  # Default to most recent
    
    if season_data.empty:
        st.warning(f"No data available for {selected_season} season.")
        st.stop()
    
    # Filter by selected week first
    season_data = season_data[season_data['week'] == selected_week]
    
    # Filter out players with minimal stats (at least 1 fantasy point or any stat > 0)
    season_data = season_data[
        (season_data['fantasy_points_ppr'] > 0) | 
        (season_data['passing_yards'] > 0) |
        (season_data['rushing_yards'] > 0) |
        (season_data['receiving_yards'] > 0)
    ]
    
    with col2:
        # Get unique positions
        positions = sorted(season_data['position'].dropna().unique().tolist())
        selected_position = st.selectbox("Position", ["All"] + positions)
    
    with col3:
        # Filter teams based on position selection
        if selected_position != "All":
            filtered_for_teams = season_data[season_data['position'] == selected_position]
        else:
            filtered_for_teams = season_data
        
        teams = sorted(filtered_for_teams['team'].dropna().unique().tolist())
        selected_team = st.selectbox("Team", ["All"] + teams)
    
    # Apply filters
    filtered_data = season_data.copy()
    
    if selected_position != "All":
        filtered_data = filtered_data[filtered_data['position'] == selected_position]
    
    if selected_team != "All":
        filtered_data = filtered_data[filtered_data['team'] == selected_team]
    
    # Sort by fantasy points (highest to lowest) to show best performers first
    filtered_data = filtered_data.sort_values('fantasy_points_ppr', ascending=False)
    
    # Column selection
    st.markdown("### Select Columns to Display")
    
    # Define column groups (using actual column names from data)
    basic_cols = ['player_name', 'position', 'team', 'week']
    passing_cols = ['completions', 'attempts', 'passing_yards', 'passing_tds', 'passing_interceptions', 'passing_epa']
    rushing_cols = ['carries', 'rushing_yards', 'rushing_tds', 'rushing_epa']
    receiving_cols = ['receptions', 'targets', 'receiving_yards', 'receiving_tds', 'target_share']
    fantasy_cols = ['fantasy_points', 'fantasy_points_ppr']
    
    # Multi-select for column groups
    col_select_1, col_select_2, col_select_3 = st.columns(3)
    
    with col_select_1:
        show_passing = st.checkbox("Passing Stats", value=True)
        show_rushing = st.checkbox("Rushing Stats", value=True)
    
    with col_select_2:
        show_receiving = st.checkbox("Receiving Stats", value=True)
        show_fantasy = st.checkbox("Fantasy Stats", value=True)
    
    with col_select_3:
        show_advanced = st.checkbox("Advanced Metrics", value=False)
    
    # Build column list
    display_cols = basic_cols.copy()
    
    if show_passing:
        display_cols.extend([col for col in passing_cols if col in filtered_data.columns])
    if show_rushing:
        display_cols.extend([col for col in rushing_cols if col in filtered_data.columns])
    if show_receiving:
        display_cols.extend([col for col in receiving_cols if col in filtered_data.columns])
    if show_fantasy:
        display_cols.extend([col for col in fantasy_cols if col in filtered_data.columns])
    if show_advanced:
        # Add all other columns not already included
        advanced_cols = [col for col in filtered_data.columns if col not in display_cols and col not in basic_cols]
        display_cols.extend(advanced_cols)
    
    # Remove duplicates while preserving order
    display_cols = list(dict.fromkeys(display_cols))
    
    # Filter to only columns that exist
    display_cols = [col for col in display_cols if col in filtered_data.columns]
    
    # Display data with renamed columns
    st.markdown(f"### Week {selected_week} Results: {len(filtered_data)} players")
    
    # Create display dataframe with renamed columns
    display_df = filtered_data[display_cols].copy()
    display_df = display_df.rename(columns=COLUMN_DISPLAY_NAMES)
    
    # Show dataframe
    st.dataframe(
        display_df,
        width='stretch',
        height=400
    )
    
    # Quick stats summary
    st.markdown("### Quick Stats Summary")
    summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
    
    with summary_col1:
        st.metric("Total Players", filtered_data['player_name'].nunique())
    
    with summary_col2:
        st.metric("Total Records", len(filtered_data))
    
    with summary_col3:
        if 'fantasy_points_ppr' in filtered_data.columns:
            avg_fantasy = filtered_data['fantasy_points_ppr'].mean()
            st.metric("Avg Fantasy Pts (PPR)", f"{avg_fantasy:.2f}")
    
    with summary_col4:
        st.metric("Weeks Available", filtered_data['week'].nunique())


# TAB 2: Performance Trends
with tab2:
    st.header("Player Performance Trends")
    
    # Season selection
    available_seasons = get_available_seasons()
    if not available_seasons:
        st.error("No data available. Please fetch data first.")
        st.stop()
    
    trend_season = st.selectbox("Select Season", available_seasons, index=0, key="trend_season")
    
    # Load season data
    trend_data = load_season_data(trend_season)
    
    if trend_data.empty:
        st.warning(f"No data available for {trend_season} season.")
        st.stop()
    
    # Calculate total stats per player for sorting
    player_stats = trend_data.groupby('player_name').agg({
        'fantasy_points_ppr': 'sum',
        'position': 'first',
        'team': 'first'
    }).reset_index()
    player_stats = player_stats.sort_values('fantasy_points_ppr', ascending=False)
    
    # Player search/filter section
    st.markdown("### Find Player")
    
    search_col1, search_col2, search_col3 = st.columns(3)
    
    with search_col1:
        # Position filter
        positions = sorted(trend_data['position'].dropna().unique().tolist())
        filter_position = st.selectbox("Filter by Position", ["All Positions"] + positions, key="filter_pos")
    
    with search_col2:
        # Team filter
        teams = sorted(trend_data['team'].dropna().unique().tolist())
        filter_team = st.selectbox("Filter by Team", ["All Teams"] + teams, key="filter_team")
    
    with search_col3:
        # Sort by
        sort_options = ["Fantasy Pts (PPR)", "Passing Yds", "Rushing Yds", "Receiving Yds"]
        sort_by = st.selectbox("Sort By", sort_options)
    
    # Apply filters to player list
    filtered_players = player_stats.copy()
    
    if filter_position != "All Positions":
        filtered_players = filtered_players[filtered_players['position'] == filter_position]
    
    if filter_team != "All Teams":
        filtered_players = filtered_players[filtered_players['team'] == filter_team]
    
    # Sort by selected metric
    sort_metric_map = {
        "Fantasy Pts (PPR)": 'fantasy_points_ppr',
        "Passing Yds": 'passing_yards',
        "Rushing Yds": 'rushing_yards',
        "Receiving Yds": 'receiving_yards'
    }
    
    if sort_by != "Fantasy Pts (PPR)":
        metric = sort_metric_map[sort_by]
        player_totals = trend_data.groupby('player_name')[metric].sum().reset_index()
        filtered_players = filtered_players.merge(player_totals, on='player_name', how='left')
        filtered_players = filtered_players.sort_values(metric, ascending=False)
    
    # Player selection with search
    st.markdown(f"**{len(filtered_players)} players found** (sorted by {sort_by})")
    
    # Create player display list with stats
    player_display_list = []
    for _, row in filtered_players.head(100).iterrows():  # Limit to top 100
        display_str = f"{row['player_name']} ({row['position']}, {row['team']}) - {row['fantasy_points_ppr']:.1f} pts"
        player_display_list.append(display_str)
    
    selected_player_display = st.selectbox("Select Player", player_display_list)
    
    # Extract player name from display string
    selected_player = selected_player_display.split(' (')[0]
    
    # Filter to selected player
    player_data = trend_data[trend_data['player_name'] == selected_player].sort_values('week')
    
    if player_data.empty:
        st.warning("No data available for selected player.")
        st.stop()
    
    # Display player info
    st.markdown(f"### {selected_player}")
    info_col1, info_col2, info_col3 = st.columns(3)
    
    with info_col1:
        position = player_data['position'].iloc[0]
        st.write(f"**Position:** {position}")
    
    with info_col2:
        team = player_data['team'].iloc[0]
        st.write(f"**Team:** {team}")
    
    with info_col3:
        games_played = len(player_data)
        st.write(f"**Games Played:** {games_played}")
    
    st.markdown("---")
    
    # Metric selection for trends
    metric_options = {
        "Fantasy Points (PPR)": "fantasy_points_ppr",
        "Fantasy Points": "fantasy_points",
        "Passing Yards": "passing_yards",
        "Passing TDs": "passing_tds",
        "Rushing Yards": "rushing_yards",
        "Rushing TDs": "rushing_tds",
        "Receptions": "receptions",
        "Receiving Yards": "receiving_yards",
        "Receiving TDs": "receiving_tds",
        "Targets": "targets"
    }
    
    # Filter metrics to only those with data
    available_metrics = {k: v for k, v in metric_options.items() if v in player_data.columns and player_data[v].notna().any()}
    
    selected_metric_name = st.selectbox("Select Metric", list(available_metrics.keys()))
    selected_metric = available_metrics[selected_metric_name]
    
    # Create trend chart
    fig = go.Figure()
    
    # Filter out weeks with 0 values for cleaner line
    chart_data = player_data[player_data[selected_metric] > 0].copy()
    
    if len(chart_data) == 0:
        st.warning(f"No data available for {selected_metric_name}")
    else:
        # Add main line
        fig.add_trace(go.Scatter(
            x=chart_data['week'],
            y=chart_data[selected_metric],
            mode='lines+markers',
            name=selected_metric_name,
            line=dict(color='#1f77b4', width=3),
            marker=dict(size=10)
        ))
        
        # Add average line
        avg_value = chart_data[selected_metric].mean()
        fig.add_hline(
            y=avg_value, 
            line_dash="dash", 
            line_color="red",
            annotation_text=f"Average: {avg_value:.2f}",
            annotation_position="right"
        )
        
        fig.update_layout(
            title=f"{selected_player} - {selected_metric_name} by Week",
            xaxis_title="Week",
            yaxis_title=selected_metric_name,
            hovermode='x unified',
            height=500
        )
        
        st.plotly_chart(fig, width='stretch')
    
    # Stats table
    st.markdown("### Weekly Performance")
    
    # Select relevant columns for display
    trend_display_cols = ['week', selected_metric]
    
    # Add related stats based on metric
    if 'passing' in selected_metric:
        related_cols = ['completions', 'attempts', 'passing_yards', 'passing_tds', 'passing_interceptions']
    elif 'rushing' in selected_metric:
        related_cols = ['carries', 'rushing_yards', 'rushing_tds']
    elif 'receiving' in selected_metric or selected_metric in ['receptions', 'targets']:
        related_cols = ['receptions', 'targets', 'receiving_yards', 'receiving_tds']
    else:
        related_cols = ['fantasy_points', 'fantasy_points_ppr']
    
    # Add existing related columns
    for col in related_cols:
        if col in player_data.columns and col not in trend_display_cols:
            trend_display_cols.append(col)
    
    # Create display dataframe with renamed columns
    trend_display_df = player_data[trend_display_cols].copy()
    trend_display_df = trend_display_df.rename(columns=COLUMN_DISPLAY_NAMES)
    
    st.dataframe(
        trend_display_df.sort_values('Week', ascending=False),
        width='stretch',
        height=300
    )
    
    # Summary stats
    st.markdown("### Season Summary")
    summary_metrics_col1, summary_metrics_col2, summary_metrics_col3, summary_metrics_col4 = st.columns(4)
    
    with summary_metrics_col1:
        total = player_data[selected_metric].sum()
        st.metric("Total", f"{total:.1f}")
    
    with summary_metrics_col2:
        avg = player_data[selected_metric].mean()
        st.metric("Average", f"{avg:.2f}")
    
    with summary_metrics_col3:
        max_val = player_data[selected_metric].max()
        st.metric("Best Game", f"{max_val:.1f}")
    
    with summary_metrics_col4:
        min_val = player_data[selected_metric].min()
        st.metric("Worst Game", f"{min_val:.1f}")


# Footer
st.markdown("---")
st.markdown("🏈 **NFL Analytics Platform** | Data from nflreadpy | Senior Design Project")