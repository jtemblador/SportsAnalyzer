
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


# Position full names mapping
POSITION_NAMES = {
    'QB': 'Quarterback',
    'RB': 'Running Back',
    'WR': 'Wide Receiver',
    'TE': 'Tight End',
    'K': 'Kicker',
    'DEF': 'Defense',
    'FB': 'Fullback',
}

# NFL Team full names mapping
TEAM_NAMES = {
    'ARI': 'Arizona Cardinals', 'ATL': 'Atlanta Falcons', 'BAL': 'Baltimore Ravens',
    'BUF': 'Buffalo Bills', 'CAR': 'Carolina Panthers', 'CHI': 'Chicago Bears',
    'CIN': 'Cincinnati Bengals', 'CLE': 'Cleveland Browns', 'DAL': 'Dallas Cowboys',
    'DEN': 'Denver Broncos', 'DET': 'Detroit Lions', 'GB': 'Green Bay Packers',
    'HOU': 'Houston Texans', 'IND': 'Indianapolis Colts', 'JAX': 'Jacksonville Jaguars',
    'KC': 'Kansas City Chiefs', 'LA': 'Los Angeles Rams', 'LAC': 'Los Angeles Chargers',
    'LAR': 'Los Angeles Rams', 'LV': 'Las Vegas Raiders', 'MIA': 'Miami Dolphins',
    'MIN': 'Minnesota Vikings', 'NE': 'New England Patriots', 'NO': 'New Orleans Saints',
    'NYG': 'New York Giants', 'NYJ': 'New York Jets', 'OAK': 'Oakland Raiders',
    'PHI': 'Philadelphia Eagles', 'PIT': 'Pittsburgh Steelers', 'SEA': 'Seattle Seahawks',
    'SF': 'San Francisco 49ers', 'TB': 'Tampa Bay Buccaneers', 'TEN': 'Tennessee Titans',
    'WAS': 'Washington Commanders', 'WSH': 'Washington Commanders',
}


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


# Helper function to load V4 predictions
@st.cache_data
def load_v4_predictions(week):
    """Load V4 predictions for specified week"""
    path = Path(__file__).parent / f"data/nfl/predictions/v4_position_specific/predictions_2025_week_{week}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()


# Helper function to get available prediction weeks
def get_available_prediction_weeks():
    """Get list of weeks with V4 predictions"""
    pred_dir = Path(__file__).parent / "data/nfl/predictions/v4_position_specific"
    if not pred_dir.exists():
        return []
    weeks = []
    for f in pred_dir.glob("predictions_2025_week_*.parquet"):
        try:
            week = int(f.stem.split('_')[-1])
            weeks.append(week)
        except ValueError:
            continue
    return sorted(weeks, reverse=True)


# Helper function to load Vegas odds
@st.cache_data
def load_vegas_lines(week):
    """Load Vegas team lines for specified week"""
    vegas_dir = Path(__file__).parent / "data/nfl/vegas_odds/team_lines"
    if not vegas_dir.exists():
        return pd.DataFrame()
    # Find the most recent file for this week
    files = list(vegas_dir.glob(f"team_lines_week_{week}_*.parquet"))
    if files:
        return pd.read_parquet(files[0])
    return pd.DataFrame()


# Main App
st.title("🏈 NFL Player Analytics Platform")

# Sidebar for data management
with st.sidebar:
    st.header("⚙️ Data Management")

    # Show current data status
    last_season, last_week = pipeline.get_last_downloaded_week()
    st.info(f"**Latest Data:**\nSeason {last_season}, Week {last_week}")

    st.markdown("---")

    # Model Information Section
    st.header("🤖 Model Info")

    # Get prediction weeks to show data availability
    pred_weeks_sidebar = get_available_prediction_weeks()
    latest_pred_week = pred_weeks_sidebar[0] if pred_weeks_sidebar else "N/A"

    st.success("**V4 Production Model**\nVegas + Position-Specific")

    # Quick stats in sidebar
    st.metric("Model MAE", "4.26", "-17% vs baseline", delta_color="inverse")

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.metric("Predictions", f"Wk {latest_pred_week}")
    with col_s2:
        st.metric("Status", "Ready")

    st.markdown("---")

    # Performance summary
    st.markdown("**Position Accuracy:**")
    st.markdown("""
    - QB: 4.67 MAE
    - RB: 4.41 MAE
    - WR: 5.06 MAE
    - TE: 2.34 MAE
    """)

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
tab_trends, tab_explorer, tab_predictions, tab_model = st.tabs(["📈 Performance Trends", "📊 Player Data Explorer", "🔮 Predictions", "📊 Model Performance"])

# TAB: Player Data Explorer
with tab_explorer:
    st.header("Player Data Explorer")
    st.caption("Browse actual player stats from completed games")

    # Load data first
    available_seasons = get_available_seasons()
    if not available_seasons:
        st.error("No data available. Please fetch data first.")
        st.stop()

    default_season = 2025 if 2025 in available_seasons else available_seasons[0]
    season_data = load_season_data(default_season)

    if season_data.empty:
        st.warning(f"No data available for {default_season} season.")
        st.stop()

    # Get most recent week
    available_weeks = sorted(season_data['week'].unique().tolist(), reverse=True)

    # Compact filter row
    filter_col1, filter_col2, filter_col3, filter_col4 = st.columns([1, 1, 1, 2])

    with filter_col1:
        selected_week = st.selectbox("Week", available_weeks, index=0, key="explorer_week")

    # Filter by selected week
    week_data = season_data[season_data['week'] == selected_week].copy()

    # Filter out players with minimal stats
    week_data = week_data[
        (week_data['fantasy_points_ppr'] > 0) |
        (week_data['passing_yards'] > 0) |
        (week_data['rushing_yards'] > 0) |
        (week_data['receiving_yards'] > 0)
    ]

    with filter_col2:
        positions = sorted(week_data['position'].dropna().unique().tolist())
        position_options = ["All Positions"] + [POSITION_NAMES.get(p, p) for p in positions]
        selected_pos_display = st.selectbox("Position", position_options, key="explorer_pos")

        if selected_pos_display == "All Positions":
            selected_position = "All"
        else:
            selected_position = [k for k, v in POSITION_NAMES.items() if v == selected_pos_display]
            selected_position = selected_position[0] if selected_position else selected_pos_display

    with filter_col3:
        if selected_position != "All":
            filtered_for_teams = week_data[week_data['position'] == selected_position]
        else:
            filtered_for_teams = week_data
        teams = sorted(filtered_for_teams['team'].dropna().unique().tolist())
        team_options = ["All Teams"] + [TEAM_NAMES.get(t, t) for t in teams]
        selected_team_display = st.selectbox("Team", team_options, key="explorer_team")

        if selected_team_display == "All Teams":
            selected_team = "All"
        else:
            selected_team = [k for k, v in TEAM_NAMES.items() if v == selected_team_display]
            selected_team = selected_team[0] if selected_team else selected_team_display

    with filter_col4:
        player_search_tab1 = st.text_input("Search Player", placeholder="Type player name...", key="explorer_search")

    # Apply filters
    filtered_data = week_data.copy()
    if selected_position != "All":
        filtered_data = filtered_data[filtered_data['position'] == selected_position]
    if selected_team != "All":
        filtered_data = filtered_data[filtered_data['team'] == selected_team]
    if player_search_tab1:
        filtered_data = filtered_data[
            filtered_data['player_name'].str.lower().str.contains(player_search_tab1.lower(), na=False)
        ]

    # Sort by fantasy points
    filtered_data = filtered_data.sort_values('fantasy_points_ppr', ascending=False)

    # Auto-select columns based on position (no checkboxes)
    basic_cols = ['player_name', 'position', 'team']
    fantasy_cols = ['fantasy_points_ppr']

    if selected_position == "QB":
        stat_cols = ['completions', 'attempts', 'passing_yards', 'passing_tds', 'passing_interceptions', 'passing_epa']
    elif selected_position == "RB":
        stat_cols = ['carries', 'rushing_yards', 'rushing_tds', 'receptions', 'receiving_yards', 'rushing_epa']
    elif selected_position in ["WR", "TE"]:
        stat_cols = ['receptions', 'targets', 'receiving_yards', 'receiving_tds', 'target_share']
    else:
        stat_cols = ['passing_yards', 'passing_tds', 'rushing_yards', 'rushing_tds', 'receiving_yards', 'receiving_tds', 'receptions']

    display_cols = basic_cols + [c for c in stat_cols if c in filtered_data.columns] + fantasy_cols
    display_cols = [c for c in display_cols if c in filtered_data.columns]

    # Create display dataframe
    display_df = filtered_data[display_cols].copy()
    display_df = display_df.rename(columns=COLUMN_DISPLAY_NAMES)

    # Display table with count
    st.caption(f"Week {selected_week} | {len(filtered_data)} players")
    st.dataframe(
        display_df,
        use_container_width=True,
        height=500,
        hide_index=True
    )

    # Expandable section for top performers
    with st.expander("⭐ Top Performers This Week", expanded=False):
        top_cols = st.columns(4)
        for i, pos in enumerate(['QB', 'RB', 'WR', 'TE']):
            with top_cols[i]:
                pos_full = POSITION_NAMES.get(pos, pos)
                st.markdown(f"**{pos_full}**")
                pos_top = week_data[week_data['position'] == pos].nlargest(3, 'fantasy_points_ppr')
                if not pos_top.empty:
                    for _, row in pos_top.iterrows():
                        st.write(f"{row['player_name']}: **{row['fantasy_points_ppr']:.1f}**")
                else:
                    st.write("No data")


# TAB: Performance Trends
with tab_trends:
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
        # Position filter with full names
        positions = sorted(trend_data['position'].dropna().unique().tolist())
        position_display_options = ["All Positions"] + [POSITION_NAMES.get(p, p) for p in positions]
        filter_position_display = st.selectbox("Filter by Position", position_display_options, key="filter_pos")

        if filter_position_display == "All Positions":
            filter_position = "All Positions"
        else:
            filter_position = [k for k, v in POSITION_NAMES.items() if v == filter_position_display]
            filter_position = filter_position[0] if filter_position else filter_position_display

    with search_col2:
        # Team filter with full names
        teams = sorted(trend_data['team'].dropna().unique().tolist())
        team_display_options = ["All Teams"] + [TEAM_NAMES.get(t, t) for t in teams]
        filter_team_display = st.selectbox("Filter by Team", team_display_options, key="filter_team")

        if filter_team_display == "All Teams":
            filter_team = "All Teams"
        else:
            filter_team = [k for k, v in TEAM_NAMES.items() if v == filter_team_display]
            filter_team = filter_team[0] if filter_team else filter_team_display
    
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
        st.write(f"**Position:** {POSITION_NAMES.get(position, position)}")

    with info_col2:
        team = player_data['team'].iloc[0]
        st.write(f"**Team:** {TEAM_NAMES.get(team, team)}")

    with info_col3:
        games_played = len(player_data)
        st.write(f"**Games Played:** {games_played}")
    
    st.markdown("---")

    # Get player's position for filtering relevant metrics
    player_position = player_data['position'].iloc[0] if not player_data.empty else None

    # Position-specific metric options (only show relevant stats)
    if player_position == 'QB':
        metric_options = {
            "Fantasy Points (PPR)": "fantasy_points_ppr",
            "Passing Yards": "passing_yards",
            "Passing TDs": "passing_tds",
            "Passing INTs": "passing_interceptions",
            "Passing EPA": "passing_epa",
            "Rushing Yards": "rushing_yards",
            "Rushing TDs": "rushing_tds",
        }
    elif player_position == 'RB':
        metric_options = {
            "Fantasy Points (PPR)": "fantasy_points_ppr",
            "Rushing Yards": "rushing_yards",
            "Rushing TDs": "rushing_tds",
            "Rushing EPA": "rushing_epa",
            "Receptions": "receptions",
            "Receiving Yards": "receiving_yards",
            "Targets": "targets",
        }
    elif player_position == 'WR':
        metric_options = {
            "Fantasy Points (PPR)": "fantasy_points_ppr",
            "Receiving Yards": "receiving_yards",
            "Receiving TDs": "receiving_tds",
            "Receptions": "receptions",
            "Targets": "targets",
            "Target Share": "target_share",
        }
    elif player_position == 'TE':
        metric_options = {
            "Fantasy Points (PPR)": "fantasy_points_ppr",
            "Receiving Yards": "receiving_yards",
            "Receiving TDs": "receiving_tds",
            "Receptions": "receptions",
            "Targets": "targets",
        }
    else:
        # Default for other positions (K, DEF, etc.)
        metric_options = {
            "Fantasy Points (PPR)": "fantasy_points_ppr",
            "Fantasy Points": "fantasy_points",
        }

    # Filter metrics to only those with actual data for this player
    available_metrics = {k: v for k, v in metric_options.items() if v in player_data.columns and player_data[v].notna().any()}
    
    selected_metric_name = st.selectbox("Select Metric", list(available_metrics.keys()))
    selected_metric = available_metrics[selected_metric_name]
    
    # Create trend chart with Predictions vs Actual overlay
    fig = go.Figure()

    # Filter out weeks with 0 values for cleaner line
    chart_data = player_data[player_data[selected_metric] > 0].copy()

    # Stats that have V4 predictions available
    predicted_stats = [
        'fantasy_points_ppr', 'passing_yards', 'passing_tds', 'passing_interceptions',
        'rushing_yards', 'rushing_tds', 'receiving_yards', 'receiving_tds', 'receptions',
        'fg_made', 'fg_att'
    ]

    # Load predictions for this player (for any predicted stat)
    player_predictions = pd.DataFrame()
    if selected_metric in predicted_stats and trend_season == 2025:
        pred_weeks = get_available_prediction_weeks()
        all_preds = []
        for week in pred_weeks:
            week_preds = load_v4_predictions(week)
            if not week_preds.empty:
                player_preds = week_preds[
                    (week_preds['player_name'] == selected_player) &
                    (week_preds['stat'] == selected_metric) &
                    (week_preds['predicted_value'].notna())
                ]
                if not player_preds.empty:
                    all_preds.append(player_preds)
        if all_preds:
            player_predictions = pd.concat(all_preds, ignore_index=True)

    if len(chart_data) == 0:
        st.warning(f"No data available for {selected_metric_name}")
    else:
        # Add actual performance line
        fig.add_trace(go.Scatter(
            x=chart_data['week'],
            y=chart_data[selected_metric],
            mode='lines+markers',
            name='Actual',
            line=dict(color='#1f77b4', width=3),
            marker=dict(size=10)
        ))

        # Add predictions overlay if available
        if not player_predictions.empty:
            # Merge predictions with actual data to calculate accuracy
            pred_with_actual = player_predictions.merge(
                chart_data[['week', selected_metric]],
                on='week',
                how='inner'
            )

            # Sort by week to ensure proper line drawing
            pred_with_actual = pred_with_actual.sort_values('week').reset_index(drop=True)

            if not pred_with_actual.empty:
                # Calculate error for coloring
                pred_with_actual['error'] = abs(pred_with_actual['predicted_value'] - pred_with_actual[selected_metric])

                # Set error thresholds based on stat type (yards need bigger thresholds than TDs)
                if selected_metric in ['passing_yards']:
                    green_threshold, yellow_threshold = 30, 60  # Yards - bigger scale
                elif selected_metric in ['rushing_yards', 'receiving_yards']:
                    green_threshold, yellow_threshold = 15, 30  # Yards - medium scale
                elif selected_metric in ['passing_tds', 'rushing_tds', 'receiving_tds']:
                    green_threshold, yellow_threshold = 0.5, 1.0  # TDs - small scale
                elif selected_metric in ['receptions']:
                    green_threshold, yellow_threshold = 1.5, 3.0  # Receptions
                else:
                    green_threshold, yellow_threshold = 2, 5  # PPR and others

                # Color-code markers based on accuracy
                colors = []
                for err in pred_with_actual['error']:
                    if err < green_threshold:
                        colors.append('#2ecc71')  # Green - accurate
                    elif err < yellow_threshold:
                        colors.append('#f39c12')  # Yellow - moderate
                    else:
                        colors.append('#e74c3c')  # Red - inaccurate

                # Add prediction line
                fig.add_trace(go.Scatter(
                    x=pred_with_actual['week'],
                    y=pred_with_actual['predicted_value'],
                    mode='lines+markers',
                    name='V4 Predicted',
                    line=dict(color='#9b59b6', width=2, dash='dash'),
                    marker=dict(size=12, color=colors, line=dict(width=2, color='white'))
                ))

                # Add confidence interval fill
                if 'confidence_lower' in pred_with_actual.columns and 'confidence_upper' in pred_with_actual.columns:
                    # Check if confidence values exist and are not all NaN
                    if pred_with_actual['confidence_lower'].notna().any() and pred_with_actual['confidence_upper'].notna().any():
                        # Filter to only rows with valid confidence values
                        conf_data = pred_with_actual[
                            pred_with_actual['confidence_lower'].notna() &
                            pred_with_actual['confidence_upper'].notna()
                        ].copy()
                        if not conf_data.empty:
                            fig.add_trace(go.Scatter(
                                x=list(conf_data['week']) + list(conf_data['week'][::-1]),
                                y=list(conf_data['confidence_upper']) + list(conf_data['confidence_lower'][::-1]),
                                fill='toself',
                                fillcolor='rgba(155, 89, 182, 0.15)',
                                line=dict(color='rgba(155, 89, 182, 0)'),
                                name='Confidence Range',
                                showlegend=True
                            ))

        # Add FUTURE prediction point (next week's prediction - no actual data yet)
        if trend_season == 2025 and selected_metric in predicted_stats:
            last_actual_week = int(chart_data['week'].max()) if not chart_data.empty else 0
            next_week = last_actual_week + 1

            # Try to load prediction for next week
            next_week_preds = load_v4_predictions(next_week)
            if not next_week_preds.empty:
                future_pred = next_week_preds[
                    (next_week_preds['player_name'] == selected_player) &
                    (next_week_preds['stat'] == selected_metric) &
                    (next_week_preds['predicted_value'].notna())
                ]
                if not future_pred.empty:
                    future_value = future_pred['predicted_value'].iloc[0]
                    # Add future prediction as a distinct star marker
                    fig.add_trace(go.Scatter(
                        x=[next_week],
                        y=[future_value],
                        mode='markers',
                        name=f'Week {next_week} Prediction',
                        marker=dict(
                            size=18,
                            color='#e74c3c',
                            symbol='star',
                            line=dict(width=2, color='white')
                        ),
                        hovertemplate=f"<b>PREDICTED Week {next_week}</b><br>" +
                                      f"{selected_metric_name}: %{{y:.1f}}<extra></extra>"
                    ))

        # Average line removed - was cluttering the visualization

        fig.update_layout(
            title=f"{selected_player} - {selected_metric_name} by Week",
            xaxis_title="Week",
            yaxis_title=selected_metric_name,
            hovermode='x unified',
            height=500,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        st.plotly_chart(fig, use_container_width=True)

        # Show prediction accuracy summary if predictions available
        if not player_predictions.empty:
            pred_with_actual = player_predictions.merge(
                chart_data[['week', selected_metric]],
                on='week',
                how='inner'
            )
            if not pred_with_actual.empty:
                pred_with_actual['error'] = abs(pred_with_actual['predicted_value'] - pred_with_actual[selected_metric])
                avg_error = pred_with_actual['error'].mean()

                # Use same thresholds as chart coloring
                if selected_metric in ['passing_yards']:
                    green_thresh, yellow_thresh = 30, 60
                    unit = "yds"
                elif selected_metric in ['rushing_yards', 'receiving_yards']:
                    green_thresh, yellow_thresh = 15, 30
                    unit = "yds"
                elif selected_metric in ['passing_tds', 'rushing_tds', 'receiving_tds']:
                    green_thresh, yellow_thresh = 0.5, 1.0
                    unit = "TDs"
                elif selected_metric in ['receptions']:
                    green_thresh, yellow_thresh = 1.5, 3.0
                    unit = "rec"
                else:
                    green_thresh, yellow_thresh = 2, 5
                    unit = "pts"

                accurate_weeks = len(pred_with_actual[pred_with_actual['error'] < green_thresh])

                st.markdown(f"### 📊 Prediction Accuracy for {selected_metric_name}")
                acc_col1, acc_col2, acc_col3 = st.columns(3)
                with acc_col1:
                    st.metric("Avg Prediction Error", f"{avg_error:.1f} {unit}")
                with acc_col2:
                    st.metric(f"Accurate Weeks (<{green_thresh} {unit})", f"{accurate_weeks}/{len(pred_with_actual)}")
                with acc_col3:
                    accuracy_pct = (accurate_weeks / len(pred_with_actual)) * 100 if len(pred_with_actual) > 0 else 0
                    st.metric("Accuracy Rate", f"{accuracy_pct:.0f}%")
                st.markdown(f"*Green = <{green_thresh} error, Yellow = {green_thresh}-{yellow_thresh} error, Red = >{yellow_thresh} error*")
    
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


# TAB: Predictions
with tab_predictions:
    st.header("🔮 V4 Model Predictions")

    # Get available prediction weeks
    pred_weeks = get_available_prediction_weeks()

    if not pred_weeks:
        st.warning("No prediction data available. Run the V4 retrain script to generate predictions.")
        st.stop()

    # Week selector at the top
    selected_pred_week = st.selectbox("Select Week", pred_weeks, index=0, key="pred_week")

    # Load predictions for selected week
    predictions_df = load_v4_predictions(selected_pred_week)

    if predictions_df.empty:
        st.warning(f"No predictions available for week {selected_pred_week}")
        st.stop()

    # Filter to PPR predictions only
    ppr_predictions = predictions_df[predictions_df['stat'] == 'fantasy_points_ppr'].copy()
    ppr_predictions = ppr_predictions[ppr_predictions['predicted_value'].notna()]

    # Load Vegas lines for context
    vegas_lines = load_vegas_lines(selected_pred_week)

    # ========== TOP PREDICTIONS BY POSITION (Featured at top) ==========
    st.markdown(f"### ⭐ Top Players to Watch - Week {selected_pred_week}")
    st.markdown("*Our V4 model's top predicted performers by position*")

    pos_cols = st.columns(4)
    position_order = ['QB', 'RB', 'WR', 'TE']

    for i, pos in enumerate(position_order):
        with pos_cols[i]:
            pos_full_name = POSITION_NAMES.get(pos, pos)
            st.markdown(f"**{pos_full_name}**")
            pos_data = ppr_predictions[ppr_predictions['position'] == pos].nlargest(5, 'predicted_value')
            if not pos_data.empty:
                for _, row in pos_data.iterrows():
                    boom_indicator = "🔥" if row['probability_over'] > 0.55 else ""
                    team = row['team']
                    st.write(f"{row['player_name']} ({team}): **{row['predicted_value']:.1f}** {boom_indicator}")
            else:
                st.write("No predictions")

    st.markdown("---")

    # ========== FULL PREDICTIONS TABLE ==========
    st.markdown("### All Predictions")

    # Filters row
    filter_col1, filter_col2, filter_col3 = st.columns([1, 1, 2])

    with filter_col1:
        positions = sorted(ppr_predictions['position'].dropna().unique().tolist())
        position_display = ["All Positions"] + [POSITION_NAMES.get(p, p) for p in positions]
        selected_pos_display = st.selectbox("Filter by Position", position_display, key="pred_position")
        # Convert back to abbreviation
        if selected_pos_display == "All Positions":
            selected_pred_position = "All"
        else:
            selected_pred_position = [k for k, v in POSITION_NAMES.items() if v == selected_pos_display]
            selected_pred_position = selected_pred_position[0] if selected_pred_position else selected_pos_display

    with filter_col2:
        teams = sorted(ppr_predictions['team'].dropna().unique().tolist())
        team_display = ["All Teams"] + [TEAM_NAMES.get(t, t) for t in teams]
        selected_team_display = st.selectbox("Filter by Team", team_display, key="pred_team")
        # Convert back to abbreviation
        if selected_team_display == "All Teams":
            selected_pred_team = "All"
        else:
            selected_pred_team = [k for k, v in TEAM_NAMES.items() if v == selected_team_display]
            selected_pred_team = selected_pred_team[0] if selected_pred_team else selected_team_display

    with filter_col3:
        player_search = st.text_input("Search Player", placeholder="Type player name...", key="pred_search")

    # Apply filters
    filtered_predictions = ppr_predictions.copy()

    if selected_pred_position != "All":
        filtered_predictions = filtered_predictions[filtered_predictions['position'] == selected_pred_position]

    if selected_pred_team != "All":
        filtered_predictions = filtered_predictions[filtered_predictions['team'] == selected_pred_team]

    if player_search:
        filtered_predictions = filtered_predictions[
            filtered_predictions['player_name'].str.lower().str.contains(player_search.lower(), na=False)
        ]

    # Sort by predicted value (highest first)
    filtered_predictions = filtered_predictions.sort_values('predicted_value', ascending=False)

    # Create display dataframe
    display_predictions = filtered_predictions[['player_name', 'position', 'team', 'opponent',
                                                  'predicted_value', 'confidence_lower', 'confidence_upper',
                                                  'baseline', 'probability_over']].copy()

    # Rename columns for display
    display_predictions = display_predictions.rename(columns={
        'player_name': 'Player',
        'position': 'Pos',
        'team': 'Team',
        'opponent': 'Opp',
        'predicted_value': 'Pred PPR',
        'confidence_lower': 'Low',
        'confidence_upper': 'High',
        'baseline': 'Baseline',
        'probability_over': 'Boom %'
    })

    # Format columns (handle NaN values)
    display_predictions['Pred PPR'] = display_predictions['Pred PPR'].round(1)
    display_predictions['Low'] = display_predictions['Low'].fillna(0).round(1)
    display_predictions['High'] = display_predictions['High'].fillna(0).round(1)
    display_predictions['Baseline'] = display_predictions['Baseline'].round(1)
    display_predictions['Boom %'] = display_predictions['Boom %'].apply(
        lambda x: f"{int(x * 100)}%" if pd.notna(x) else "N/A"
    )

    # Add difference from baseline
    vs_baseline = filtered_predictions['predicted_value'] - filtered_predictions['baseline']
    display_predictions['vs Baseline'] = vs_baseline.round(1).fillna(0)

    # Reorder columns
    display_predictions = display_predictions[['Player', 'Pos', 'Team', 'Opp', 'Pred PPR', 'Low', 'High',
                                                'Baseline', 'vs Baseline', 'Boom %']]

    st.caption(f"Showing {len(display_predictions)} players")

    # Display table
    st.dataframe(
        display_predictions,
        use_container_width=True,
        height=400,
        hide_index=True
    )

    # Vegas Context Section (if data available)
    if not vegas_lines.empty:
        st.markdown("---")
        st.markdown("### Vegas Game Context")
        st.markdown("*Higher implied totals = more fantasy opportunity*")

        vegas_display = vegas_lines[['home_team', 'away_team', 'spread', 'over_under',
                                      'home_implied_total', 'away_implied_total']].copy()
        vegas_display = vegas_display.rename(columns={
            'home_team': 'Home',
            'away_team': 'Away',
            'spread': 'Spread',
            'over_under': 'O/U',
            'home_implied_total': 'Home Pts',
            'away_implied_total': 'Away Pts'
        })

        # Sort by over/under (highest scoring games first)
        vegas_display = vegas_display.sort_values('O/U', ascending=False)

        st.dataframe(
            vegas_display,
            use_container_width=True,
            height=250,
            hide_index=True
        )

    st.markdown("---")
    st.caption("**Legend:** Pred PPR = Predicted Fantasy Points (PPR) | Boom % = Probability of exceeding baseline | 🔥 = High boom potential (>55%)")


# TAB: Model Performance Dashboard
with tab_model:
    st.header("🔮 V4 Model Performance (Production)")
    st.markdown("Model Validation Results | Vegas + Position-Specific Strategy")
    st.markdown("---")

    # Model metrics data
    model_metrics = {
        'overall_mae': 4.26,
        'overall_r2': 0.442,
        'overall_industry_std': '4.5-5.5',
        'positions': {
            'QB': {'mae': 4.67, 'r2': 0.442, 'players': 64, 'improvement': '-29%'},
            'RB': {'mae': 4.41, 'r2': 0.527, 'players': 166, 'improvement': '-3%'},
            'WR': {'mae': 5.06, 'r2': 0.183, 'players': 273, 'improvement': '+1%'},
            'TE': {'mae': 2.34, 'r2': 0.546, 'players': 144, 'improvement': '-62%'}
        }
    }

    # Overall performance metrics
    st.subheader("📈 Overall Performance Summary")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Model Accuracy", f"{model_metrics['overall_mae']:.2f} MAE",
                  "-17% vs baseline", delta_color="inverse")

    with col2:
        st.metric("R² Score", f"{model_metrics['overall_r2']:.3f}",
                  "Model explains 44.2% of variance")

    with col3:
        st.metric("Industry Standard", model_metrics['overall_industry_std'],
                  "✓ EXCEEDED")

    with col4:
        st.metric("Validation Data", "647 players",
                  "Weeks 12-14 held-out test set")

    st.markdown("---")

    # Position-specific performance
    st.subheader("🎯 Position-Specific Performance")

    position_data = []
    for pos, metrics in model_metrics['positions'].items():
        position_data.append({
            'Position': pos,
            'MAE': metrics['mae'],
            'R²': metrics['r2'],
            'Players Validated': metrics['players'],
            'Improvement vs V1': metrics['improvement']
        })

    positions_df = pd.DataFrame(position_data)

    # Display as styled table
    col_table1, col_table2 = st.columns([2, 1])

    with col_table1:
        st.dataframe(
            positions_df,
            use_container_width=True,
            height=250,
            hide_index=True
        )

    with col_table2:
        # Best performing position
        best_pos = min(model_metrics['positions'].items(), key=lambda x: x[1]['mae'])
        st.success(f"**Best:** {best_pos[0]}\n{best_pos[1]['mae']:.2f} MAE\n({best_pos[1]['improvement']} improvement)")

        st.markdown("---")

        # Most validated
        most_validated = max(model_metrics['positions'].items(), key=lambda x: x[1]['players'])
        st.info(f"**Most Validated:** {most_validated[0]}\n{most_validated[1]['players']} players")

    st.markdown("---")

    # Performance comparison chart
    st.subheader("📊 MAE Comparison by Position")

    # Create comparison chart
    fig_mae = px.bar(
        positions_df,
        x='Position',
        y='MAE',
        color='MAE',
        color_continuous_scale=['green', 'yellow', 'red'],
        labels={'MAE': 'Mean Absolute Error'},
        title='Lower is Better'
    )

    fig_mae.update_layout(
        height=400,
        showlegend=False,
        hovermode='x unified'
    )

    fig_mae.add_hline(y=4.5, line_dash="dash", line_color="blue",
                      annotation_text="Industry Min (4.5)", annotation_position="right")

    st.plotly_chart(fig_mae, use_container_width=True)

    st.markdown("---")

    # R² comparison
    st.subheader("🎯 Model Confidence by Position (R² Score)")

    fig_r2 = px.bar(
        positions_df,
        x='Position',
        y='R²',
        color='R²',
        color_continuous_scale=['red', 'yellow', 'green'],
        labels={'R²': 'R² Score'},
        title='Higher is Better - Explains Variance in Actual Performance'
    )

    fig_r2.update_layout(
        height=400,
        showlegend=False,
        hovermode='x unified'
    )

    fig_r2.add_hline(y=0.5, line_dash="dash", line_color="orange",
                     annotation_text="Good Threshold (0.5)", annotation_position="right")

    st.plotly_chart(fig_r2, use_container_width=True)

    st.markdown("---")

    # Model insights
    st.subheader("💡 Key Model Insights")

    insights_col1, insights_col2 = st.columns(2)

    with insights_col1:
        st.markdown("""
        **✅ Strengths:**
        - **TE Excellence** (2.34 MAE): 62% improvement - Vegas game context perfect for role variance
        - **QB Breakthrough** (4.67 MAE): 29% improvement - Spread/totals capture game script
        - **RB Solid** (4.41 MAE): Position-specific tuning (depth=5) effective
        - **Overall Professional Grade** (4.26): Exceeds industry standard of 4.5-5.5
        """)

    with insights_col2:
        st.markdown("""
        **⚠️ Considerations:**
        - **WR Challenge** (5.06 MAE): Different validation weeks may explain variance difference
        - **WR R² Low** (0.183): Target distribution may not follow team totals as closely
        - **Recommendation**: Consider V2 (4.57 MAE) for WR predictions if prioritizing accuracy

        **Strategy:** Vegas odds capture game context better than advanced metrics
        """)

    st.markdown("---")

    # Technical details
    st.subheader("🔧 Model Architecture")

    arch_col1, arch_col2, arch_col3 = st.columns(3)

    with arch_col1:
        st.markdown("""
        **Features Used:**
        - Vegas spread
        - Team totals
        - Implied game volume
        - Home/away context
        - Variance indicators
        - Position-specific tuning
        """)

    with arch_col2:
        st.markdown("""
        **Hyperparameters:**
        - QB: depth=9 (complex game scripts)
        - RB: depth=5 (moderate variance)
        - WR: depth=7 (high variance)
        - TE: depth=6 (role variance)
        - Learning rate: adaptive
        """)

    with arch_col3:
        st.markdown("""
        **Validation Strategy:**
        - 647 players evaluated
        - 3-week held-out test (weeks 12-14)
        - MAE and R² computed per position
        - Cross-season stability tested
        """)


# Footer
st.markdown("---")
st.markdown("🏈 **NFL Analytics Platform** | Data from nflreadpy | Senior Design Project")