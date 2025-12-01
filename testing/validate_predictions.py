#!/usr/bin/env python3
"""
File: testing/validate_predictions.py

Interactive Streamlit app to validate ML model predictions against actual player stats.
Shows comprehensive comparison with visualizations and tooltips.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# Stat descriptions for tooltips
STAT_DESCRIPTIONS = {
    # Predictions
    'POB': 'Probability Over Baseline - Likelihood (0-100%) that player will exceed their rolling average',
    'EVOB': 'Expected Value Over Baseline - Predicted difference from rolling average (+ is better than average)',
    'Predicted Value': 'Model\'s predicted stat value for this week',
    'Confidence Interval': 'Range where we expect the actual value to fall (90% confidence)',
    'Baseline': 'Player\'s rolling average from previous 6 games (with decay)',

    # Fantasy Stats
    'fantasy_points': 'Standard fantasy points (no PPR)',
    'fantasy_points_ppr': 'Fantasy points with Point Per Reception scoring',

    # Passing Stats
    'passing_yards': 'Total passing yards',
    'passing_tds': 'Passing touchdowns',
    'passing_interceptions': 'Interceptions thrown',
    'completions': 'Passes completed',
    'attempts': 'Pass attempts',
    'completion_pct': 'Completion percentage',

    # Rushing Stats
    'rushing_yards': 'Total rushing yards',
    'rushing_tds': 'Rushing touchdowns',
    'carries': 'Rushing attempts',

    # Receiving Stats
    'receiving_yards': 'Total receiving yards',
    'receiving_tds': 'Receiving touchdowns',
    'receptions': 'Passes caught',
    'targets': 'Times targeted',

    # Kicking Stats
    'fg_made': 'Field goals made',
    'fg_att': 'Field goal attempts',
    'fg_pct': 'Field goal percentage',
    'pat_made': 'Extra points made',

    # Rolling Averages
    'rolling_avg_fantasy_pts': 'Average fantasy points over last 6 games',
    'rolling_avg_fantasy_ppr': 'Average PPR points over last 6 games',
    'rolling_avg_passing_yds': 'Average passing yards over last 6 games',
    'rolling_avg_passing_tds': 'Average passing TDs over last 6 games',
    'rolling_avg_rushing_yds': 'Average rushing yards over last 6 games',
    'rolling_avg_receiving_yds': 'Average receiving yards over last 6 games',
    'games_in_history': 'Number of games used for rolling averages',
}


def load_data(season: int, week: int):
    """Load predictions, actual stats, and features for a given week"""
    data_dir = Path('data/nfl')

    # Load predictions (for week N+1)
    pred_file = data_dir / 'predictions' / f'predictions_{season}_week_{week}.parquet'
    if not pred_file.exists():
        return None, None, None
    pred_df = pd.read_parquet(pred_file)

    # Load actual stats (for week N)
    stats_file = data_dir / 'raw' / f'player_stats_{season}_week_{week}.parquet'
    if not stats_file.exists():
        return pred_df, None, None
    stats_df = pd.read_parquet(stats_file)

    # Load features (from week N-1 since predictions are for week N)
    feat_file = data_dir / 'cleaned' / f'features_{season}_week_{week-1}.parquet'
    feat_df = None
    if feat_file.exists():
        feat_df = pd.read_parquet(feat_file)

    return pred_df, stats_df, feat_df


def get_available_weeks():
    """Get list of weeks that have both predictions and actual stats"""
    pred_dir = Path('data/nfl/predictions')
    stats_dir = Path('data/nfl/raw')

    pred_files = set(f.stem.replace('predictions_', '') for f in pred_dir.glob('predictions_*.parquet'))
    stats_files = set(f.stem.replace('player_stats_', '') for f in stats_dir.glob('player_stats_*.parquet'))

    # Find weeks that have both
    matching = pred_files & stats_files

    # Parse and sort
    weeks = []
    for match in matching:
        try:
            parts = match.split('_week_')
            season = int(parts[0])
            week = int(parts[1])
            weeks.append((season, week))
        except:
            continue

    return sorted(weeks)


def display_player_header(player_data, stats_row):
    """Display player header with photo and basic info"""
    col1, col2, col3 = st.columns([1, 2, 2])

    with col1:
        if 'headshot_url' in stats_row and pd.notna(stats_row['headshot_url']):
            st.image(stats_row['headshot_url'], width=150)
        else:
            st.info("No photo available")

    with col2:
        st.markdown(f"### {player_data['player_name']}")
        st.markdown(f"**Position:** {player_data['position']}")
        st.markdown(f"**Team:** {player_data['team']}")
        st.markdown(f"**Week:** {player_data['season']} Week {player_data['week']}")

    with col3:
        st.markdown(f"**Opponent:** {player_data['opponent']}")
        if 'team' in stats_row and 'opponent_team' in stats_row:
            st.markdown(f"**Matchup:** {stats_row['team']} vs {stats_row['opponent_team']}")


def create_comparison_chart(stat_name, actual, predicted, confidence_lower, confidence_upper, baseline):
    """Create a bar chart comparing actual vs predicted with confidence interval"""
    fig = go.Figure()

    # Add bars
    fig.add_trace(go.Bar(
        name='Actual',
        x=['Result'],
        y=[actual],
        marker_color='#00CC96',
        text=[f'{actual:.1f}'],
        textposition='auto',
    ))

    fig.add_trace(go.Bar(
        name='Predicted',
        x=['Result'],
        y=[predicted],
        marker_color='#636EFA',
        text=[f'{predicted:.1f}'],
        textposition='auto',
        error_y=dict(
            type='data',
            symmetric=False,
            array=[confidence_upper - predicted],
            arrayminus=[predicted - confidence_lower],
            color='rgba(99, 110, 250, 0.3)'
        )
    ))

    # Add baseline line
    fig.add_hline(
        y=baseline,
        line_dash="dash",
        line_color="orange",
        annotation_text=f"Baseline: {baseline:.1f}",
        annotation_position="right"
    )

    fig.update_layout(
        title=stat_name.replace('_', ' ').title(),
        yaxis_title="Value",
        barmode='group',
        height=300,
        showlegend=True
    )

    return fig


def display_prediction_table(predictions, actuals, features_row):
    """Display comprehensive prediction table with tooltips"""

    # Separate EVOB and POB predictions
    evob_preds = predictions[predictions['model_type'] == 'evob']
    pob_preds = predictions[predictions['model_type'] == 'pob']

    # Create comparison data
    comparison_data = []

    for _, pred in evob_preds.iterrows():
        stat = pred['stat']
        actual_val = actuals.get(stat, np.nan)

        # Calculate error
        error = actual_val - pred['predicted_value'] if not np.isnan(actual_val) else np.nan
        abs_error = abs(error) if not np.isnan(error) else np.nan
        pct_error = (error / actual_val * 100) if not np.isnan(actual_val) and actual_val != 0 else np.nan

        # Check if within confidence interval
        within_ci = (
            confidence_lower <= actual_val <= confidence_upper
            if not np.isnan(actual_val)
            else None
        ) if (confidence_lower := pred['confidence_lower']) and (confidence_upper := pred['confidence_upper']) else None

        comparison_data.append({
            'Stat': stat.replace('_', ' ').title(),
            'Actual': actual_val,
            'Predicted': pred['predicted_value'],
            'Error': error,
            'Abs Error': abs_error,
            'Error %': pct_error,
            'Baseline': pred['baseline'],
            'Confidence': f"[{pred['confidence_lower']:.1f}, {pred['confidence_upper']:.1f}]",
            'In CI?': '✓' if within_ci else '✗' if within_ci is not None else '-'
        })

    if comparison_data:
        df = pd.DataFrame(comparison_data)

        st.markdown("### 📊 Prediction vs Actual Comparison")

        # Format and display
        styled_df = df.style.format({
            'Actual': '{:.2f}',
            'Predicted': '{:.2f}',
            'Error': '{:+.2f}',
            'Abs Error': '{:.2f}',
            'Error %': '{:+.1f}%',
            'Baseline': '{:.2f}'
        })

        st.dataframe(styled_df, use_container_width=True, height=400)

        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            mae = df['Abs Error'].mean()
            st.metric("Mean Absolute Error", f"{mae:.2f}")
        with col2:
            rmse = np.sqrt((df['Error']**2).mean())
            st.metric("RMSE", f"{rmse:.2f}")
        with col3:
            in_ci_pct = (df['In CI?'] == '✓').sum() / len(df) * 100
            st.metric("In Confidence Interval", f"{in_ci_pct:.0f}%")
        with col4:
            mape = df['Error %'].abs().mean()
            st.metric("Mean Abs % Error", f"{mape:.1f}%")


def display_pob_predictions(pob_preds, actuals):
    """Display POB (probability) predictions"""
    if len(pob_preds) == 0:
        return

    st.markdown("### 🎲 Probability Over Baseline (POB)")

    pob_data = []
    for _, pred in pob_preds.iterrows():
        stat = pred['stat']
        actual_val = actuals.get(stat, np.nan)
        baseline = pred['baseline']

        # Did player beat baseline?
        beat_baseline = actual_val > baseline if not np.isnan(actual_val) else None

        pob_data.append({
            'Stat': stat.replace('_', ' ').title(),
            'Probability': f"{pred['probability_over']*100:.1f}%",
            'Baseline': f"{baseline:.1f}",
            'Actual': f"{actual_val:.1f}" if not np.isnan(actual_val) else 'N/A',
            'Beat Baseline?': '✓' if beat_baseline else '✗' if beat_baseline is not None else '-',
            'Model Correct?': (
                '✓' if (beat_baseline and pred['probability_over'] > 0.5) or (not beat_baseline and pred['probability_over'] <= 0.5)
                else '✗' if beat_baseline is not None else '-'
            )
        })

    df = pd.DataFrame(pob_data)
    st.dataframe(df, use_container_width=True)

    # POB accuracy
    if len(df) > 0:
        correct = (df['Model Correct?'] == '✓').sum()
        total = (df['Model Correct?'] != '-').sum()
        if total > 0:
            accuracy = correct / total * 100
            st.metric("POB Accuracy", f"{accuracy:.1f}%",
                     delta=f"{accuracy-50:.1f}% vs random" if accuracy > 0 else None)


def display_rolling_averages(features_row):
    """Display player's rolling averages and history"""
    if features_row is None:
        st.info("No feature data available for this player")
        return

    st.markdown("### 📈 Player History & Rolling Averages")

    # Filter relevant columns
    rolling_cols = [col for col in features_row.index if 'rolling_avg' in col or 'games_in_history' in col]

    data = []
    for col in rolling_cols:
        if pd.notna(features_row[col]) and features_row[col] != 0:
            data.append({
                'Metric': col.replace('rolling_avg_', '').replace('_', ' ').title(),
                'Value': f"{features_row[col]:.2f}"
            })

    if data:
        df = pd.DataFrame(data)
        st.dataframe(df, use_container_width=True, height=300)


def create_stat_comparison_charts(predictions, actuals, stat_names):
    """Create multiple comparison charts"""
    charts = []

    evob_preds = predictions[predictions['model_type'] == 'evob']

    for stat in stat_names:
        pred_row = evob_preds[evob_preds['stat'] == stat]
        if len(pred_row) == 0:
            continue

        pred_row = pred_row.iloc[0]
        actual = actuals.get(stat, np.nan)

        if np.isnan(actual):
            continue

        fig = create_comparison_chart(
            stat,
            actual,
            pred_row['predicted_value'],
            pred_row['confidence_lower'],
            pred_row['confidence_upper'],
            pred_row['baseline']
        )
        charts.append((stat, fig))

    return charts


def main():
    st.set_page_config(
        page_title="NFL Prediction Validator",
        page_icon="🏈",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("🏈 NFL ML Model Prediction Validator")
    st.markdown("Compare predicted player stats against actual performance")

    # Sidebar for week selection
    st.sidebar.header("Select Week")

    available_weeks = get_available_weeks()
    if not available_weeks:
        st.error("No prediction data found. Run generate_all_predictions.py first.")
        return

    # Format for display
    week_options = [f"{season} Week {week}" for season, week in available_weeks]
    selected_week = st.sidebar.selectbox("Week", week_options, index=len(week_options)-1)

    # Parse selection
    season_str, week_str = selected_week.split(" Week ")
    season = int(season_str)
    week = int(week_str)

    # Load data
    with st.spinner("Loading data..."):
        pred_df, stats_df, feat_df = load_data(season, week)

    if pred_df is None:
        st.error(f"No prediction data for {season} Week {week}")
        return

    if stats_df is None:
        st.warning(f"No actual stats available for {season} Week {week}. Predictions only.")

    # Player selection
    st.sidebar.header("Select Player")

    # Get unique players
    players = pred_df[['player_id', 'player_name', 'position', 'team']].drop_duplicates()
    players = players.sort_values(['position', 'player_name'])

    # Position filter
    positions = ['All'] + sorted(players['position'].unique().tolist())
    selected_position = st.sidebar.selectbox("Position", positions)

    if selected_position != 'All':
        players = players[players['position'] == selected_position]

    # Player selection
    player_names = [f"{row['player_name']} ({row['position']}, {row['team']})"
                   for _, row in players.iterrows()]

    selected_player_display = st.sidebar.selectbox("Player", player_names)

    # Parse player selection
    player_name = selected_player_display.split(' (')[0]
    selected_player_id = players[players['player_name'] == player_name]['player_id'].iloc[0]

    # Get player data
    player_predictions = pred_df[pred_df['player_id'] == selected_player_id]
    player_data = player_predictions.iloc[0].to_dict()

    # Get actual stats
    player_actuals = {}
    stats_row = None
    if stats_df is not None:
        stats_rows = stats_df[stats_df['player_id'] == selected_player_id]
        if len(stats_rows) > 0:
            stats_row = stats_rows.iloc[0]
            player_actuals = stats_row.to_dict()

    # Get features
    features_row = None
    if feat_df is not None:
        feat_rows = feat_df[feat_df['player_id'] == selected_player_id]
        if len(feat_rows) > 0:
            features_row = feat_rows.iloc[0]

    # Display player header
    if stats_row is not None:
        display_player_header(player_data, stats_row)
    else:
        st.markdown(f"### {player_data['player_name']}")
        st.markdown(f"**{player_data['position']} | {player_data['team']} vs {player_data['opponent']}**")

    st.markdown("---")

    # Create tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Comparison Table", "📈 Visual Comparison", "🎲 POB Predictions", "📋 Player History"])

    with tab1:
        if stats_row is not None:
            display_prediction_table(player_predictions, player_actuals, features_row)
        else:
            st.info("No actual stats available for comparison yet.")
            st.markdown("### Predictions")
            evob_preds = player_predictions[player_predictions['model_type'] == 'evob']
            for _, pred in evob_preds.iterrows():
                st.markdown(f"**{pred['stat'].replace('_', ' ').title()}:** {pred['predicted_value']:.2f} "
                          f"(baseline: {pred['baseline']:.2f}, diff: {pred['predicted_diff']:+.2f})")

    with tab2:
        if stats_row is not None:
            st.markdown("### Actual vs Predicted Comparison")

            # Get all EVOB stats
            evob_stats = player_predictions[player_predictions['model_type'] == 'evob']['stat'].tolist()

            # Create charts
            charts = create_stat_comparison_charts(player_predictions, player_actuals, evob_stats)

            if charts:
                # Display in grid
                cols = st.columns(2)
                for i, (stat, fig) in enumerate(charts):
                    with cols[i % 2]:
                        st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No matching actual stats available for visualization")
        else:
            st.info("No actual stats available for visualization yet.")

    with tab3:
        pob_preds = player_predictions[player_predictions['model_type'] == 'pob']
        if len(pob_preds) > 0:
            display_pob_predictions(pob_preds, player_actuals)
        else:
            st.info("No POB predictions available for this player")

    with tab4:
        display_rolling_averages(features_row)

    # Info tooltip in sidebar
    with st.sidebar:
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.markdown("""
        **Metrics Explained:**

        **EVOB (Expected Value Over Baseline)**
        - Predicts actual stat value
        - Shows confidence interval
        - Compares to rolling average

        **POB (Probability Over Baseline)**
        - % chance to beat baseline
        - Binary classification (yes/no)

        **Baseline**
        - Rolling average from last 6 games
        - Uses 0.9 decay factor

        **Error Metrics:**
        - MAE: Mean Absolute Error
        - RMSE: Root Mean Squared Error
        - MAPE: Mean Absolute % Error
        """)


if __name__ == "__main__":
    main()
