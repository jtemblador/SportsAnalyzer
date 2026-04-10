# Vegas Odds API - Usage Guide

## Overview

The Vegas odds system is optimized for your workflow:
1. Run your ML predictions
2. Identify top 20-30 players with high confidence
3. Fetch Vegas props **ONLY** for those players (saves API credits)
4. Compare your predictions vs Vegas

## File Structure

```
data/nfl/vegas_odds/
├── team_lines/
│   ├── team_lines_week_14_fetched_2025-12-06.parquet
│   └── team_lines_week_15_fetched_2025-12-06.parquet
├── player_props/
│   └── player_props_week_14_fetched_2025-12-06.parquet
└── fetch_log.json
```

## Smart Caching

- **Cache-first approach**: Always checks cache before API
- **Incremental updates**: Only fetches missing players
- **Stops early**: Stops searching once all players found
- **Example**: Request 5 players → 3 in cache → only fetch 2 → saves API calls!

## Usage Example

```python
from src.nfl.v4_odds_fetcher import VegasLinesFetcher

# Initialize
API_KEY = 'your_key_here'
fetcher = VegasLinesFetcher(odds_api_key=API_KEY)

# Your workflow:
# 1. Get your top predictions
top_players = [
    'Patrick Mahomes',
    'Josh Allen',
    'Travis Kelce',
    # ... your top 20-30 players
]

# 2. Fetch their Vegas props (uses cache intelligently)
week = 14
props_df = fetcher.fetch_player_props(
    player_names=top_players,
    week=week,
    force_refresh=False  # Use cache
)

# 3. Compare your predictions vs Vegas
for player in top_players:
    player_props = props_df[props_df['player_name'] == player]
    # Compare player_props['prop_value'] with your predictions
```

## What You Get

### Team Lines (Game Context)
```
Chiefs vs Texans (Week 14):
- Spread: KC -3.5
- Over/Under: 51.5
- Chiefs implied: 27.5 pts
- Texans implied: 24.0 pts
```

### Player Props (Vegas Predictions)
```
Patrick Mahomes (Week 14):
- passing_yards: 240.5
- passing_tds: 1.5
- completions: 22.5
- rush_yards: 22.5
- rush_attempts: 4.5
```

## API Usage Tracking

- **Total requests**: 500/month (free tier)
- **Current remaining**: 303
- **Usage per fetch**:
  - Team lines: 1-2 requests
  - Player props (full week): ~13 requests
  - Player props (specific players, cached): 0 requests!

## Key Methods

### `fetch_team_lines(force_refresh=False)`
Fetches team-level game context (spreads, totals)

### `fetch_player_props(player_names, week, force_refresh=False)`
**Your main method** - Fetches props for specific players only

### `load_cached_team_lines(week=None)`
Loads cached team lines without API calls

### `load_cached_player_props(week=None)`
Loads cached player props without API calls

## Testing

```bash
# Test fetching specific players
python3 src/nfl/v4_odds_fetcher.py
# Choose option 4

# Or directly
python3 -c "
from src.nfl.v4_odds_fetcher import VegasLinesFetcher
fetcher = VegasLinesFetcher(odds_api_key='your_key')
props = fetcher.fetch_player_props(['Patrick Mahomes'], week=14)
print(props)
"
```

## Optimization Features

✅ **Cache-first**: Checks cache before API  
✅ **Incremental**: Only fetches missing players  
✅ **Early stopping**: Stops when all players found  
✅ **Weekly caching**: Organizes by week for easy access  
✅ **Date tracking**: Knows when data was fetched  
✅ **No duplicate calls**: Won't re-fetch cached data  

## Next Steps

Integrate into v4_feature_engineer.py for validation/comparison workflow.
