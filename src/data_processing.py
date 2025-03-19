import pandas as pd
import numpy as np
from datetime import datetime

def load_data(data_dir='data'):
    """Load NBA datasets from the data directory"""
    try:
        teams = pd.read_csv(f'{data_dir}/NBA_TEAMS.csv')
        players = pd.read_csv(f'{data_dir}/NBA_PLAYERS.csv')
        games = pd.read_csv(f'{data_dir}/NBA_GAMES.csv')
        player_games = pd.read_csv(f'{data_dir}/NBA_PLAYER_GAMES.csv')
        
        print(f"Teams: {teams.shape}")
        print(f"Players: {players.shape}")
        print(f"Games: {games.shape}")
        print(f"Player Games: {player_games.shape}")
        
        return teams, players, games, player_games
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None, None, None

def preprocess_data(teams, players, games, player_games):
    """Preprocess NBA datasets for analysis"""
    # Check for missing values
    print("\nChecking for missing values:")
    print(f"Teams: {teams.isnull().sum().sum()}")
    print(f"Players: {players.isnull().sum().sum()}")
    print(f"Games: {games.isnull().sum().sum()}")
    print(f"Player Games: {player_games.isnull().sum().sum()}")
    
    # Convert date strings to datetime objects
    games['GAME_DATE'] = pd.to_datetime(games['GAME_DATE'], format='%b %d, %Y')
    player_games['GAME_DATE'] = pd.to_datetime(player_games['GAME_DATE'], format='%b %d, %Y')
    
    # Create team lookup dictionary
    team_dict = teams.set_index('id').to_dict(orient='index')
    player_dict = players.set_index('id').to_dict(orient='index')
    
    # Add team and player names to game data
    games['TeamName'] = games['Team_ID'].map(lambda x: team_dict.get(x, {}).get('full_name'))
    player_games['PlayerName'] = player_games['Player_ID'].map(lambda x: player_dict.get(x, {}).get('full_name'))
    
    # Engineer features for games dataset
    games['PointsPerPossession'] = games['PTS'] / (games['FGA'] - games['OREB'] + games['TOV'] + 0.44 * games['FTA'])
    games['AssistRatio'] = games['AST'] / (games['FGA'] + 0.44 * games['FTA'] + games['AST'] + games['TOV'])
    games['TurnoverRatio'] = games['TOV'] / (games['FGA'] + 0.44 * games['FTA'] + games['AST'] + games['TOV'])
    games['EffectiveFG'] = (games['FGM'] + 0.5 * games['FG3M']) / games['FGA']
    games['DefensiveRebound%'] = games['DREB'] / (games['DREB'] + games['OREB'])
    games['OffensiveRebound%'] = games['OREB'] / (games['DREB'] + games['OREB'])
    
    # Create features for player_games dataset
    player_games['UsageRate'] = (player_games['FGA'] + 0.44 * player_games['FTA'] + player_games['TOV']) / player_games['MIN']
    player_games['EffectiveFG'] = (player_games['FGM'] + 0.5 * player_games['FG3M']) / player_games['FGA']
    player_games['TrueShootingPct'] = player_games['PTS'] / (2 * (player_games['FGA'] + 0.44 * player_games['FTA']))
    player_games['PointsPerMinute'] = player_games['PTS'] / player_games['MIN']
    player_games['ReboundsPerMinute'] = player_games['REB'] / player_games['MIN']
    player_games['AssistsPerMinute'] = player_games['AST'] / player_games['MIN']
    
    # Handle infinite values from division by zero
    for df in [games, player_games]:
        for col in df.select_dtypes(include=['float64']).columns:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
    
    # Fill NaN values with appropriate replacements
    games = games.fillna(0)
    player_games = player_games.fillna(0)
    
    print("\nDataset shapes after preprocessing:")
    print(f"Games: {games.shape}")
    print(f"Player Games: {player_games.shape}")
    
    return games, player_games

def create_temporal_features(player_games_df):
    """Add temporal features to player game data"""
    # Group by player
    player_temporal_data = []
    
    for player_id, group in player_games_df.groupby('Player_ID'):
        if len(group) >= 5:  # Only consider players with at least 5 games
            # Sort by date
            player_df = group.sort_values('GAME_DATE')
            
            # Initialize new columns
            player_df['PTS_MA5'] = player_df['PTS'].rolling(window=5, min_periods=1).mean()
            player_df['PTS_Trend'] = player_df['PTS'] - player_df['PTS_MA5']
            player_df['PTS_Volatility'] = player_df['PTS'].rolling(window=5, min_periods=1).std()
            
            # Calculate game-to-game changes
            player_df['PTS_Change'] = player_df['PTS'].diff()
            player_df['PLUS_MINUS_Change'] = player_df['PLUS_MINUS'].diff()
            
            # Create momentum features
            player_df['PTS_Momentum'] = player_df['PTS_Change'].rolling(window=3, min_periods=1).mean()
            player_df['Performance_Momentum'] = player_df['PLUS_MINUS'].rolling(window=3, min_periods=1).mean()
            
            player_temporal_data.append(player_df)
    
    # Combine all players
    player_temporal_df = pd.concat(player_temporal_data)
    
    return player_temporal_df

def calculate_player_dynamics(player_temporal_df):
    """Calculate player dynamics metrics"""
    # Group by player
    player_dynamics = []
    
    for player_id, group in player_temporal_df.groupby('Player_ID'):
        if len(group) >= 10:  # Only consider players with enough games for meaningful analysis
            player_name = group['PlayerName'].iloc[0]
            
            # Calculate basic stats
            avg_pts = group['PTS'].mean()
            avg_plus_minus = group['PLUS_MINUS'].mean()
            games_played = len(group)
            
            # Calculate stability metrics
            pts_volatility = group['PTS_Volatility'].mean()
            plus_minus_volatility = group['PLUS_MINUS'].std()
            
            # Calculate system stability (lower = more stable)
            system_stability = (pts_volatility / max(1, avg_pts) + 
                               plus_minus_volatility / max(1, abs(avg_plus_minus))) / 2
            
            # Calculate performance entropy (measure of predictability)
            # Normalize PTS to 0-1 range for entropy calculation
            pts_normalized = (group['PTS'] - group['PTS'].min()) / (group['PTS'].max() - group['PTS'].min() + 1e-10)
            pts_bins = np.histogram(pts_normalized, bins=10)[0]
            pts_probs = pts_bins / len(group)
            performance_entropy = -np.sum(pts_probs * np.log2(pts_probs + 1e-10))
            
            player_dynamics.append({
                'player_id': player_id,
                'player_name': player_name,
                'games_played': games_played,
                'avg_pts': avg_pts,
                'avg_plus_minus': avg_plus_minus,
                'pts_volatility': pts_volatility,
                'plus_minus_volatility': plus_minus_volatility,
                'system_stability': system_stability,
                'performance_entropy': performance_entropy
            })
    
    # Convert to dataframe
    player_dynamics_df = pd.DataFrame(player_dynamics)
    
    return player_dynamics_df

def extract_team_styles(games_df, n_clusters=4):
    """Extract team playing styles using clustering"""
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    
    # Aggregate game data by team
    team_stats = games_df.groupby('Team_ID').agg({
        'TeamName': 'first',
        'PTS': 'mean',
        'FGA': 'mean',
        'FG3A': 'mean',
        'FTA': 'mean',
        'AST': 'mean',
        'OREB': 'mean',
        'DREB': 'mean',
        'STL': 'mean',
        'BLK': 'mean',
        'TOV': 'mean',
        'PLUS_MINUS': 'mean',
        'WL': lambda x: (x == 'W').mean(),  # Win percentage
        'PointsPerPossession': 'mean',
        'AssistRatio': 'mean',
        'TurnoverRatio': 'mean',
        'EffectiveFG': 'mean',
        'DefensiveRebound%': 'mean',
        'OffensiveRebound%': 'mean'
    }).reset_index()
    
    # Rename win percentage column
    team_stats = team_stats.rename(columns={'WL': 'win_pct'})
    
    # Create style metrics
    team_stats['pace'] = team_stats['FGA'] + team_stats['TOV'] - team_stats['OREB']
    team_stats['three_point_rate'] = team_stats['FG3A'] / team_stats['FGA']
    team_stats['assist_rate'] = team_stats['AST'] / team_stats['FGA']
    team_stats['defensive_focus'] = (team_stats['STL'] + team_stats['BLK']) / (team_stats['STL'] + team_stats['BLK'] + team_stats['DREB'] + team_stats['OREB'])
    team_stats['offensive_efficiency'] = team_stats['PointsPerPossession']
    
    # Calculate style entropy (tactical diversity)
    # Higher values indicate more diverse playing styles
    style_metrics = ['pace', 'three_point_rate', 'assist_rate', 'defensive_focus', 'offensive_efficiency']
    
    # Normalize metrics for entropy calculation
    scaler = StandardScaler()
    style_data = scaler.fit_transform(team_stats[style_metrics])
    
    # Calculate entropy for each team
    style_entropy = []
    for i in range(len(team_stats)):
        team_style = style_data[i]
        # Convert to probability distribution (normalize to sum to 1)
        style_probs = np.abs(team_style) / (np.sum(np.abs(team_style)) + 1e-10)
        # Calculate entropy
        entropy = -np.sum(style_probs * np.log2(style_probs + 1e-10))
        style_entropy.append(entropy)
    
    team_stats['style_entropy'] = style_entropy
    
    # Cluster teams by style
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    team_stats['style_cluster'] = kmeans.fit_predict(style_data)
    
    # Name the clusters
    style_names = {
        0: 'Modern Pace-and-Space',
        1: 'Traditional Inside-Out',
        2: 'Defensive-Oriented',
        3: 'Balanced Attack'
    }
    
    team_stats['style_name'] = team_stats['style_cluster'].map(style_names)
    
    return team_stats

def calculate_player_team_fit(player_games_df, team_stats_df):
    """Calculate player fit with different team styles"""
    # Get team style clusters
    team_styles = team_stats_df[['Team_ID', 'style_cluster', 'style_name']].copy()
    
    # Merge player games with team styles
    player_style_games = pd.merge(player_games_df, team_styles, on='Team_ID', how='inner')
    
    # Calculate player performance by team style
    player_style_performance = []
    
    for player_id, player_games in player_style_games.groupby('Player_ID'):
        if len(player_games) >= 10:  # Only consider players with enough games
            player_name = player_games['PlayerName'].iloc[0]
            games_played = len(player_games)
            
            # Calculate overall performance
            overall_plus_minus = player_games['PLUS_MINUS'].mean()
            
            # Calculate performance by style
            style_performance = {}
            best_style = None
            best_style_plus_minus = -float('inf')
            
            for style, style_games in player_games.groupby('style_cluster'):
                if len(style_games) >= 3:  # Minimum games in a style
                    style_name = style_games['style_name'].iloc[0]
                    style_plus_minus = style_games['PLUS_MINUS'].mean()
                    style_performance[style] = {
                        'style_name': style_name,
                        'games': len(style_games),
                        'plus_minus': style_plus_minus
                    }
                    
                    if style_plus_minus > best_style_plus_minus:
                        best_style_plus_minus = style_plus_minus
                        best_style = style_name
            
            # Calculate style gap (difference between best and worst style)
            if len(style_performance) >= 2:
                style_values = [data['plus_minus'] for data in style_performance.values()]
                style_gap = max(style_values) - min(style_values)
                
                # Calculate adaptability score (inverse of style gap, normalized)
                # Higher score = more adaptable across styles
                adaptability_score = 1 / (1 + style_gap)
            else:
                style_gap = 0
                adaptability_score = 0.5  # Neutral if not enough data
            
            player_style_performance.append({
                'player_id': player_id,
                'player_name': player_name,
                'games_played': games_played,
                'overall_plus_minus': overall_plus_minus,
                'best_style': best_style,
                'best_style_plus_minus': best_style_plus_minus,
                'style_gap': style_gap,
                'adaptability_score': adaptability_score,
                'style_performance': style_performance
            })
    
    # Convert to dataframe
    player_team_fit_df = pd.DataFrame(player_style_performance)
    
    return player_team_fit_df

def calculate_player_impact_model(player_dynamics_df, player_team_fit_df, centrality_df=None):
    """Calculate comprehensive player impact model"""
    # Merge player dynamics and team fit data
    player_impact = pd.merge(
        player_dynamics_df,
        player_team_fit_df[['player_id', 'adaptability_score', 'best_style']],
        on='player_id',
        how='inner'
    )
    
    # Add network centrality if available
    if centrality_df is not None:
        player_impact = pd.merge(
            player_impact,
            centrality_df[['player_id', 'influence_index']],
            on='player_id',
            how='left'
        )
        player_impact['influence_index'] = player_impact['influence_index'].fillna(0)
    else:
        player_impact['influence_index'] = 0
    
    # Calculate component scores
    # 1. Production component (traditional stats)
    player_impact['production_component'] = (
        player_impact['avg_pts'] / player_impact['avg_pts'].max() * 0.7 +
        (player_impact['avg_plus_minus'] - player_impact['avg_plus_minus'].min()) / 
        (player_impact['avg_plus_minus'].max() - player_impact['avg_plus_minus'].min() + 1e-10) * 0.3
    ) * 10
    
    # 2. Stability component
    player_impact['stability_component'] = (
        (1 - player_impact['system_stability'] / player_impact['system_stability'].max()) * 0.6 +
        (1 - player_impact['performance_entropy'] / player_impact['performance_entropy'].max()) * 0.4
    ) * 10
    
    # 3. Adaptability component
    player_impact['adaptability_component'] = player_impact['adaptability_score'] * 10
    
    # 4. Network component (teammate influence)
    player_impact['network_component'] = player_impact['influence_index'] * 10
    
    # Calculate overall impact score
    player_impact['impact_score'] = (
        player_impact['production_component'] * 0.4 +
        player_impact['stability_component'] * 0.25 +
        player_impact['adaptability_component'] * 0.2 +
        player_impact['network_component'] * 0.15
    )
    
    return player_impact

def save_processed_data(data_dict, output_dir='data/processed'):
    """Save processed dataframes to CSV files"""
    import os
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save each dataframe
    for name, df in data_dict.items():
        output_path = f"{output_dir}/{name}.csv"
        df.to_csv(output_path, index=False)
        print(f"Saved {name} to {output_path}")