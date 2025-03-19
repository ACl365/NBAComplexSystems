import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR
from scipy import stats

def calculate_var_model(player_data, features=['PTS', 'PLUS_MINUS'], lag=1):
    """Calculate Vector Autoregression model for player performance"""
    # Ensure data is sorted by date
    player_data = player_data.sort_values('GAME_DATE')
    
    # Extract features for VAR model
    var_data = player_data[features].copy()
    
    # Check if we have enough data points
    if len(var_data) <= lag + 1:
        return None, None, None
    
    try:
        # Fit VAR model
        model = VAR(var_data)
        results = model.fit(lag)
        
        # Extract coefficient matrix
        coef_matrix = results.coefs[0]
        
        # Calculate eigenvalues to assess stability
        eigenvalues = np.linalg.eigvals(coef_matrix)
        
        # Calculate Lyapunov exponents
        # Lyapunov exponents measure the rate of separation of infinitesimally close trajectories
        # Positive exponents indicate chaos, negative indicate stability
        lyapunov_exponents = np.log(np.abs(eigenvalues))
        
        return results, eigenvalues, lyapunov_exponents
    except Exception as e:
        print(f"Error calculating VAR model: {e}")
        return None, None, None

def calculate_stability_metrics(player_temporal_df):
    """Calculate stability metrics for all players"""
    stability_metrics = []
    
    for player_id, player_data in player_temporal_df.groupby('Player_ID'):
        if len(player_data) >= 10:  # Need enough games for meaningful analysis
            player_name = player_data['PlayerName'].iloc[0]
            
            # Calculate VAR model
            var_results, eigenvalues, lyapunov_exponents = calculate_var_model(player_data)
            
            if var_results is not None and eigenvalues is not None:
                # Calculate stability metrics
                max_eigenvalue = np.max(np.abs(eigenvalues))
                max_lyapunov = np.max(lyapunov_exponents)
                
                # Calculate performance entropy
                pts_normalized = (player_data['PTS'] - player_data['PTS'].min()) / (player_data['PTS'].max() - player_data['PTS'].min() + 1e-10)
                pts_bins = np.histogram(pts_normalized, bins=10)[0]
                pts_probs = pts_bins / len(player_data)
                performance_entropy = -np.sum(pts_probs * np.log2(pts_probs + 1e-10))
                
                # Calculate volatility
                pts_volatility = player_data['PTS'].std() / player_data['PTS'].mean() if player_data['PTS'].mean() > 0 else 0
                pm_volatility = player_data['PLUS_MINUS'].std()
                
                # Calculate system stability score (lower = more stable)
                system_stability = 0.3 * max_eigenvalue + 0.3 * (max_lyapunov if max_lyapunov > 0 else 0) + 0.2 * pts_volatility + 0.2 * (performance_entropy / 3)
                
                # Classify stability type
                if system_stability < 0.7:
                    stability_type = "Highly Stable"
                elif system_stability < 0.9:
                    stability_type = "Stable"
                elif system_stability < 1.1:
                    stability_type = "Moderately Stable"
                elif system_stability < 1.3:
                    stability_type = "Volatile"
                else:
                    stability_type = "Chaotic"
                
                stability_metrics.append({
                    'player_id': player_id,
                    'player_name': player_name,
                    'max_eigenvalue': max_eigenvalue,
                    'max_lyapunov': max_lyapunov,
                    'performance_entropy': performance_entropy,
                    'pts_volatility': pts_volatility,
                    'pm_volatility': pm_volatility,
                    'system_stability': system_stability,
                    'stability_type': stability_type
                })
    
    # Convert to dataframe
    stability_df = pd.DataFrame(stability_metrics)
    
    return stability_df

def plot_stability_distribution(stability_df):
    """Plot the distribution of player stability metrics"""
    plt.figure(figsize=(12, 8))
    
    # Create histogram of system stability
    plt.hist(stability_df['system_stability'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    
    # Add vertical lines for stability categories
    plt.axvline(x=0.7, color='green', linestyle='--', label='Highly Stable')
    plt.axvline(x=0.9, color='blue', linestyle='--', label='Stable')
    plt.axvline(x=1.1, color='orange', linestyle='--', label='Moderately Stable')
    plt.axvline(x=1.3, color='red', linestyle='--', label='Volatile')
    
    # Add labels and title
    plt.xlabel('System Stability Score (lower = more stable)', fontsize=12)
    plt.ylabel('Number of Players', fontsize=12)
    plt.title('Distribution of Player Performance Stability', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add annotation
    plt.annotate('Based on dynamical systems analysis of game-to-game performance patterns', 
                xy=(0.5, -0.15), xycoords='axes fraction', ha='center', fontsize=10)
    
    plt.tight_layout()
    
    # Save the figure
    plt.savefig('results/stability_distribution.png')
    print("Stability distribution visualization saved to results/stability_distribution.png")
    
    return plt

def calculate_performance_prediction(player_temporal_df, test_size=0.2):
    """Build a performance prediction model using XGBoost"""
    import xgboost as xgb
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    # Prepare features for prediction
    prediction_data = []
    
    for player_id, player_data in player_temporal_df.groupby('Player_ID'):
        if len(player_data) >= 15:  # Need enough games for train/test split
            # Sort by date
            player_data = player_data.sort_values('GAME_DATE')
            
            # Create lagged features
            for lag in range(1, 4):  # Use up to 3 previous games
                player_data[f'PTS_lag{lag}'] = player_data['PTS'].shift(lag)
                player_data[f'PLUS_MINUS_lag{lag}'] = player_data['PLUS_MINUS'].shift(lag)
            
            # Drop rows with NaN from lagging
            player_data = player_data.dropna()
            
            # Add to prediction data
            prediction_data.append(player_data)
    
    # Combine all players
    if not prediction_data:
        print("Not enough data for prediction model")
        return None, None
        
    all_data = pd.concat(prediction_data)
    
    # Define features and target
    features = [
        'PTS_lag1', 'PTS_lag2', 'PTS_lag3',
        'PLUS_MINUS_lag1', 'PLUS_MINUS_lag2', 'PLUS_MINUS_lag3',
        'PTS_MA5', 'PTS_Volatility', 'PTS_Momentum', 'Performance_Momentum'
    ]
    
    # Add player ID as categorical feature
    all_data['player_id_cat'] = all_data['Player_ID'].astype('category').cat.codes
    features.append('player_id_cat')
    
    X = all_data[features]
    y_pts = all_data['PTS']
    y_pm = all_data['PLUS_MINUS']
    
    # Split into train and test sets
    X_train, X_test, y_pts_train, y_pts_test = train_test_split(
        X, y_pts, test_size=test_size, random_state=42
    )
    
    _, _, y_pm_train, y_pm_test = train_test_split(
        X, y_pm, test_size=test_size, random_state=42
    )
    
    # Train XGBoost model for points
    pts_model = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
    pts_model.fit(X_train, y_pts_train)
    
    # Train XGBoost model for plus/minus
    pm_model = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
    pm_model.fit(X_train, y_pm_train)
    
    # Make predictions
    pts_pred = pts_model.predict(X_test)
    pm_pred = pm_model.predict(X_test)
    
    # Calculate metrics
    pts_rmse = np.sqrt(mean_squared_error(y_pts_test, pts_pred))
    pts_mae = mean_absolute_error(y_pts_test, pts_pred)
    pts_r2 = r2_score(y_pts_test, pts_pred)
    
    pm_rmse = np.sqrt(mean_squared_error(y_pm_test, pm_pred))
    pm_mae = mean_absolute_error(y_pm_test, pm_pred)
    pm_r2 = r2_score(y_pm_test, pm_pred)
    
    print("\nPerformance Prediction Model Results:")
    print(f"Points Prediction - RMSE: {pts_rmse:.2f}, MAE: {pts_mae:.2f}, R²: {pts_r2:.3f}")
    print(f"Plus/Minus Prediction - RMSE: {pm_rmse:.2f}, MAE: {pm_mae:.2f}, R²: {pm_r2:.3f}")
    
    # Feature importance
    pts_importance = pts_model.feature_importances_
    pm_importance = pm_model.feature_importances_
    
    importance_df = pd.DataFrame({
        'Feature': features,
        'Points_Importance': pts_importance,
        'PlusMinus_Importance': pm_importance
    })
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    
    importance_df = importance_df.sort_values('Points_Importance', ascending=False)
    plt.barh(importance_df['Feature'], importance_df['Points_Importance'], color='skyblue')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Feature Importance for Points Prediction')
    plt.tight_layout()
    
    # Save the figure
    plt.savefig('results/feature_importance.png')
    print("Feature importance visualization saved to results/feature_importance.png")
    
    # Create model results dictionary
    model_results = {
        'pts_model': pts_model,
        'pm_model': pm_model,
        'pts_metrics': {
            'rmse': pts_rmse,
            'mae': pts_mae,
            'r2': pts_r2
        },
        'pm_metrics': {
            'rmse': pm_rmse,
            'mae': pm_mae,
            'r2': pm_r2
        },
        'importance_df': importance_df
    }
    
    return model_results, all_data[['Player_ID', 'PlayerName', 'GAME_DATE'] + features + ['PTS', 'PLUS_MINUS']]

def decompose_performance_factors(player_temporal_df, player_dynamics_df, player_team_fit_df):
    """Decompose player performance into intrinsic skill, context, and momentum"""
    # Merge player data
    player_data = pd.merge(
        player_temporal_df,
        player_dynamics_df[['player_id', 'system_stability', 'performance_entropy']],
        left_on='Player_ID',
        right_on='player_id',
        how='inner'
    )
    
    player_data = pd.merge(
        player_data,
        player_team_fit_df[['player_id', 'adaptability_score', 'best_style']],
        on='player_id',
        how='inner'
    )
    
    # Calculate performance components
    performance_components = []
    
    for player_id, player_games in player_data.groupby('Player_ID'):
        if len(player_games) >= 10:
            player_name = player_games['PlayerName'].iloc[0]
            
            # Calculate baseline (intrinsic skill)
            baseline_pts = player_games['PTS'].mean()
            
            for _, game in player_games.iterrows():
                # Calculate momentum component
                momentum_component = game['PTS_Momentum'] if not np.isnan(game['PTS_Momentum']) else 0
                
                # Calculate contextual component (deviation from baseline not explained by momentum)
                contextual_component = game['PTS'] - baseline_pts - momentum_component
                
                performance_components.append({
                    'player_id': player_id,
                    'player_name': player_name,
                    'game_date': game['GAME_DATE'],
                    'total_points': game['PTS'],
                    'baseline_component': baseline_pts,
                    'momentum_component': momentum_component,
                    'contextual_component': contextual_component,
                    'system_stability': game['system_stability'],
                    'adaptability_score': game['adaptability_score']
                })
    
    # Convert to dataframe
    components_df = pd.DataFrame(performance_components)
    
    # Calculate component percentages
    components_df['baseline_pct'] = np.abs(components_df['baseline_component']) / components_df['total_points'] * 100
    components_df['momentum_pct'] = np.abs(components_df['momentum_component']) / components_df['total_points'] * 100
    components_df['contextual_pct'] = np.abs(components_df['contextual_component']) / components_df['total_points'] * 100
    
    # Handle division by zero
    components_df = components_df.replace([np.inf, -np.inf], np.nan)
    components_df = components_df.fillna(0)
    
    # Calculate average component breakdown by player
    player_components = components_df.groupby('player_id').agg({
        'player_name': 'first',
        'total_points': 'mean',
        'baseline_component': 'mean',
        'momentum_component': 'mean',
        'contextual_component': 'mean',
        'baseline_pct': 'mean',
        'momentum_pct': 'mean',
        'contextual_pct': 'mean',
        'system_stability': 'first',
        'adaptability_score': 'first'
    }).reset_index()
    
    return components_df, player_components

def plot_performance_decomposition(player_components, n_players=10):
    """Plot the decomposition of player performance factors"""
    # Select top players by total points
    top_players = player_components.nlargest(n_players, 'total_points')
    
    plt.figure(figsize=(14, 10))
    
    # Create stacked bar chart
    bar_width = 0.6
    y_pos = np.arange(len(top_players))
    
    # Calculate component values
    baseline = top_players['baseline_pct']
    momentum = top_players['momentum_pct']
    contextual = top_players['contextual_pct']
    
    # Create stacked bars
    plt.barh(y_pos, baseline, bar_width, label='Intrinsic Skill', color='#3498db')
    plt.barh(y_pos, momentum, bar_width, left=baseline, label='Momentum', color='#2ecc71')
    plt.barh(y_pos, contextual, bar_width, left=baseline+momentum, label='Contextual Factors', color='#e74c3c')
    
    # Add labels and title
    plt.yticks(y_pos, top_players['player_name'])
    plt.xlabel('Percentage of Performance', fontsize=12)
    plt.title('Performance Factor Decomposition', fontsize=14)
    plt.legend(loc='lower right')
    
    # Add stability annotations
    for i, player in enumerate(top_players.itertuples()):
        stability_text = f"Stability: {player.system_stability:.2f}"
        plt.text(101, i, stability_text, va='center', fontsize=9)
    
    # Set x-axis limit to make room for annotations
    plt.xlim(0, 130)
    
    # Add explanation
    plt.figtext(0.5, 0.01, 
               "Intrinsic Skill: Baseline performance level\n" +
               "Momentum: Impact of recent performance trends\n" +
               "Contextual: Game situation, matchups, and other external factors", 
               ha='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Save the figure
    plt.savefig('results/performance_decomposition.png')
    print("Performance decomposition visualization saved to results/performance_decomposition.png")
    
    return plt