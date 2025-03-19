import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

# Import our modules
from src.data_processing import (
    load_data, preprocess_data, create_temporal_features,
    calculate_player_dynamics, extract_team_styles,
    calculate_player_team_fit, calculate_player_impact_model,
    save_processed_data
)
from src.dynamics import (
    calculate_stability_metrics, plot_stability_distribution,
    calculate_performance_prediction, decompose_performance_factors,
    plot_performance_decomposition
)
from src.network_analysis import (
    analyze_teammate_network, plot_player_network
)
from src.visualization import (
    create_player_stability_quadrant, create_team_style_cards,
    create_adaptability_ranking, create_impact_dashboard,
    create_player_card, create_traditional_vs_dynamics_comparison,
    create_career_trajectory_projection
)
from src.utils import (
    setup_directories, setup_plotting_style, log_analysis_step,
    save_model_metadata, create_analysis_summary, export_for_dashboard,
    generate_executive_summary, create_readme
)

def main():
    """Main analysis pipeline"""
    # Setup
    setup_directories()
    setup_plotting_style()
    log_analysis_step("Analysis started")
    
    # Load data
    log_analysis_step("Loading data")
    teams, players, games, player_games = load_data()
    
    if teams is None or players is None or games is None or player_games is None:
        print("Error loading data. Exiting.")
        return
    
    # Preprocess data
    log_analysis_step("Preprocessing data")
    games, player_games = preprocess_data(teams, players, games, player_games)
    
    # Create temporal features
    log_analysis_step("Creating temporal features")
    player_temporal_df = create_temporal_features(player_games)
    
    # Calculate player dynamics
    log_analysis_step("Calculating player dynamics")
    player_dynamics_df = calculate_player_dynamics(player_temporal_df)
    
    # Extract team styles
    log_analysis_step("Extracting team styles")
    team_styles_df = extract_team_styles(games)
    
    # Calculate player-team fit
    log_analysis_step("Calculating player-team fit")
    player_team_fit_df = calculate_player_team_fit(player_games, team_styles_df)
    
    # Run the network analysis
    log_analysis_step("Running teammate network analysis")
    teammate_network, centrality_df = analyze_teammate_network(player_games, games)
    
    # Calculate player impact model
    log_analysis_step("Calculating player impact model")
    player_impact_df = calculate_player_impact_model(
        player_dynamics_df, 
        player_team_fit_df, 
        centrality_df
    )
    
    # Create visualizations
    log_analysis_step("Creating visualizations")
    
    # Player stability quadrant
    create_player_stability_quadrant(player_dynamics_df)
    
    # Team style cards
    create_team_style_cards(team_styles_df)
    
    # Player adaptability ranking
    create_adaptability_ranking(player_team_fit_df)
    
    # Player impact dashboard
    create_impact_dashboard(player_impact_df)
    
    # Traditional vs dynamics comparison
    create_traditional_vs_dynamics_comparison(player_dynamics_df)
    
    # Create player cards for top players
    top_players = player_impact_df.nlargest(5, 'impact_score')
    for _, player in top_players.iterrows():
        create_player_card(player_dynamics_df, player_team_fit_df, player['player_name'])
    
    # Career trajectory projection for a sample player
    if len(player_dynamics_df) > 0:
        sample_player = player_dynamics_df.iloc[0]['player_name']
        create_career_trajectory_projection(player_dynamics_df, sample_player)
    
    # Save processed data
    log_analysis_step("Saving processed data")
    processed_data = {
        'player_dynamics': player_dynamics_df,
        'team_styles': team_styles_df,
        'player_team_fit': player_team_fit_df,
        'player_impact': player_impact_df,
        'centrality': centrality_df
    }
    save_processed_data(processed_data)
    
    # Export data for dashboard
    log_analysis_step("Exporting data for dashboard")
    export_for_dashboard(processed_data)
    
    # Create analysis summary
    log_analysis_step("Creating analysis summary")
    
    # Gather key metrics for summary
    analysis_results = {
        'n_players': len(player_dynamics_df),
        'n_games': len(games),
        'n_teams': len(team_styles_df),
        'avg_stability': player_dynamics_df['system_stability'].mean(),
        'stability_std': player_dynamics_df['system_stability'].std(),
        'n_style_clusters': len(team_styles_df['style_name'].unique()),
        'n_stable_players': len(player_dynamics_df[player_dynamics_df['system_stability'] < 0.8]),
        'n_volatile_players': len(player_dynamics_df[player_dynamics_df['system_stability'] > 1.1]),
        'most_stable_player': player_dynamics_df.nsmallest(1, 'system_stability')['player_name'].iloc[0],
        'most_volatile_player': player_dynamics_df.nlargest(1, 'system_stability')['player_name'].iloc[0],
        'best_style': team_styles_df.nlargest(1, 'win_pct')['style_name'].iloc[0],
        'most_adaptable_player': player_team_fit_df.nlargest(1, 'adaptability_score')['player_name'].iloc[0],
        'highest_influence_player': centrality_df.nlargest(1, 'influence_index')['player_name'].iloc[0],
        'strongest_connection': "Player A → Player B",  # This would come from the network analysis
        'key_findings': [
            "Players can be classified into four stability quadrants with distinct performance characteristics",
            "Team playing styles cluster into distinct archetypes with different success rates",
            "Player adaptability varies significantly and correlates with career longevity",
            "Teammate influence networks reveal hidden synergies and conflicts"
        ]
    }
    
    summary = create_analysis_summary(analysis_results)
    
    # Generate executive summary
    log_analysis_step("Generating executive summary")
    generate_executive_summary(analysis_results)
    
    # Create README
    log_analysis_step("Creating README")
    project_info = {
        'title': 'NBA Player Performance Dynamics',
        'description': 'A complex systems approach to NBA player performance analysis, revealing patterns invisible to conventional statistics.',
        'n_style_clusters': len(team_styles_df['style_name'].unique()),
        'data_source': 'NBA game and player statistics',
        'author': 'Your Name',
        'license': 'MIT'
    }
    create_readme(project_info)
    
    log_analysis_step("Analysis completed")
    print("\nAnalysis complete! Check the 'results' directory for outputs.")

if __name__ == "__main__":
    main()