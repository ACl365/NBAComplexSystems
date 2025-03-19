import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns

def create_player_stability_quadrant(player_dynamics_df):
    """Create a quadrant plot showing performance vs. stability"""
    # Create a 4-quadrant plot
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot each player
    scatter = ax.scatter(
        player_dynamics_df['system_stability'],
        player_dynamics_df['avg_plus_minus'],
        c=player_dynamics_df['avg_pts'],
        cmap='viridis',
        alpha=0.7,
        s=80,
        edgecolors='white',
        linewidths=0.5
    )
    
    # Add quadrant lines
    stability_threshold = 1.0  # Threshold between stable and unstable
    impact_threshold = 0  # Threshold between positive and negative impact
    
    ax.axvline(x=stability_threshold, color='gray', linestyle='--', alpha=0.7)
    ax.axhline(y=impact_threshold, color='gray', linestyle='--', alpha=0.7)
    
    # Add quadrant labels with basketball terminology
    ax.text(0.5, 3, "HIGH-VALUE STABILITY", ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
    ax.text(1.5, 3, "HIGH-CEILING VOLATILITY", ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
    ax.text(0.5, -3, "LOW-IMPACT CONSISTENCY", ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
    ax.text(1.5, -3, "HIGH-RISK VARIABILITY", ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
    
    # Label top players in each quadrant
    quadrants = {
        'Q1': (player_dynamics_df['system_stability'] < stability_threshold) & 
              (player_dynamics_df['avg_plus_minus'] > impact_threshold),
        'Q2': (player_dynamics_df['system_stability'] >= stability_threshold) & 
              (player_dynamics_df['avg_plus_minus'] > impact_threshold),
        'Q3': (player_dynamics_df['system_stability'] < stability_threshold) & 
              (player_dynamics_df['avg_plus_minus'] <= impact_threshold),
        'Q4': (player_dynamics_df['system_stability'] >= stability_threshold) & 
              (player_dynamics_df['avg_plus_minus'] <= impact_threshold)
    }
    
    for q, mask in quadrants.items():
        top_players = player_dynamics_df[mask].nlargest(3, 'avg_pts')
        for _, player in top_players.iterrows():
            ax.annotate(
                player['player_name'],
                (player['system_stability'], player['avg_plus_minus']),
                fontsize=9,
                xytext=(5, 5),
                textcoords='offset points'
            )
    
    # Add colorbar for points
    cbar = plt.colorbar(scatter)
    cbar.set_label('Points per Game', fontsize=12)
    
    # Set labels and title
    ax.set_xlabel('Performance Stability (lower = more consistent)', fontsize=12)
    ax.set_ylabel('Impact (Plus/Minus)', fontsize=12)
    ax.set_title('Player Value Quadrants: Stability vs. Impact', fontsize=14)
    
    # Add explanatory footnote
    plt.figtext(0.1, 0.01, "Stability represents game-to-game consistency, derived from dynamical systems analysis.", 
                fontsize=8, ha="left")
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the figure
    plt.savefig('results/player_stability_quadrant.png')
    print("Player stability quadrant visualization saved to results/player_stability_quadrant.png")
    
    return fig

def create_team_style_cards(team_signatures_df, top_n=6):
    """Create style profile cards for top teams"""
    # Sort by win percentage
    top_teams = team_signatures_df.nlargest(top_n, 'win_pct')
    
    # Create a grid of radar charts
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), subplot_kw=dict(polar=True))
    axes = axes.flatten()
    
    # Style metrics to display
    metrics = ['pace', 'three_point_rate', 'assist_rate', 
               'defensive_focus', 'offensive_efficiency', 'style_entropy']
    metrics_labels = ['Pace', '3PT Rate', 'Ball Movement', 
                    'Defense', 'Off. Efficiency', 'Tactical Diversity']
    
    # Max values for scaling
    max_values = {metric: team_signatures_df[metric].max() for metric in metrics}
    
    # Create radar chart for each team
    for i, (_, team) in enumerate(top_teams.iterrows()):
        if i >= len(axes):
            break
            
        ax = axes[i]
        
        # Scale values from 0 to 1
        values = [team[metric] / max_values[metric] for metric in metrics]
        values.append(values[0])  # Close the loop
        
        # Draw radar
        angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Close the loop
        
        ax.plot(angles, values, linewidth=2, linestyle='solid')
        ax.fill(angles, values, alpha=0.25)
        
        # Add labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics_labels, fontsize=8)
        
        # Team name and style
        team_name = team['team_name'].split()[-1] if len(team['team_name'].split()) > 1 else team['team_name']
        ax.set_title(f"{team_name}\n{team['style_name']}", fontsize=10, fontweight='bold')
        
        # Record and PPG
        win_pct = team['win_pct'] * 100
        ax.text(0, 0, f"Win%: {win_pct:.1f}\nPPG: {team['avg_pts']:.1f}", 
                ha='center', va='center', fontsize=9)
        
    # Remove empty subplots if any
    for i in range(len(top_teams), len(axes)):
        fig.delaxes(axes[i])
        
    plt.tight_layout()
    fig.suptitle('Team Style Profiles', fontsize=16, y=1.05)
    
    # Save the figure
    plt.savefig('results/team_style_cards.png')
    print("Team style cards visualization saved to results/team_style_cards.png")
    
    return fig

def create_adaptability_ranking(player_team_fit_df, min_games=20):
    """Create a visual ranking of player adaptability"""
    # Filter players with enough games
    qualified = player_team_fit_df[player_team_fit_df['games_played'] >= min_games]
    
    # Get top 15 most and least adaptable players
    most_adaptable = qualified.nlargest(15, 'adaptability_score')
    least_adaptable = qualified.nsmallest(15, 'adaptability_score')
    
    # Create the figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 10))
    
    # Plot most adaptable
    y_pos = np.arange(len(most_adaptable))
    bars1 = ax1.barh(y_pos, most_adaptable['adaptability_score'], height=0.7)
    
    # Color bars by plus/minus
    norm = plt.Normalize(most_adaptable['best_style_plus_minus'].min(), 
                         most_adaptable['best_style_plus_minus'].max())
    colors = plt.cm.RdYlGn(norm(most_adaptable['best_style_plus_minus']))
    
    for bar, color in zip(bars1, colors):
        bar.set_color(color)
    
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(most_adaptable['player_name'])
    ax1.invert_yaxis()
    ax1.set_xlabel('Adaptability Score')
    ax1.set_title('Most Adaptable Players', fontsize=14)
    
    # Add best fit annotation
    for i, player in enumerate(most_adaptable.itertuples()):
        ax1.text(player.adaptability_score + 0.01, i, 
                f"Best: {player.best_style}", fontsize=8, va='center')
    
    # Plot least adaptable
    y_pos = np.arange(len(least_adaptable))
    bars2 = ax2.barh(y_pos, least_adaptable['adaptability_score'], height=0.7)
    
    # Color bars by gap size
    norm = plt.Normalize(least_adaptable['style_gap'].min(), 
                         least_adaptable['style_gap'].max())
    colors = plt.cm.RdYlBu_r(norm(least_adaptable['style_gap']))
    
    for bar, color in zip(bars2, colors):
        bar.set_color(color)
    
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(least_adaptable['player_name'])
    ax2.invert_yaxis()
    ax2.set_xlabel('Adaptability Score')
    ax2.set_title('Most Style-Dependent Players', fontsize=14)
    
    # Add style preference annotation
    for i, player in enumerate(least_adaptable.itertuples()):
        gap_text = f"{player.best_style} (+{player.style_gap:.1f})"
        ax2.text(player.adaptability_score + 0.01, i, gap_text, fontsize=8, va='center')
    
    plt.suptitle('Player Adaptability Rankings', fontsize=16)
    plt.tight_layout()
    
    # Add explanation
    plt.figtext(0.5, 0.01, 
                "Adaptability Score measures a player's ability to maintain performance across different team styles.\nHigher score = more versatile player. Lower score = more system-dependent player.", 
                ha='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    
    # Save the figure
    plt.savefig('results/player_adaptability_ranking.png')
    print("Player adaptability ranking visualization saved to results/player_adaptability_ranking.png")
    
    return fig

def create_impact_dashboard(player_impact_df, n_players=10):
    """Create an executive dashboard of player impact"""
    top_players = player_impact_df.nlargest(n_players, 'impact_score')
    
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 10))
    
    # Define grid layout
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1, 1.5])
    
    # Total impact ranking
    ax1 = fig.add_subplot(gs[0, 0])
    y_pos = np.arange(len(top_players))
    ax1.barh(y_pos, top_players['impact_score'], height=0.7)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(top_players['player_name'])
    ax1.invert_yaxis()
    ax1.set_title('Total Impact Score', fontsize=12)
    
    # Impact breakdown - stacked bar
    ax2 = fig.add_subplot(gs[0, 1])
    bottom = np.zeros(len(top_players))
    
    components = [
        ('production_component', 'Production', '#3498db'),
        ('stability_component', 'Consistency', '#2ecc71'),
        ('adaptability_component', 'Adaptability', '#f39c12'),
        ('network_component', 'Teammate Influence', '#9b59b6')
    ]
    
    for component, label, color in components:
        values = top_players[component].values
        ax2.barh(y_pos, values, left=bottom, height=0.7, label=label, color=color)
        bottom += values
    
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([])
    ax2.invert_yaxis()
    ax2.set_title('Impact Components', fontsize=12)
    ax2.legend(loc='lower right')
    
    # Component comparison - radar chart
    ax3 = fig.add_subplot(gs[1, :], polar=True)
    
    # Prepare data for radar chart
    categories = ['Production', 'Consistency', 'Adaptability', 'Teammate Influence']
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Draw radar for each player
    for i, player in enumerate(top_players.itertuples()):
        values = [
            player.production_component / top_players['production_component'].max(),
            player.stability_component / top_players['stability_component'].max(),
            player.adaptability_component / top_players['adaptability_component'].max(),
            player.network_component / top_players['network_component'].max() if 'network_component' in top_players else 0
        ]
        values += values[:1]  # Close the loop
        
        # Plot the player's line
        ax3.plot(angles, values, linewidth=2, linestyle='solid', 
                 label=player.player_name, alpha=0.8)
        
        # If this is a key player, fill the area
        if i < 5:
            ax3.fill(angles, values, alpha=0.1)
    
    # Set radar chart properties
    ax3.set_xticks(angles[:-1])
    ax3.set_xticklabels(categories)
    ax3.set_title('Multi-Dimensional Player Comparison', fontsize=14)
    ax3.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    plt.tight_layout()
    fig.suptitle('Player Impact Dashboard', fontsize=16, y=0.98)
    
    # Add explanation
    plt.figtext(0.5, 0.01, 
               "This dashboard shows player impact across four dimensions: traditional production (pts, +/-),\n" +
               "performance consistency, style adaptability, and influence on teammates.", 
               ha='center', fontsize=10)
    
    # Save the figure
    plt.savefig('results/player_impact_dashboard.png')
    print("Player impact dashboard visualization saved to results/player_impact_dashboard.png")
    
    return fig

def create_player_card(player_dynamics_df, player_team_fit_df, player_name):
    """Create an intuitive player card visualization"""
    # Get player data
    player_dynamics = player_dynamics_df[player_dynamics_df['player_name'] == player_name]
    player_fit = player_team_fit_df[player_team_fit_df['player_name'] == player_name]
    
    if len(player_dynamics) == 0 or len(player_fit) == 0:
        return None
    
    player_dyn = player_dynamics.iloc[0]
    player_fit = player_fit.iloc[0]
    
    # Create player card figure
    fig = plt.figure(figsize=(8, 10))
    
    # Define grid layout
    gs = GridSpec(4, 2, figure=fig, height_ratios=[1, 2, 2, 1])
    
    # Header with player name
    ax_header = fig.add_subplot(gs[0, :])
    ax_header.text(0.5, 0.5, player_name, fontsize=24, weight='bold', ha='center', va='center')
    ax_header.axis('off')
    
    # Basic stats panel
    ax_stats = fig.add_subplot(gs[1, 0])
    ax_stats.axis('off')
    
    stats_text = (
        f"Games Played: {player_dyn['games_played']}\n"
        f"Points Per Game: {player_dyn['avg_pts']:.1f}\n"
        f"Plus/Minus: {player_dyn['avg_plus_minus']:.1f}\n"
        f"Performance Entropy: {player_dyn['performance_entropy']:.2f}\n"
        f"System Stability: {player_dyn['system_stability']:.2f}\n"
        f"Adaptability Score: {player_fit['adaptability_score']:.2f}\n"
    )
    
    ax_stats.text(0.1, 0.9, "Player Stats", fontsize=14, weight='bold')
    ax_stats.text(0.1, 0.8, stats_text, fontsize=12, va='top', linespacing=1.5)
    
    # Stability gauge
    ax_stability = fig.add_subplot(gs[1, 1])
    
    stability = player_dyn['system_stability']
    stability_normalized = min(1, stability / 2)  # Normalize to 0-1 range
    
    gauge = plt.cm.RdYlGn_r(stability_normalized)
    ax_stability.add_patch(plt.Circle((0.5, 0.5), 0.4, color=gauge))
    
    stability_text = "HIGH STABILITY" if stability < 0.8 else "MODERATE" if stability < 1.1 else "VOLATILE"
    ax_stability.text(0.5, 0.5, stability_text, fontsize=14, weight='bold', ha='center', va='center')
    ax_stability.text(0.5, 0.3, f"{stability:.2f}", fontsize=18, ha='center', va='center')
    
    ax_stability.set_xlim(0, 1)
    ax_stability.set_ylim(0, 1)
    ax_stability.axis('off')
    
    # Style fit radar chart
    ax_radar = fig.add_subplot(gs[2, 0], polar=True)
    
    # Set radar chart properties
    categories = ['Pace & Space', 'Inside-Out', 'Defensive', 'Balanced']
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Style fit values (could be derived from actual data)
    if player_fit['best_style'] == 'Modern Pace-and-Space':
        values = [0.9, 0.6, 0.5, 0.7]
    elif player_fit['best_style'] == 'Traditional Inside-Out':
        values = [0.6, 0.9, 0.7, 0.5]
    elif player_fit['best_style'] == 'Defensive-Oriented':
        values = [0.5, 0.7, 0.9, 0.6]
    else:
        values = [0.7, 0.7, 0.7, 0.9]
    
    values += values[:1]  # Close the loop
    
    ax_radar.plot(angles, values, linewidth=2)
    ax_radar.fill(angles, values, alpha=0.25)
    
    # Set radar chart properties
    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels(categories, fontsize=8)
    ax_radar.set_yticklabels([])
    ax_radar.set_title("Style Fit", fontsize=14)
    
    # Performance trend panel
    ax_trend = fig.add_subplot(gs[2, 1])
    
    # Create a mock performance trend (in a real implementation, use actual data)
    games = np.arange(1, 21)
    performance = np.cumsum(np.random.normal(0, 1, 20)) + player_dyn['avg_plus_minus']
    moving_avg = np.convolve(performance, np.ones(5)/5, mode='valid')
    
    ax_trend.plot(games, performance, marker='o', markersize=4, alpha=0.7)
    ax_trend.plot(np.arange(5, 21), moving_avg, linewidth=2, color='red')
    
    ax_trend.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
    ax_trend.set_xlabel("Games")
    ax_trend.set_ylabel("Plus/Minus")
    ax_trend.set_title("Recent Performance Trend", fontsize=14)
    
    # Key insights
    ax_insights = fig.add_subplot(gs[3, :])
    ax_insights.axis('off')
    
    # Determine player type based on metrics
    if stability < 0.8 and player_dyn['avg_plus_minus'] > 0:
        player_type = "HIGH-VALUE STABILITY"
        description = "Consistent positive impact; cornerstone player"
    elif stability >= 1.1 and player_dyn['avg_plus_minus'] > 0:
        player_type = "HIGH-CEILING VOLATILITY"
        description = "Game-changer with high variance; manage situations"
    elif stability < 0.8 and player_dyn['avg_plus_minus'] <= 0:
        player_type = "LOW-IMPACT CONSISTENCY" 
        description = "Reliable role player; specific situational value"
    else:
        player_type = "HIGH-RISK VARIABILITY"
        description = "Unpredictable performance; needs system adjustment"
    
    # Add best team fit
    best_fit = f"Best Team Fit: {player_fit['best_style']}"
    
    # Add insights text
    ax_insights.text(0.5, 0.8, "KEY INSIGHTS", fontsize=14, weight='bold', ha='center')
    ax_insights.text(0.5, 0.6, player_type, fontsize=12, weight='bold', ha='center')
    ax_insights.text(0.5, 0.4, description, fontsize=10, ha='center')
    ax_insights.text(0.5, 0.2, best_fit, fontsize=10, ha='center', style='italic')
    
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.4)
    
    # Save the figure
    plt.savefig(f'results/player_card_{player_name.replace(" ", "_")}.png')
    print(f"Player card for {player_name} saved to results/player_card_{player_name.replace(' ', '_')}.png")
    
    return fig

def create_traditional_vs_dynamics_comparison(player_dynamics_df, min_games=20):
    """Compare traditional metrics with dynamics-based metrics"""
    # Filter players with enough games
    qualified = player_dynamics_df[player_dynamics_df['games_played'] >= min_games]
    
    # Calculate traditional metrics
    qualified['traditional_score'] = (qualified['avg_pts'] * 0.6 + 
                                     qualified['avg_plus_minus'] * 0.4)
    
    # Calculate dynamics-based score
    qualified['dynamics_score'] = (qualified['avg_pts'] * 0.4 + 
                                  qualified['avg_plus_minus'] * 0.3 +
                                  (10 / (1 + qualified['system_stability'])) * 0.3)
    
    # Find players with biggest differences
    qualified['metric_difference'] = qualified['dynamics_score'] - qualified['traditional_score']
    
    # Normalize for visualization
    max_trad = qualified['traditional_score'].max()
    max_dyn = qualified['dynamics_score'].max()
    qualified['traditional_normalized'] = qualified['traditional_score'] / max_trad
    qualified['dynamics_normalized'] = qualified['dynamics_score'] / max_dyn
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10))
    
    # Players undervalued by traditional metrics
    undervalued = qualified.nlargest(10, 'metric_difference')
    
    # Plot comparison
    x = np.arange(len(undervalued))
    width = 0.35
    
    ax1.bar(x - width/2, undervalued['traditional_normalized'], width, 
           label='Traditional Rating', color='#3498db')
    ax1.bar(x + width/2, undervalued['dynamics_normalized'], width,
           label='Dynamics Rating', color='#e74c3c')
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(undervalued['player_name'], rotation=45, ha='right')
    ax1.legend()
    ax1.set_title('Players Undervalued by Traditional Metrics', fontsize=14)
    ax1.set_ylim(0, 1.2)
    
    # Players overvalued by traditional metrics
    overvalued = qualified.nsmallest(10, 'metric_difference')
    
    ax2.bar(x - width/2, overvalued['traditional_normalized'], width, 
           label='Traditional Rating', color='#3498db')
    ax2.bar(x + width/2, overvalued['dynamics_normalized'], width,
           label='Dynamics Rating', color='#e74c3c')
    
    ax2.set_xticks(x)
    ax2.set_xticklabels(overvalued['player_name'], rotation=45, ha='right')
    ax2.legend()
    ax2.set_title('Players Overvalued by Traditional Metrics', fontsize=14)
    ax2.set_ylim(0, 1.2)
    
    plt.suptitle('Traditional vs. Dynamics-Based Player Evaluation', fontsize=16)
    plt.tight_layout()
    
    # Save the figure
    plt.savefig('results/traditional_vs_dynamics.png')
    print("Traditional vs dynamics comparison visualization saved to results/traditional_vs_dynamics.png")
    
    return fig, qualified[['player_name', 'traditional_score', 'dynamics_score', 'metric_difference']]

def create_career_trajectory_projection(player_dynamics_df, player_name, player_age=25):
    """Project future career trajectory based on dynamics model"""
    # Get player data
    player = player_dynamics_df[player_dynamics_df['player_name'] == player_name]
    
    if len(player) == 0:
        return None
    
    player = player.iloc[0]
    
    # Create career timeline
    ages = np.arange(player_age, player_age + 10)
    years = np.arange(2023, 2033)
    
    # Baseline projection
    points_projection = []
    stability_projection = []
    
    # Current values
    current_pts = player['avg_pts']
    current_stability = player['system_stability']
    
    for age in ages:
        # Simple model: Players peak at 27-29, then decline
        if age < 27:
            pts_factor = 1 + (age - player_age) * 0.03
        elif age < 30:
            pts_factor = 1 + (27 - player_age) * 0.03
        else:
            # Decline phase
            pts_factor = 1 + (27 - player_age) * 0.03 - (age - 29) * 0.04
        
        # Stability typically improves with age
        if age < 33:
            stability_factor = max(0.7, current_stability - (age - player_age) * 0.03)
        else:
            # Late career may become less consistent
            stability_factor = max(0.7, current_stability - (32 - player_age) * 0.03 + (age - 32) * 0.02)
        
        points_projection.append(current_pts * pts_factor)
        stability_projection.append(stability_factor)
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), height_ratios=[3, 2])
    
    # Points projection
    ax1.plot(ages, points_projection, marker='o', markersize=8, linewidth=2)
    ax1.set_xlabel('Age')
    ax1.set_ylabel('Projected Points Per Game')
    ax1.grid(True, alpha=0.3)
    
    # Annotate peak
    peak_idx = np.argmax(points_projection)
    peak_age = ages[peak_idx]
    peak_pts = points_projection[peak_idx]
    
    ax1.annotate(f"Projected Peak: {peak_pts:.1f} PPG",
                xy=(peak_age, peak_pts), xytext=(peak_age+1, peak_pts+2),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
    
    # Add age milestones
    for i, age in enumerate(ages):
        if age in [27, 30, 33]:
            ax1.axvline(x=age, color='gray', linestyle='--', alpha=0.5)
            ax1.text(age, min(points_projection) - 1, f"Age {age}", 
                    ha='center', fontsize=9)
    
    # Stability projection
    ax2.plot(ages, stability_projection, marker='s', color='green', markersize=8, linewidth=2)
    ax2.set_xlabel('Age')
    ax2.set_ylabel('Projected System Stability')
    ax2.set_ylim(0, max(stability_projection) * 1.5)
    ax2.grid(True, alpha=0.3)
    
    # Add stability interpretation
    ax2.axhspan(0, 0.8, color='green', alpha=0.2, label='High Stability')
    ax2.axhspan(0.8, 1.1, color='yellow', alpha=0.2, label='Moderate Stability')
    ax2.axhspan(1.1, 2, color='red', alpha=0.2, label='Low Stability')
    ax2.legend(loc='upper right')
    
    # Title
    plt.suptitle(f"{player_name}: 10-Year Career Projection", fontsize=16)
    
    # Add caption
    fig.text(0.5, 0.01, 
            "Projection based on dynamical systems model analysis of performance patterns.\n" +
            "Stability improvements reflect typical career development of game awareness and consistency.",
            ha='center', fontsize=10)
    
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(f'results/career_projection_{player_name.replace(" ", "_")}.png')
    print(f"Career projection for {player_name} saved to results/career_projection_{player_name.replace(' ', '_')}.png")
    
    return fig