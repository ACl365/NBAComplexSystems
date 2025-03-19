import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

def render_player_card(player_dynamics_df, player_team_fit_df, player_name):
    """Render a player card component in the Streamlit dashboard"""
    # Get player data
    player_dynamics = player_dynamics_df[player_dynamics_df['player_name'] == player_name]
    player_fit = player_team_fit_df[player_team_fit_df['player_name'] == player_name]
    
    if len(player_dynamics) == 0 or len(player_fit) == 0:
        st.warning(f"No data available for {player_name}")
        return
    
    player_dyn = player_dynamics.iloc[0]
    player_fit = player_fit.iloc[0]
    
    # Create columns for the card
    col1, col2 = st.columns([1, 1])
    
    # Column 1: Basic stats and stability gauge
    with col1:
        st.subheader("Player Stats")
        
        # Create metrics
        metrics_col1, metrics_col2 = st.columns(2)
        
        with metrics_col1:
            st.metric("Games Played", f"{player_dyn['games_played']}")
            st.metric("Points Per Game", f"{player_dyn['avg_pts']:.1f}")
            st.metric("Plus/Minus", f"{player_dyn['avg_plus_minus']:.1f}")
        
        with metrics_col2:
            # Determine stability color and label
            stability = player_dyn['system_stability']
            if stability < 0.8:
                stability_color = "green"
                stability_label = "HIGH STABILITY"
            elif stability < 1.1:
                stability_color = "orange"
                stability_label = "MODERATE"
            else:
                stability_color = "red"
                stability_label = "VOLATILE"
            
            st.metric("System Stability", f"{stability:.2f}", 
                     delta=stability_label,
                     delta_color="off")
            
            st.metric("Performance Entropy", f"{player_dyn['performance_entropy']:.2f}")
            st.metric("Adaptability Score", f"{player_fit['adaptability_score']:.2f}")
    
    # Column 2: Style fit and insights
    with col2:
        st.subheader("Team Fit Analysis")
        
        # Best team fit
        st.info(f"**Best Team Style:** {player_fit['best_style']}")
        
        # Determine player type based on metrics
        if stability < 0.8 and player_dyn['avg_plus_minus'] > 0:
            player_type = "HIGH-VALUE STABILITY"
            description = "Consistent positive impact; cornerstone player"
            icon = "🟢"
        elif stability >= 1.1 and player_dyn['avg_plus_minus'] > 0:
            player_type = "HIGH-CEILING VOLATILITY"
            description = "Game-changer with high variance; manage situations"
            icon = "🟠"
        elif stability < 0.8 and player_dyn['avg_plus_minus'] <= 0:
            player_type = "LOW-IMPACT CONSISTENCY" 
            description = "Reliable role player; specific situational value"
            icon = "🟡"
        else:
            player_type = "HIGH-RISK VARIABILITY"
            description = "Unpredictable performance; needs system adjustment"
            icon = "🔴"
        
        # Display player type
        st.success(f"**Player Type:** {icon} {player_type}")
        st.markdown(f"*{description}*")
    
    # Performance visualization
    st.subheader("Performance Visualization")
    
    # Create figure for visualization
    fig = plt.figure(figsize=(10, 6))
    
    # Create mock performance trend (in a real implementation, use actual data)
    games = np.arange(1, 21)
    performance = np.cumsum(np.random.normal(0, 1, 20)) + player_dyn['avg_plus_minus']
    moving_avg = np.convolve(performance, np.ones(5)/5, mode='valid')
    
    plt.plot(games, performance, marker='o', markersize=4, alpha=0.7, label='Game Plus/Minus')
    plt.plot(np.arange(5, 21), moving_avg, linewidth=2, color='red', label='5-Game Moving Average')
    
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
    plt.xlabel("Games")
    plt.ylabel("Plus/Minus")
    plt.title(f"{player_name}: Recent Performance Trend", fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Display the figure
    st.pyplot(fig)
    
    # Business recommendations
    st.subheader("Basketball Operations Recommendations")
    
    # Generate recommendations based on player type
    if player_type == "HIGH-VALUE STABILITY":
        st.markdown("""
        - **Contract Value:** High long-term value, cornerstone player
        - **Usage:** Consistent minutes, high-leverage situations
        - **Development Focus:** Leadership, system mastery
        """)
    elif player_type == "HIGH-CEILING VOLATILITY":
        st.markdown("""
        - **Contract Value:** Moderate with incentives for consistency
        - **Usage:** Strategic deployment, matchup-dependent
        - **Development Focus:** Consistency, mental approach
        """)
    elif player_type == "LOW-IMPACT CONSISTENCY":
        st.markdown("""
        - **Contract Value:** Role player value, system-dependent
        - **Usage:** Specific situations aligned with strengths
        - **Development Focus:** Expanding impact areas
        """)
    else:  # HIGH-RISK VARIABILITY
        st.markdown("""
        - **Contract Value:** Short-term, performance-based
        - **Usage:** Limited minutes, closely monitored
        - **Development Focus:** Fundamentals, decision-making
        """)
    
    return

def render_player_comparison(player_dynamics_df, player_names):
    """Render a comparison of multiple players"""
    if len(player_names) < 2:
        st.warning("Please select at least two players to compare")
        return
    
    # Filter data for selected players
    players_data = player_dynamics_df[player_dynamics_df['player_name'].isin(player_names)]
    
    if len(players_data) == 0:
        st.warning("No data available for the selected players")
        return
    
    # Create comparison visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Points vs Stability
    for _, player in players_data.iterrows():
        ax1.scatter(player['system_stability'], player['avg_pts'], 
                   s=100, label=player['player_name'])
    
    ax1.set_xlabel('System Stability (lower = more stable)')
    ax1.set_ylabel('Points Per Game')
    ax1.set_title('Points vs. Stability')
    ax1.grid(True, alpha=0.3)
    
    # Plus/Minus vs Entropy
    for _, player in players_data.iterrows():
        ax2.scatter(player['performance_entropy'], player['avg_plus_minus'], 
                   s=100, label=player['player_name'])
    
    ax2.set_xlabel('Performance Entropy')
    ax2.set_ylabel('Plus/Minus')
    ax2.set_title('Impact vs. Predictability')
    ax2.grid(True, alpha=0.3)
    
    # Add legend
    handles, labels = ax2.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0), 
              ncol=min(5, len(player_names)))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    
    # Display the figure
    st.pyplot(fig)
    
    # Comparison table
    st.subheader("Player Comparison Table")
    
    comparison_df = players_data[['player_name', 'avg_pts', 'avg_plus_minus', 
                                 'system_stability', 'performance_entropy']]
    comparison_df = comparison_df.rename(columns={
        'player_name': 'Player',
        'avg_pts': 'PPG',
        'avg_plus_minus': '+/-',
        'system_stability': 'Stability',
        'performance_entropy': 'Entropy'
    })
    
    st.dataframe(comparison_df)
    
    return