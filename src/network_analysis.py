import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from IPython.display import display

def add_team_info_to_player_games(player_games_df, games_df):
    """Add team identification to player game records"""
    
    # Create a mapping from Game_ID to Team_ID and opponent Team_ID
    game_team_map = {}
    for _, game_row in games_df.iterrows():
        game_id = game_row['Game_ID']
        team_id = game_row['Team_ID']
        
        # Extract opponent team from MATCHUP
        matchup = game_row['MATCHUP']
        is_home = ' vs.' in matchup
        teams = matchup.replace(' vs. ', ' @ ').split(' @ ')
        team_abbr = teams[0]
        
        if game_id not in game_team_map:
            game_team_map[game_id] = []
        
        game_team_map[game_id].append((team_id, team_abbr, is_home))
    
    # Add Team_ID to player_games
    team_id_list = []
    
    for _, row in player_games_df.iterrows():
        game_id = row['Game_ID']
        matchup = row['MATCHUP']
        player_wl = row['WL']  # Win/Loss status for this player
        
        if game_id in game_team_map:
            # Find the team that matches this player's Win/Loss status
            for team_id, team_abbr, is_home in game_team_map[game_id]:
                game_teams = matchup.replace(' vs. ', ' @ ').split(' @ ')
                player_team_abbr = game_teams[0] if ' vs.' in matchup else game_teams[1]
                
                if team_abbr == player_team_abbr:
                    team_id_list.append(team_id)
                    break
            else:
                # If no match found (should be rare)
                team_id_list.append(None)
        else:
            team_id_list.append(None)
    
    player_games_df['Team_ID'] = team_id_list
    
    return player_games_df

def create_teammate_influence_network(player_games_df, min_games_together=5):
    """Create a network of teammate interactions and influence"""
    # Group players by game and team to identify teammates
    game_team_players = {}
    
    for _, row in player_games_df.iterrows():
        game_id = row['Game_ID']
        team_id = row['Team_ID']
        player_id = row['Player_ID']
        
        if team_id is None:
            continue
            
        team_game_key = (game_id, team_id)
        if team_game_key not in game_team_players:
            game_team_players[team_game_key] = []
            
        game_team_players[team_game_key].append((player_id, row['PLUS_MINUS'], row['PTS']))
    
    # Count co-occurrences and calculate influence ONLY between teammates
    player_pairs = {}
    
    for (game_id, team_id), players in game_team_players.items():
        for i, (player1, pm1, pts1) in enumerate(players):
            for j, (player2, pm2, pts2) in enumerate(players):
                if i != j:  # Don't pair a player with themselves
                    pair = (min(player1, player2), max(player1, player2))
                    
                    if pair not in player_pairs:
                        player_pairs[pair] = {
                            'count': 0,
                            'pm_correlation': [],
                            'pts_correlation': []
                        }
                        
                    player_pairs[pair]['count'] += 1
                    player_pairs[pair]['pm_correlation'].append((pm1, pm2))
                    player_pairs[pair]['pts_correlation'].append((pts1, pts2))
    
    # Create network edges
    edges = []
    
    for (player1, player2), data in player_pairs.items():
        if data['count'] >= min_games_together:
            # Calculate correlation of plus/minus if enough data points
            if len(data['pm_correlation']) >= 5:
                pm_values1 = [p[0] for p in data['pm_correlation']]
                pm_values2 = [p[1] for p in data['pm_correlation']]
                
                # Check for variation in the data
                if np.std(pm_values1) > 0 and np.std(pm_values2) > 0:
                    pm_corr = np.corrcoef(pm_values1, pm_values2)[0, 1]
                else:
                    pm_corr = 0
                
                # Calculate correlation of points
                pts_values1 = [p[0] for p in data['pts_correlation']]
                pts_values2 = [p[1] for p in data['pts_correlation']]
                
                if np.std(pts_values1) > 0 and np.std(pts_values2) > 0:
                    pts_corr = np.corrcoef(pts_values1, pts_values2)[0, 1]
                else:
                    pts_corr = 0
                
                # Skip if NaN
                if np.isnan(pm_corr) or np.isnan(pts_corr):
                    continue
                    
                edges.append((
                    player1,
                    player2,
                    {
                        'weight': data['count'],
                        'pm_correlation': pm_corr,
                        'pts_correlation': pts_corr,
                        'influence_score': 0.7 * pm_corr + 0.3 * pts_corr
                    }
                ))
    
    # Create network
    G = nx.Graph()
    
    # Add players as nodes
    for player_id in player_games_df['Player_ID'].unique():
        player_games_subset = player_games_df[player_games_df['Player_ID'] == player_id]
        if len(player_games_subset) > 0:
            player_name = player_games_subset['PlayerName'].iloc[0]
            G.add_node(player_id, name=player_name)
    
    # Add edges
    G.add_edges_from(edges)
    
    return G

# Function to analyze the teammate network
def analyze_teammate_network(player_games_df, games_df):
    """Analyze teammate influence network"""
    # Add team information to player game records
    player_games_with_teams = add_team_info_to_player_games(player_games_df, games_df)
    
    # Create teammate influence network
    teammate_network = create_teammate_influence_network(player_games_with_teams)
    
    # Display basic network statistics
    print(f"Teammate Influence Network Statistics:")
    print(f"Number of players (nodes): {teammate_network.number_of_nodes()}")
    print(f"Number of connections (edges): {teammate_network.number_of_edges()}")
    print(f"Network density: {nx.density(teammate_network):.4f}")
    
    # Calculate centrality metrics
    centrality_data = []
    
    for player_id in teammate_network.nodes():
        try:
            player_name = teammate_network.nodes[player_id]['name']
            
            # Calculate centrality metrics
            degree = nx.degree_centrality(teammate_network)[player_id]
            
            # Check if the graph is connected enough for betweenness centrality
            if nx.number_connected_components(teammate_network) < teammate_network.number_of_nodes():
                betweenness = nx.betweenness_centrality(teammate_network)[player_id]
            else:
                betweenness = 0
            
            # Try eigenvector centrality, fall back to degree if it fails
            try:
                eigenvector = nx.eigenvector_centrality(teammate_network, max_iter=1000)[player_id]
            except:
                eigenvector = degree
            
            centrality_data.append({
                'player_id': player_id,
                'player_name': player_name,
                'degree_centrality': degree,
                'betweenness_centrality': betweenness,
                'eigenvector_centrality': eigenvector,
                'influence_index': 0.2 * degree + 0.3 * betweenness + 0.5 * eigenvector
            })
        except Exception as e:
            print(f"Error calculating centrality for player {player_id}: {e}")
            continue
    
    # Convert to dataframe and display top influencers
    centrality_df = pd.DataFrame(centrality_data)
    
    print("\nTop Players by Teammate Network Influence:")
    display(centrality_df.sort_values('influence_index', ascending=False).head(15)[
        ['player_name', 'influence_index', 'degree_centrality', 'betweenness_centrality', 'eigenvector_centrality']
    ])
    
    # Find positive and negative player connections
    positive_influences = []
    negative_influences = []
    
    for u, v, data in teammate_network.edges(data=True):
        u_name = teammate_network.nodes[u]['name']
        v_name = teammate_network.nodes[v]['name']
        
        if data['influence_score'] > 0.5:
            positive_influences.append((u_name, v_name, data['influence_score'], data['weight']))
        elif data['influence_score'] < -0.5:
            negative_influences.append((u_name, v_name, data['influence_score'], data['weight']))
    
    print("\nStrongest Positive Teammate Influences:")
    for u, v, score, games in sorted(positive_influences, key=lambda x: x[2], reverse=True)[:10]:
        print(f"{u} → {v}: {score:.3f} (based on {games} games)")
    
    print("\nStrongest Negative Teammate Influences:")
    for u, v, score, games in sorted(negative_influences, key=lambda x: x[2])[:10]:
        print(f"{u} → {v}: {score:.3f} (based on {games} games)")
    
    # Visualize the network (limited to strongest connections for clarity)
    plot_player_network(teammate_network, title="Teammate Influence Network")
    
    return teammate_network, centrality_df

# Helper function to visualize the network
def plot_player_network(G, title="Teammate Influence Network", min_influence=0.3, max_nodes=50):
    """Plot player influence network visualization"""
    # Create subgraph with only stronger connections
    strong_edges = [(u, v) for u, v, d in G.edges(data=True) if abs(d['influence_score']) >= min_influence]
    
    if not strong_edges:
        print("No edges meet the minimum influence threshold. Try lowering min_influence.")
        return
        
    H = G.edge_subgraph(strong_edges)
    
    # Limit to top players by degree if needed
    if H.number_of_nodes() > max_nodes:
        degrees = dict(H.degree())
        top_nodes = sorted(degrees.keys(), key=lambda x: degrees[x], reverse=True)[:max_nodes]
        H = H.subgraph(top_nodes)
    
    # Set up layout
    pos = nx.spring_layout(H, seed=42)
    
    plt.figure(figsize=(12, 10))
    
    # Draw nodes
    node_sizes = [100 + 500 * nx.degree_centrality(H)[node] for node in H.nodes()]
    nx.draw_networkx_nodes(H, pos, node_size=node_sizes, alpha=0.8, node_color='lightblue')
    
    # Draw edges with color based on influence (positive=blue, negative=red)
    edge_colors = [H[u][v]['influence_score'] for u, v in H.edges()]
    edge_alphas = [0.5 + 0.5 * abs(H[u][v]['influence_score']) for u, v in H.edges()]
    
    edges = nx.draw_networkx_edges(
        H, pos,
        width=2,
        edge_color=edge_colors,
        edge_cmap=plt.cm.coolwarm,
        edge_vmin=-1,
        edge_vmax=1,
        alpha=edge_alphas
    )
    
    # Add node labels for high-degree nodes
    node_labels = {node: H.nodes[node]['name'] for node in H.nodes()}
    nx.draw_networkx_labels(H, pos, labels=node_labels, font_size=8)
    
    plt.title(title, fontsize=14)
    plt.colorbar(edges, label='Influence Score')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('results/network_influence.png')
    plt.close()
    
    print(f"Network visualization saved to results/network_influence.png")
    return plt