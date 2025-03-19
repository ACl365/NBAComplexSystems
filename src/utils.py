import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime

def setup_directories():
    """Create necessary directories for the project"""
    directories = [
        'data/processed',
        'results',
        'results/model_outputs',
        'results/visualizations',
        'presentation'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

def setup_plotting_style():
    """Set up consistent plotting style for visualizations"""
    # Set seaborn style
    sns.set_style("whitegrid")
    
    # Set matplotlib parameters
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['figure.titlesize'] = 16
    
    # Set color palette
    sns.set_palette("viridis")
    
    print("Plotting style configured")

def log_analysis_step(step_name, details=None):
    """Log analysis steps with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {step_name}"
    
    if details:
        log_entry += f": {details}"
    
    print(log_entry)
    
    # Append to log file
    with open('results/analysis_log.txt', 'a') as f:
        f.write(log_entry + '\n')

def save_model_metadata(model_name, metadata):
    """Save model metadata to JSON file"""
    output_path = f"results/model_outputs/{model_name}_metadata.json"
    
    # Convert numpy types to Python native types
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    # Process metadata
    processed_metadata = {}
    for key, value in metadata.items():
        if isinstance(value, dict):
            processed_metadata[key] = {k: convert_numpy(v) for k, v in value.items()}
        else:
            processed_metadata[key] = convert_numpy(value)
    
    # Save to file
    with open(output_path, 'w') as f:
        json.dump(processed_metadata, f, indent=2)
    
    print(f"Model metadata saved to {output_path}")

def create_analysis_summary(results_dict):
    """Create a summary of analysis results"""
    summary = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'dataset_sizes': {
            'players': results_dict.get('n_players', 0),
            'games': results_dict.get('n_games', 0),
            'teams': results_dict.get('n_teams', 0)
        },
        'model_metrics': {
            'player_dynamics': {
                'avg_stability': results_dict.get('avg_stability', 0),
                'stability_std': results_dict.get('stability_std', 0)
            },
            'team_styles': {
                'n_clusters': results_dict.get('n_style_clusters', 0),
                'silhouette_score': results_dict.get('style_silhouette', 0)
            },
            'prediction': {
                'pts_rmse': results_dict.get('pts_rmse', 0),
                'pts_r2': results_dict.get('pts_r2', 0),
                'pm_rmse': results_dict.get('pm_rmse', 0),
                'pm_r2': results_dict.get('pm_r2', 0)
            }
        },
        'key_findings': results_dict.get('key_findings', [])
    }
    
    # Save summary to file
    output_path = "results/analysis_summary.json"
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Analysis summary saved to {output_path}")
    
    return summary

def export_for_dashboard(dataframes_dict, output_dir='dashboard/data'):
    """Export processed dataframes for use in the dashboard"""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save each dataframe
    for name, df in dataframes_dict.items():
        output_path = f"{output_dir}/{name}.csv"
        df.to_csv(output_path, index=False)
        print(f"Exported {name} to {output_path}")

def generate_executive_summary(analysis_results, output_path='presentation/executive_summary.md'):
    """Generate an executive summary markdown file"""
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Format the summary
    summary = f"""# NBA Player Performance Dynamics: Executive Summary

## Project Overview
This analysis applies complex systems theory to NBA player performance data, revealing patterns and insights invisible to conventional statistical approaches.

## Key Findings

1. **Player Stability Classification**
   - {analysis_results.get('n_stable_players', 0)} players classified as "Highly Stable"
   - {analysis_results.get('n_volatile_players', 0)} players classified as "Volatile" or "Chaotic"
   - Most consistent player: {analysis_results.get('most_stable_player', 'N/A')}
   - Most volatile player: {analysis_results.get('most_volatile_player', 'N/A')}

2. **Team Style Analysis**
   - Identified {analysis_results.get('n_style_clusters', 0)} distinct playing styles
   - Most successful style: {analysis_results.get('best_style', 'N/A')}
   - Most adaptable player: {analysis_results.get('most_adaptable_player', 'N/A')}

3. **Network Influence**
   - Highest teammate influence: {analysis_results.get('highest_influence_player', 'N/A')}
   - Strongest positive connection: {analysis_results.get('strongest_connection', 'N/A')}

4. **Performance Prediction**
   - Points prediction accuracy: {analysis_results.get('pts_r2', 0):.2f} R²
   - Plus/Minus prediction accuracy: {analysis_results.get('pm_r2', 0):.2f} R²

## Business Impact

- **Roster Construction**: Optimize team composition with complementary stability profiles
- **Player Valuation**: Identify undervalued players with favorable dynamics metrics
- **Game Strategy**: Target opponent weaknesses based on stability patterns
- **Player Development**: Focus training on specific aspects of performance consistency

## Methodology

This analysis combines dynamical systems theory, network analysis, and machine learning to extract deeper insights from basketball performance data. The approach reveals patterns in:

- Game-to-game consistency and volatility
- Adaptability across different team systems
- Teammate influence networks
- Performance predictability

## Next Steps

1. Incorporate spatial data for court-region specific analysis
2. Track stability metrics over time to identify career transition points
3. Develop interactive tools for coaching staff decision support

*Generated on {datetime.now().strftime("%Y-%m-%d")}*
"""
    
    # Write to file
    with open(output_path, 'w') as f:
        f.write(summary)
    
    print(f"Executive summary generated at {output_path}")

def create_readme(project_info, output_path='README.md'):
    """Create a README.md file for the GitHub repository"""
    # Format the README
    readme = f"""# {project_info.get('title', 'NBA Player Performance Dynamics')}

![Player Stability Quadrant](results/player_stability_quadrant.png)

## Overview
{project_info.get('description', 'A complex systems approach to NBA player performance analysis.')}

## Key Features
- **Dynamical Systems Analysis**: Quantify player performance stability and predictability
- **Team Style Extraction**: Identify distinct playing styles and tactical patterns
- **Player-Team Fit Analysis**: Measure player adaptability across different systems
- **Network Analysis**: Map teammate influence and synergy patterns
- **Performance Prediction**: Decompose and forecast player performance

## Business Applications
- Identify undervalued players with favorable stability metrics
- Optimize roster construction with complementary player types
- Target opponent weaknesses based on stability patterns
- Develop tailored player development strategies

## Repository Structure
```
nba-dynamics/
├── README.md                  # Project overview
├── presentation/              # Portfolio materials
│   ├── case_study.md          # Narrative case study
│   ├── dashboard_demo.gif     # Animated demo of interactive features
│   └── executive_summary.pdf  # 1-page business impact summary
├── notebooks/                 # Analysis notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_dynamical_systems.ipynb
│   ├── 03_team_styles.ipynb
│   ├── 04_player_fit.ipynb
│   └── 05_network_analysis.ipynb
├── src/                       # Clean, modular code
│   ├── data_processing.py
│   ├── dynamics.py
│   ├── network_analysis.py
│   ├── visualization.py
│   └── utils.py
├── dashboard/                 # Interactive dashboard code
│   ├── app.py                 # Streamlit/Dash app
│   └── components/            # Dashboard components
└── tests/                     # Unit tests
    └── test_dynamics.py       # Shows software engineering rigor
```

## Quick Demo
![Dashboard Demo](presentation/dashboard_demo.gif)

## Key Findings
1. Players can be classified into four stability quadrants with distinct performance characteristics
2. Team playing styles cluster into {project_info.get('n_style_clusters', 4)} distinct archetypes
3. Player adaptability varies significantly and correlates with career longevity
4. Teammate influence networks reveal hidden synergies and conflicts

## Setup Instructions
```bash
# Clone the repository
git clone https://github.com/username/nba-dynamics.git
cd nba-dynamics

# Install dependencies
pip install -r requirements.txt

# Run the analysis
python run_analysis.py

# Launch the dashboard
cd dashboard
streamlit run app.py
```

## Data Sources
This project uses NBA game and player data from {project_info.get('data_source', 'public NBA statistics APIs')}.

## License
{project_info.get('license', 'MIT')}

## Author
{project_info.get('author', 'Your Name')}

## Acknowledgements
{project_info.get('acknowledgements', 'Thanks to the NBA for providing the data used in this analysis.')}
"""
    
    # Write to file
    with open(output_path, 'w') as f:
        f.write(readme)
    
    print(f"README.md generated at {output_path}")