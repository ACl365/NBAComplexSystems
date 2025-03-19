# NBA Player Performance Dynamics

![NBA Analytics](https://i.imgur.com/AzpIwMR.png)

## Overview
A complex systems approach to NBA player performance analysis, revealing patterns invisible to conventional statistics.

This project applies dynamical systems theory, network analysis, and machine learning to extract deeper insights from basketball performance data, going beyond traditional box score statistics to understand the underlying patterns and dynamics of player performance.

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
│   ├── data_processing.py     # Data loading and preprocessing
│   ├── dynamics.py            # Dynamical systems analysis
│   ├── network_analysis.py    # Teammate network analysis
│   ├── visualization.py       # Enhanced visualizations
│   └── utils.py               # Utility functions
├── dashboard/                 # Interactive dashboard code
│   ├── app.py                 # Streamlit dashboard
│   └── components/            # Dashboard components
└── tests/                     # Unit tests
    └── test_dynamics.py       # Shows software engineering rigor
```

## Key Findings
1. Players can be classified into four stability quadrants with distinct performance characteristics:
   - **High-Value Stability**: Consistently positive impact players
   - **High-Ceiling Volatility**: Players with game-changing ability but inconsistent output
   - **Low-Impact Consistency**: Reliably average performers
   - **High-Risk Variability**: Unpredictable performers with negative tendencies

2. Team playing styles cluster into 4 distinct archetypes:
   - **Modern Pace-and-Space**: High tempo, three-point focused offense with spacing
   - **Traditional Inside-Out**: Post-oriented offense with methodical pace
   - **Defensive-Oriented**: Defense-first approach with opportunistic offense
   - **Balanced Attack**: Well-rounded approach without extreme tendencies

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

## Visualizations

### Player Stability Quadrant
![Player Stability Quadrant](https://i.imgur.com/AzpIwMR.png)

### Team Style Cards
![Team Style Cards](https://i.imgur.com/bPFYlm5.png)

### Player Impact Dashboard
![Player Impact Dashboard](https://i.imgur.com/AzpIwMR.png)

## Data Sources
This project uses NBA game and player data from public NBA statistics APIs.

## License
MIT

## Author
Your Name

## Acknowledgements
Thanks to the NBA for providing the data used in this analysis.