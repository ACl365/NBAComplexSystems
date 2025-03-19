import unittest
import numpy as np
import pandas as pd
import sys
import os

# Add the project root to the path so we can import our modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.dynamics import (
    calculate_var_model,
    calculate_stability_metrics
)

class TestDynamics(unittest.TestCase):
    """Test cases for the dynamics module"""
    
    def setUp(self):
        """Set up test data"""
        # Create sample player data
        dates = pd.date_range(start='2023-01-01', periods=20)
        
        # Player with stable performance
        self.stable_player_data = pd.DataFrame({
            'GAME_DATE': dates,
            'Player_ID': [1] * 20,
            'PlayerName': ['Stable Player'] * 20,
            'PTS': [20, 22, 19, 21, 20, 18, 22, 21, 19, 20, 21, 22, 20, 19, 21, 20, 22, 21, 20, 19],
            'PLUS_MINUS': [5, 6, 4, 5, 7, 5, 6, 4, 5, 6, 5, 7, 6, 5, 4, 6, 5, 7, 6, 5]
        })
        
        # Player with volatile performance
        self.volatile_player_data = pd.DataFrame({
            'GAME_DATE': dates,
            'Player_ID': [2] * 20,
            'PlayerName': ['Volatile Player'] * 20,
            'PTS': [10, 30, 5, 25, 40, 15, 35, 8, 28, 12, 32, 7, 27, 38, 14, 33, 9, 29, 13, 31],
            'PLUS_MINUS': [-5, 15, -10, 12, 20, -8, 18, -12, 14, -6, 16, -9, 13, 19, -7, 17, -11, 15, -8, 14]
        })
        
        # Combine players
        self.player_temporal_df = pd.concat([self.stable_player_data, self.volatile_player_data])
    
    def test_var_model_calculation(self):
        """Test VAR model calculation"""
        # Test stable player
        var_results, eigenvalues, lyapunov_exponents = calculate_var_model(self.stable_player_data)
        
        # Check if model was created
        self.assertIsNotNone(var_results)
        self.assertIsNotNone(eigenvalues)
        self.assertIsNotNone(lyapunov_exponents)
        
        # Check stability (eigenvalues should be small for stable player)
        self.assertTrue(np.max(np.abs(eigenvalues)) < 1.0)
        
        # Test volatile player
        var_results, eigenvalues, lyapunov_exponents = calculate_var_model(self.volatile_player_data)
        
        # Check if model was created
        self.assertIsNotNone(var_results)
        self.assertIsNotNone(eigenvalues)
        self.assertIsNotNone(lyapunov_exponents)
        
        # Check volatility (eigenvalues should be larger for volatile player)
        self.assertTrue(np.max(np.abs(eigenvalues)) > 0.5)
    
    def test_stability_metrics_calculation(self):
        """Test stability metrics calculation"""
        # Calculate stability metrics
        stability_df = calculate_stability_metrics(self.player_temporal_df)
        
        # Check if dataframe was created
        self.assertIsNotNone(stability_df)
        self.assertEqual(len(stability_df), 2)  # Should have 2 players
        
        # Get player metrics
        stable_metrics = stability_df[stability_df['player_name'] == 'Stable Player'].iloc[0]
        volatile_metrics = stability_df[stability_df['player_name'] == 'Volatile Player'].iloc[0]
        
        # Check stability scores
        self.assertLess(stable_metrics['system_stability'], volatile_metrics['system_stability'])
        
        # Check stability types
        self.assertIn('Stable', stable_metrics['stability_type'])
        self.assertIn('Volatile', volatile_metrics['stability_type'])
        
        # Check performance entropy (should be higher for volatile player)
        self.assertLess(stable_metrics['performance_entropy'], volatile_metrics['performance_entropy'])

if __name__ == '__main__':
    unittest.main()