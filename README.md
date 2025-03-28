# Feedback-analyser

This script analyzes user feedback data to extract actionable insights and visualize trends.
It processes feedback from CSV files containing timestamps and text feedback entries.

Key Features:
- Handles large datasets through intelligent sampling
- Categorizes user problems using OpenAI's API
- Identifies trending issues across time periods
- Generates visualizations of most common problems
- Provides interactive exploration of analysis results
- Saves results to desktop as JSON and PNG files
- Allows selection of specific months to analyze

Usage:
- Set your OpenAI API key in the script
- Ensure your CSV has 'time' and 'feedback' columns (or similar)
- Run the script and follow the interactive prompts
- Results will be saved to your desktop automatically
