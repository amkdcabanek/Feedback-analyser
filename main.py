"""
Feedback Analytics Agent
========================

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

Author: A Cabanek
Date: March 26, 2025
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from collections import Counter
from openai import OpenAI
import random

# Define desktop path explicitly
DESKTOP_PATH = os.path.join(os.path.expanduser("~"), "Desktop")
print(f"Files will be saved to: {DESKTOP_PATH}")

# API key setup
os.environ["OPENAI_API_KEY"] = ""  # Empty by default
api_key = input("Enter your OpenAI API key (press Enter to use environment variable if set): ").strip()
if api_key:
    os.environ["OPENAI_API_KEY"] = api_key
elif not os.environ.get("OPENAI_API_KEY"):
    print("Warning: No OpenAI API key provided. Please set OPENAI_API_KEY environment variable.")
client = OpenAI()

def clean_date(date_str):
    """
    Clean date strings that might be in range format like '2025.01.01 - 2025.01.03'
    by extracting just the first date
    """
    if not isinstance(date_str, str):
        return pd.NaT
        
    if ' - ' in date_str:
        # Extract the first date from the range
        date_str = date_str.split(' - ')[0]
    
    # Try to handle the YYYY.MM.DD format
    try:
        return pd.to_datetime(date_str, format='%Y.%m.%d')
    except:
        # Fall back to letting pandas guess the format
        try:
            return pd.to_datetime(date_str, errors='coerce')
        except:
            return pd.NaT

def get_csv_file_path():
    """Prompt the user for a CSV file path"""
    print("\n=== SELECT FEEDBACK DATA FILE ===")
    
    # Default path as a fallback
    default_path = "C:/Users/admin/Downloads/feature_feedback.csv"
    
    # Ask user for file path
    user_path = input(f"Enter the path to your CSV file [default: {default_path}]: ").strip()
    
    # Use the default if nothing is entered
    if not user_path:
        print(f"Using default path: {default_path}")
        return default_path
    
    # Check if the file exists
    if not os.path.exists(user_path):
        print(f"Warning: File not found at {user_path}")
        use_anyway = input("Would you like to try this path anyway? (y/n): ").strip().lower()
        if use_anyway != 'y':
            print(f"Using default path instead: {default_path}")
            return default_path
    
    print(f"Using file: {user_path}")
    return user_path

def get_available_months(feedback_data):
    """Extract available months from the feedback data"""
    try:
        df = pd.DataFrame(feedback_data)
        df['time'] = df['time'].apply(clean_date)
        df = df.dropna(subset=['time'])
        df['month_year'] = df['time'].dt.strftime('%Y-%m')
        
        # Get unique months sorted chronologically
        months = sorted(df['month_year'].unique())
        return months
    except Exception as e:
        print(f"Error extracting months: {e}")
        return []

def load_feedback_data(file_path):
    """Load feedback data from CSV file with time and feedback columns"""
    try:
        df = pd.read_csv(file_path)
        print(f"Successfully loaded {len(df)} feedback entries from {file_path}")
        
        # Print column names to debug
        print(f"CSV columns: {df.columns.tolist()}")
        
        # Ensure required columns exist
        if 'time' not in df.columns or 'feedback' not in df.columns:
            print("Warning: CSV is missing required 'time' or 'feedback' columns!")
            print("Available columns:", df.columns.tolist())
            
            # Try to guess which columns might contain time and feedback data
            time_candidates = [col for col in df.columns if 'time' in col.lower() or 'date' in col.lower()]
            feedback_candidates = [col for col in df.columns if 'feedback' in col.lower() or 'comment' in col.lower()]
            
            if time_candidates and feedback_candidates:
                print(f"Using {time_candidates[0]} as time column and {feedback_candidates[0]} as feedback column")
                df = df.rename(columns={time_candidates[0]: 'time', feedback_candidates[0]: 'feedback'})
            else:
                print("Could not find suitable columns for time and feedback data")
                return []
        
        # Convert to records format
        return df.to_dict('records')
    except Exception as e:
        print(f"Error loading feedback data: {e}")
        return []

def sample_feedback(feedback_data, sample_size=1000, method='random', selected_months=None):
    """
    Sample the feedback data to reduce processing time
    
    Parameters:
    - feedback_data: List of dictionaries with feedback data
    - sample_size: Maximum number of samples to return
    - method: 'random' or 'stratified' sampling
    - selected_months: List of specific months to include (format: YYYY-MM)
    """
    df = pd.DataFrame(feedback_data)
    
    # Clean and convert date strings using our custom function
    print("Cleaning date strings...")
    df['time'] = df['time'].apply(clean_date)
    
    # Drop rows with invalid dates
    df = df.dropna(subset=['time'])
    print(f"Found {len(df)} entries with valid dates")
    
    # Add month column for filtering and stratification
    df['month_year'] = df['time'].dt.strftime('%Y-%m')
    
    # Filter by selected months if provided
    if selected_months:
        print(f"Filtering to include only selected months: {', '.join(selected_months)}")
        df = df[df['month_year'].isin(selected_months)]
        print(f"After filtering: {len(df)} entries remain")
        
        if len(df) == 0:
            print("No entries found for the selected months!")
            return []
    
    if method == 'random':
        # Simple random sampling
        if len(df) > sample_size:
            print(f"Sampling {sample_size} entries randomly from {len(df)} total entries")
            df = df.sample(n=sample_size, random_state=42)
            
    elif method == 'stratified':
        # Stratified sampling by month
        months = df['month_year'].unique()
        
        # Calculate samples per month - REMOVED THE 200 SAMPLE LIMIT
        samples_per_month = sample_size // max(1, len(months))
        print(f"Performing stratified sampling across {len(months)} different months")
        print(f"Using approximately {samples_per_month} samples per month")
        
        sampled_dfs = []
        for month in months:
            month_data = df[df['month_year'] == month]
            if len(month_data) > samples_per_month:
                sampled_dfs.append(month_data.sample(n=samples_per_month, random_state=42))
            else:
                sampled_dfs.append(month_data)
        
        df = pd.concat(sampled_dfs)
        print(f"Sampled {len(df)} entries using stratified sampling")
    
    return df.to_dict('records')

def analyze_feedback_batch(feedback_batch):
    """Process a single batch of feedback using the OpenAI API"""
    prompt = f"""
    Analyze these user feedback entries and:
    1. Extract and categorize the main problems (max 35 categories)
    2. Count frequency of each problem category
    3. Identify the most pressing issues
    
    User feedback:
    {feedback_batch}
    
    Return the results in valid JSON format with these keys:
    - categories: {{category_name: frequency}}
    - top_issues: [list of most critical problems]
    - actionable_insights: [specific suggestions]
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a feedback analysis expert who extracts actionable insights. Always respond in valid JSON format."},
                {"role": "user", "content": prompt}
            ]
        )
        
        result = response.choices[0].message.content
        
        # Verify the response is valid JSON
        try:
            json.loads(result)
            return result
        except json.JSONDecodeError:
            # Try to extract JSON from response
            if result.find('{') >= 0 and result.rfind('}') >= 0:
                json_start = result.find('{')
                json_end = result.rfind('}') + 1
                json_str = result[json_start:json_end]
                try:
                    json.loads(json_str)
                    return json_str
                except:
                    return None
            return None
    except Exception as e:
        print(f"  Error processing batch: {e}")
        return None

def analyze_sentiment(feedback_batch):
    """Analyze sentiment of feedback entries using OpenAI API"""
    prompt = f"""
    Analyze these user feedback entries and classify each one as either POSITIVE or NEGATIVE.
    Return the results in valid JSON format as a list with the format:
    [
        {{
            "feedback": "original feedback text",
            "sentiment": "POSITIVE or NEGATIVE",
            "confidence": "score between 0 and 1"
        }},
        ...
    ]
    
    User feedback:
    {feedback_batch}
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a sentiment analysis expert. Always respond in valid JSON format."},
                {"role": "user", "content": prompt}
            ]
        )
        
        result = response.choices[0].message.content
        
        # Verify the response is valid JSON
        try:
            return json.loads(result)
        except json.JSONDecodeError:
            # Try to extract JSON from response
            if result.find('[') >= 0 and result.rfind(']') >= 0:
                json_start = result.find('[')
                json_end = result.rfind(']') + 1
                json_str = result[json_start:json_end]
                try:
                    return json.loads(json_str)
                except:
                    return []
            return []
    except Exception as e:
        print(f"  Error analyzing sentiment: {e}")
        return []

def filter_by_sentiment(feedback_data, sentiment_filter="all", batch_size=20):
    """
    Filter feedback data by sentiment (positive or negative)
    
    Parameters:
    - feedback_data: List of dictionaries with feedback data
    - sentiment_filter: 'positive', 'negative', or 'all'
    - batch_size: Size of batches for API processing
    """
    if sentiment_filter == "all":
        return feedback_data
    
    print(f"\nAnalyzing sentiment to filter {sentiment_filter} feedback...")
    df = pd.DataFrame(feedback_data)
    filtered_data = []
    
    # Process in batches to avoid token limits
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i+batch_size].copy()
        feedback_list = batch['feedback'].tolist()
        print(f"  Processing sentiment batch {i//batch_size + 1} with {len(feedback_list)} items")
        
        sentiment_results = analyze_sentiment(feedback_list)
        
        if sentiment_results:
            for j, result in enumerate(sentiment_results):
                if i+j < len(df) and result.get('sentiment', '').upper() == sentiment_filter.upper():
                    filtered_data.append(feedback_data[i+j])
    
    print(f"Filtered to {len(filtered_data)} {sentiment_filter} feedback entries")
    return filtered_data

def analyze_feedback(feedback_data, time_grouping='month', max_entries_per_group=100, batch_size=15):
    """
    Analyze a list of feedback entries and categorize them using sampling techniques
    
    Parameters:
    - feedback_data: List of dictionaries with 'time' and 'feedback' keys
    - time_grouping: 'week' or 'month' for grouping data
    - max_entries_per_group: Maximum entries to process per time period
    - batch_size: Size of batches for API processing
    """
    if not feedback_data:
        print("No feedback data provided!")
        return {}
    
    # Create DataFrame from the sampled data
    df = pd.DataFrame(feedback_data)
    
    # Add time period for grouping
    if time_grouping == 'week':
        df['time_group'] = df['time'].dt.isocalendar().week
        period_name = 'Week'
    else:  # month grouping
        df['time_group'] = df['time'].dt.strftime('%Y-%m')
        period_name = 'Month'
    
    # Group feedback by time period
    grouped_feedback = df.groupby('time_group')['feedback'].apply(list).to_dict()
    print(f"Found feedback for {len(grouped_feedback)} different {time_grouping}s")
    
    # Sample each group to reduce processing time
    for group, feedbacks in grouped_feedback.items():
        if len(feedbacks) > max_entries_per_group:
            random.seed(42)  # For reproducibility
            grouped_feedback[group] = random.sample(feedbacks, max_entries_per_group)
            print(f"  Sampled {max_entries_per_group} entries from {period_name} {group} (original: {len(feedbacks)})")
    
    results = {}
    for group, feedbacks in grouped_feedback.items():
        print(f"\nProcessing {period_name} {group} with {len(feedbacks)} feedback entries")
        
        # Process in small batches to avoid token limits
        all_batch_results = []
        
        for i in range(0, len(feedbacks), batch_size):
            batch = feedbacks[i:i+batch_size]
            print(f"  Processing batch {i//batch_size + 1} with {len(batch)} items")
            
            batch_result = analyze_feedback_batch(batch)
            if batch_result:
                all_batch_results.append(batch_result)
                print("  Batch processed successfully")
        
        # Merge batch results
        if len(all_batch_results) > 1:
            print(f"  Merging {len(all_batch_results)} batch results...")
            
            # Create a simpler merge prompt with fewer batch results to avoid token limits
            # If there are many batches, just use the first few
            batches_to_merge = all_batch_results[:5] if len(all_batch_results) > 5 else all_batch_results
            
            merge_prompt = f"""
            Merge these separate feedback analysis results into a consolidated analysis:
            
            {batches_to_merge}
            
            Return a consolidated analysis in valid JSON format with these keys:
            - categories: {{category_name: frequency}}
            - top_issues: [list of most critical problems]
            - actionable_insights: [specific suggestions]
            """
            
            try:
                merge_response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a feedback analysis expert who extracts actionable insights. Always respond in valid JSON format."},
                        {"role": "user", "content": merge_prompt}
                    ]
                )
                
                merged_result = merge_response.choices[0].message.content
                
                # Verify the response is valid JSON
                try:
                    json.loads(merged_result)
                    results[f"{period_name} {group}"] = merged_result
                    print("  Merged results successfully")
                except json.JSONDecodeError:
                    print("  Warning: Model returned invalid JSON for merge. Using first batch result instead.")
                    results[f"{period_name} {group}"] = all_batch_results[0]
                
            except Exception as e:
                print(f"  Error merging results: {e}")
                # If merging fails, just use the first batch result
                results[f"{period_name} {group}"] = all_batch_results[0] if all_batch_results else "{}"
        elif len(all_batch_results) == 1:
            # If we only have one batch, just use its result
            results[f"{period_name} {group}"] = all_batch_results[0]
            print("  Using single batch result")
        else:
            print(f"  No valid results for this {time_grouping}")
            results[f"{period_name} {group}"] = "{}"
    
    # Analyze trends if we have multiple periods
    if len(results) > 1:
        print(f"\nAnalyzing trends across {time_grouping}s...")
        
        # Prepare a simplified version of the data to avoid token limits
        simplified_data = []
        for period_name, result_json in list(results.items())[:5]:  # Limit to 5 periods for token considerations
            try:
                data = json.loads(result_json)
                # Just use top categories and issues to reduce size
                top_categories = dict(sorted(data.get('categories', {}).items(), key=lambda x: int(x[1]) if isinstance(x[1], (int, str)) else 0, reverse=True)[:10])
                simplified_data.append({
                    'period': period_name,
                    'top_categories': top_categories,
                    'top_issues': data.get('top_issues', [])[:5]
                })
            except:
                print(f"  Could not parse JSON for {period_name}")
        
        if simplified_data:
            trend_prompt = f"""
            Compare these feedback analyses across different time periods and identify:
            1. Trends in problem categories over time
            2. Emerging issues not present in earlier periods
            3. Resolved issues that are no longer appearing
            
            Data by period:
            {simplified_data}
            
            Return your analysis in valid JSON format with these keys:
            - trends: [descriptions of significant trends]
            - emerging_issues: [new problems]
            - resolved_issues: [problems that disappeared]
            """
            
            try:
                trend_response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a feedback analysis expert who extracts actionable insights. Always respond in valid JSON format."},
                        {"role": "user", "content": trend_prompt}
                    ]
                )
                
                trend_result = trend_response.choices[0].message.content
                
                # Verify the response is valid JSON
                try:
                    json.loads(trend_result)
                    results["Trend Analysis"] = trend_result
                    print("Trend analysis completed successfully")
                except json.JSONDecodeError:
                    print("Warning: Model returned invalid JSON for trend analysis")
            
            except Exception as e:
                print(f"Error analyzing trends: {e}")
    
    return results

def visualize_results(results, selected_period=None):
    """
    Generate visualizations of the feedback analysis using pie charts
    
    Parameters:
    - results: Dictionary of analysis results
    - selected_period: If provided, only visualize this specific period
    """
    print("\nGenerating visualizations...")
    
    # Parse the results to extract categories
    categories_by_period = {}
    for period, result_json in results.items():
        if period == "Trend Analysis":
            continue
            
        # If a specific period is selected, only include that one
        if selected_period and period != selected_period:
            continue
            
        try:
            data = json.loads(result_json)
            categories = data.get('categories', {})
            # Convert string numbers to integers
            categories = {k: int(v) if isinstance(v, str) and v.isdigit() else v for k, v in categories.items()}
            categories_by_period[period] = categories
        except:
            print(f"Could not parse JSON for {period}")
    
    if not categories_by_period:
        print("No valid data for visualization")
        return
    
    # Prepare the visualization
    periods = list(categories_by_period.keys())
    
    # Create a pie chart for each period's top categories
    plt.figure(figsize=(15, 5 * len(periods)))
    
    for i, (period, categories) in enumerate(categories_by_period.items()):
        plt.subplot(len(periods), 1, i+1, aspect='equal')
        
        # Get top 10 categories
        try:
            top_categories = dict(sorted(categories.items(), key=lambda x: x[1] if isinstance(x[1], (int)) else 0, reverse=True)[:10])
            
            # Colors for pie chart
            colors = plt.cm.tab20(range(len(top_categories)))
            
            # Create pie chart
            wedges, texts, autotexts = plt.pie(
                [v if isinstance(v, (int)) else 0 for v in top_categories.values()],
                labels=None,
                autopct='%1.1f%%',
                startangle=90,
                colors=colors
            )
            
            # Customize text appearance
            plt.setp(autotexts, size=10, weight='bold')
            
            # Add a legend outside the pie chart
            plt.legend(
                wedges,
                top_categories.keys(),
                title="Categories",
                loc="center left",
                bbox_to_anchor=(1, 0, 0.5, 1)
            )
            
            plt.title(f'Top Categories for {period}')
            plt.tight_layout()
        except Exception as e:
            print(f"Error creating visualization for {period}: {e}")
    
    plt.tight_layout(pad=3.0)
    
    # Name the file based on whether we're showing all periods or a specific one
    if selected_period:
        chart_filename = f'feedback_categories_{selected_period.replace(" ", "_")}.png'
    else:
        chart_filename = 'feedback_categories_by_period.png'
        
    # Save to desktop with explicit path
    categories_chart_path = os.path.join(DESKTOP_PATH, chart_filename)
    plt.savefig(categories_chart_path)
    print(f"Visualization saved as '{categories_chart_path}'")
    
    # Create a combined pie chart for overall top categories
    all_categories = {}
    for categories in categories_by_period.values():
        for category, count in categories.items():
            count_value = int(count) if isinstance(count, (int, str)) and str(count).isdigit() else 0
            all_categories[category] = all_categories.get(category, 0) + count_value
    
    # Get top 15 overall categories
    top_overall = dict(sorted(all_categories.items(), key=lambda x: x[1], reverse=True)[:15])
    
    plt.figure(figsize=(12, 8))
    
    # Colors for pie chart
    colors = plt.cm.tab20(range(len(top_overall)))
    
    # Create pie chart
    wedges, texts, autotexts = plt.pie(
        top_overall.values(),
        labels=None,
        autopct='%1.1f%%',
        startangle=90,
        colors=colors
    )
    
    # Customize text appearance
    plt.setp(autotexts, size=10, weight='bold')
    
    # Add a legend outside the pie chart
    plt.legend(
        wedges,
        top_overall.keys(),
        title="Categories",
        loc="center left",
        bbox_to_anchor=(1, 0, 0.5, 1)
    )
    
    if selected_period:
        plt.title(f'Top Categories for {selected_period}')
        chart_filename = f'top_categories_{selected_period.replace(" ", "_")}.png'
    else:
        plt.title('Top Overall Feedback Categories')
        chart_filename = 'top_overall_categories.png'
        
    plt.tight_layout()
    
    # Save to desktop with explicit path
    overall_chart_path = os.path.join(DESKTOP_PATH, chart_filename)
    plt.savefig(overall_chart_path)
    print(f"Overall visualization saved as '{overall_chart_path}'")

def get_user_preferences(available_months):
    """
    Get user preferences for analysis including month selection and sentiment filtering
    
    Parameters:
    - available_months: List of available months in the data (YYYY-MM format)
    """
    print("\n=== USER PREFERENCES ===")
    
    # Ask for sampling size
    try:
        sample_size = int(input("Enter sample size (1000-5000 recommended): ").strip() or "2000")
        if sample_size < 100:
            print("Sample size too small, using minimum of 100")
            sample_size = 100
        elif sample_size > 10000:
            print("Sample size too large, using maximum of 10000")
            sample_size = 10000
    except ValueError:
        print("Invalid input, using default sample size of 2000")
        sample_size = 2000
    
    # Ask for time grouping
    time_grouping = input("Group by month or week? (month/week) [default: month]: ").strip().lower()
    if time_grouping not in ['month', 'week']:
        print("Invalid option, using default (month)")
        time_grouping = 'month'
    
    # Ask for sentiment filtering
    sentiment_filter = input("Filter by sentiment? (all/positive/negative) [default: all]: ").strip().lower()
    if sentiment_filter not in ['all', 'positive', 'negative']:
        print("Invalid option, using default (all)")
        sentiment_filter = 'all'
    
    # Ask if they want to select specific months
    selected_months = []
    if available_months:
        filter_months = input("Do you want to analyze all months or select specific ones? (all/select) [default: all]: ").strip().lower()
        
        if filter_months == 'select':
            print("\nAvailable months:")
            for i, month in enumerate(available_months):
                print(f"{i+1}. {month}")
            
            # Let user select specific months
            month_input = input("\nEnter month numbers separated by commas (e.g., 1,3,5): ").strip()
            if month_input:
                try:
                    month_indices = [int(idx.strip()) - 1 for idx in month_input.split(',')]
                    for idx in month_indices:
                        if 0 <= idx < len(available_months):
                            selected_months.append(available_months[idx])
                    
                    if selected_months:
                        print(f"Selected months: {', '.join(selected_months)}")
                    else:
                        print("No valid months selected, using all months")
                except ValueError:
                    print("Invalid input format, using all months")
    
    return {
        'sample_size': sample_size,
        'time_grouping': time_grouping,
        'sentiment_filter': sentiment_filter,
        'selected_months': selected_months
    }

def display_interactive_menu(analysis_results):
    """Display an interactive menu for exploring the analysis results"""
    while True:
        # Get available periods (excluding Trend Analysis)
        available_periods = [p for p in analysis_results.keys() if p != "Trend Analysis"]
        
        print("\n=== FEEDBACK ANALYSIS MENU ===")
        print("1. View summary of all periods")
        print("2. View specific period details")
        print("3. View trend analysis")
        print("4. Visualize specific period")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == '1':
            # Show summary of all periods
            print("\n=== SUMMARY OF ALL PERIODS ===")
            for period in available_periods:
                try:
                    data = json.loads(analysis_results[period])
                    print(f"\n----- {period} -----")
                    if 'top_issues' in data:
                        print("Top Issues:")
                        for i, issue in enumerate(data['top_issues'][:3]):
                            print(f"  {i+1}. {issue}")
                except:
                    print(f"Could not parse results for {period}")
        
        elif choice == '2':
            # Show specific period details
            if not available_periods:
                print("No period data available")
                continue
                
            print("\nAvailable periods:")
            for i, period in enumerate(available_periods):
                print(f"{i+1}. {period}")
                
            try:
                period_idx = int(input("\nSelect period (number): ").strip()) - 1
                if 0 <= period_idx < len(available_periods):
                    selected_period = available_periods[period_idx]
                    try:
                        data = json.loads(analysis_results[selected_period])
                        print(f"\n=== DETAILS FOR {selected_period} ===")
                        
                        if 'top_issues' in data:
                            print("\nTop Issues:")
                            for i, issue in enumerate(data['top_issues']):
                                print(f"  {i+1}. {issue}")
                                
                        if 'categories' in data:
                            print("\nTop Categories:")
                            top_cats = dict(sorted(data['categories'].items(), key=lambda x: int(x[1]) if isinstance(x[1], (int, str)) and str(x[1]).isdigit() else 0, reverse=True)[:10])
                            for cat, count in top_cats.items():
                                print(f"  - {cat}: {count}")
                                
                        if 'actionable_insights' in data:
                            print("\nActionable Insights:")
                            for i, insight in enumerate(data['actionable_insights']):
                                print(f"  {i+1}. {insight}")
                    except:
                        print(f"Could not parse results for {selected_period}")
                else:
                    print("Invalid selection")
            except ValueError:
                print("Invalid input")
        
        elif choice == '3':
            # Show trend analysis
            if "Trend Analysis" in analysis_results:
                try:
                    data = json.loads(analysis_results["Trend Analysis"])
                    print("\n=== TREND ANALYSIS ===")
                    
                    if 'trends' in data:
                        print("\nTrends:")
                        for i, trend in enumerate(data['trends']):
                            print(f"  {i+1}. {trend}")
                            
                    if 'emerging_issues' in data:
                        print("\nEmerging Issues:")
                        for i, issue in enumerate(data['emerging_issues']):
                            print(f"  {i+1}. {issue}")
                            
                    if 'resolved_issues' in data:
                        print("\nResolved Issues:")
                        for i, issue in enumerate(data['resolved_issues']):
                            print(f"  {i+1}. {issue}")
                except:
                    print("Could not parse trend analysis results")
            else:
                print("No trend analysis available")
        
        elif choice == '4':
            # Visualize specific period
            if not available_periods:
                print("No period data available")
                continue
                
            print("\nAvailable periods:")
            for i, period in enumerate(available_periods):
                print(f"{i+1}. {period}")
                
            try:
                period_idx = int(input("\nSelect period to visualize (number): ").strip()) - 1
                if 0 <= period_idx < len(available_periods):
                    selected_period = available_periods[period_idx]
                    print(f"Generating visualization for {selected_period}...")
                    
                    # Create a filtered results dictionary with just the selected period
                    filtered_results = {
                        selected_period: analysis_results[selected_period]
                    }
                    
                    # Generate visualization for this period only
                    visualize_results(filtered_results, selected_period)
                    print(f"Visualization for {selected_period} has been saved to your desktop")
                else:
                    print("Invalid selection")
            except ValueError:
                print("Invalid input")
                
        elif choice == '5':
            print("Exiting menu. Analysis results and visualizations are saved on your desktop.")
            break
            
        else:
            print("Invalid choice. Please enter a number from 1 to 5.")
        
        input("\nPress Enter to continue...")

def main():
    """
    Run the optimized feedback analysis workflow with user interaction
    """
    # Get CSV file path from user
    csv_path = get_csv_file_path()
    print(f"Starting feedback analysis for {csv_path}")
    
    # 1. Load the feedback data
    feedback_data = load_feedback_data(csv_path)
    
    if not feedback_data:
        print("No feedback data to analyze. Exiting.")
        return
    
    # Extract available months for user selection
    available_months = get_available_months(feedback_data)
    print(f"\nAvailable months in the data: {len(available_months)}")
    
    # Get user preferences including month selection and sentiment filtering
    prefs = get_user_preferences(available_months)
    sample_size = prefs['sample_size']
    time_grouping = prefs['time_grouping']
    sentiment_filter = prefs['sentiment_filter']
    selected_months = prefs['selected_months']
    
    print(f"Using sample size: {sample_size}, Time grouping: {time_grouping}")
    if selected_months:
        print(f"Analyzing only selected months: {', '.join(selected_months)}")
    else:
        print("Analyzing all available months")
    
    if sentiment_filter != 'all':
        print(f"Filtering feedback by sentiment: {sentiment_filter}")
    
    # 2. Sample the data for faster processing
    sampled_data = sample_feedback(
        feedback_data, 
        sample_size=sample_size, 
        method='stratified',
        selected_months=selected_months
    )
    
    if not sampled_data:
        print("Sampling produced no valid data. Exiting.")
        return
    
    # 3. Apply sentiment filtering if requested
    if sentiment_filter != 'all':
        sampled_data = filter_by_sentiment(sampled_data, sentiment_filter)
        if not sampled_data:
            print("No feedback entries match the sentiment filter. Exiting.")
            return
    
    # Calculate max entries per group based on sample size
    if time_grouping == 'month':
        # Estimate 12 months in a year, or use actual count if filtering
        num_months = len(selected_months) if selected_months else 12
        max_entries = max(50, sample_size // max(1, num_months))
    else:  # week
        # Estimate 52 weeks in a year
        max_entries = max(20, sample_size // 52)
    
    print(f"Processing up to {max_entries} entries per {time_grouping}")
    
    # 4. Analyze the sampled feedback with the calculated max entries per group
    analysis_results = analyze_feedback(
        sampled_data, 
        time_grouping=time_grouping,
        max_entries_per_group=max_entries,
        batch_size=15
    )
    
    if not analysis_results:
        print("Analysis produced no results. Exiting.")
        return
    
    # 5. Save results to file on desktop with explicit path
    results_file_path = os.path.join(DESKTOP_PATH, 'feedback_analysis_results.json')
    with open(results_file_path, 'w') as f:
        json.dump(analysis_results, f, indent=2)
    
    print(f"Analysis results saved to '{results_file_path}'")
    
    # 6. Generate visualizations for all periods
    visualize_results(analysis_results)
    
    # 7. Display interactive menu for exploring results
    display_interactive_menu(analysis_results)
    
    print("\nAnalysis complete! Files saved to your desktop:")
    print(f"1. Results: {results_file_path}")
    print(f"2. Period chart: {os.path.join(DESKTOP_PATH, 'feedback_categories_by_period.png')}")
    print(f"3. Overall chart: {os.path.join(DESKTOP_PATH, 'top_overall_categories.png')}")

# Entry point for the script
if __name__ == "__main__":
    main()
