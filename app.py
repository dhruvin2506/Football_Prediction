import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import joblib
from src.data.data_collection import FootballDataCollector
import numpy as np

def create_form_data(team_matches: pd.DataFrame, selected_team: str) -> pd.DataFrame:
    """Create form data for a team"""
    form_data = []
    for _, match in team_matches.iterrows():
        if match['home_team'] == selected_team:
            result = 'Win' if match['home_score'] > match['away_score'] else 'Draw' if match['home_score'] == match['away_score'] else 'Loss'
            score = f"{match['home_score']}-{match['away_score']}"
            opponent = match['away_team']
        else:
            result = 'Win' if match['away_score'] > match['home_score'] else 'Draw' if match['away_score'] == match['home_score'] else 'Loss'
            score = f"{match['away_score']}-{match['home_score']}"
            opponent = match['home_team']
        
        form_data.append({
            'Date': match['date'],
            'Result': result,
            'Score': score,
            'Opponent': opponent
        })
    
    return pd.DataFrame(form_data)

def create_form_visualization(form_data: pd.DataFrame, selected_team: str) -> go.Figure:
    """Create form visualization"""
    fig = px.bar(
        form_data,
        x='Date',
        color='Result',
        title=f"{selected_team} Form Timeline",
        color_discrete_map={'Win': '#2ecc71', 'Draw': '#3498db', 'Loss': '#e74c3c'},
        height=500  # Set fixed height
    )
    fig = styled_plot(fig)
    fig.update_layout(
        xaxis=dict(
            title="Match Date",
            tickangle=-45
        ),
        yaxis=dict(
            title="Result",
            showticklabels=False
        )
    )
    return fig

def display_model_performance(model):
    """Display model performance metrics and information"""
    st.markdown("""
        <div class="custom-card">
            <h2 style="color: #3498db;">Model Performance Metrics</h2>
            <p style="opacity: 0.8;">Evaluation metrics and model insights</p>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if os.path.exists('models/plots/confusion_matrix.png'):
            st.image('models/plots/confusion_matrix.png', caption='Confusion Matrix')
    
    with col2:
        if os.path.exists('models/plots/feature_importance.png'):
            st.image('models/plots/feature_importance.png', caption='Feature Importance')
    
    if model is not None:
        st.markdown("""
            <div class="custom-card">
                <h3 style="color: #3498db;">Model Information</h3>
            </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            styled_metric("Model Type", type(model).__name__, "Current model architecture")
        with col2:
            if hasattr(model, 'n_estimators'):
                styled_metric("Number of Trees", str(model.n_estimators), "Model complexity")

# Load custom CSS
def load_css():
    with open('style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Page config with custom theme
st.set_page_config(
    page_title="⚽ Football Predictor Pro",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load CSS
load_css()

# Custom styled header
def custom_header():
    st.markdown(
        """
        <div class="main-header">
            <h1>⚽ Football Predictor Pro</h1>
            <p style="font-size: 1.2rem; opacity: 0.8;">Premier League Match Prediction & Analysis Platform</p>
        </div>
        """, 
        unsafe_allow_html=True
    )

def styled_metric(label, value, help_text=None):
    """Create a styled metric display"""
    st.markdown(f"""
        <div class="custom-card" style="text-align: center;">
            <h3 style="color: #3498db; margin-bottom: 0.5rem;">{label}</h3>
            <h2 style="font-size: 2rem; margin: 0.5rem 0;">{value}</h2>
            {f'<p style="font-size: 0.8rem; opacity: 0.8;">{help_text}</p>' if help_text else ''}
        </div>
    """, unsafe_allow_html=True)

def styled_plot(fig):
    """Apply consistent styling to plotly figures"""
    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_family="Poppins",
        font_color="#ffffff",
        title_font_size=24,
        title_x=0.5,
        height=500,  # Set fixed height
        width=None,  # Allow width to be responsive
        margin=dict(t=100, l=50, r=50, b=50),
        showlegend=True,
        legend=dict(
            bgcolor="rgba(255,255,255,0.05)",
            bordercolor="rgba(255,255,255,0.1)"
        )
    )
    return fig

def load_model():
    """Load the trained model"""
    model_path = 'models/football_predictor_latest.pkl'
    scaler_path = 'models/football_predictor_latest_scaler.pkl'
    
    if not os.path.exists(model_path):
        st.error(f"Model file not found at: {model_path}")
        return None, None
    if not os.path.exists(scaler_path):
        st.error(f"Scaler file not found at: {scaler_path}")
        return None, None
        
    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler
    except Exception as e:
        st.error(f"Error loading model and scaler: {str(e)}")
        return None, None

def process_teams_data(home_team, away_team, historical_data):
    """Process team data to create features for prediction"""
    # Convert date column to datetime if it's not already
    historical_data['date'] = pd.to_datetime(historical_data['date'])
    
    # Sort data by date
    historical_data = historical_data.sort_values('date')
    
    # Calculate basic match features
    features = {}
    
    # Form features
    for team_type in ['home', 'away']:
        team = home_team if team_type == 'home' else away_team
        team_matches = historical_data[
            (historical_data['home_team'] == team) | 
            (historical_data['away_team'] == team)
        ].tail(5)
        
        if len(team_matches) > 0:
            # Goals scored and conceded
            features[f'{team_type}_team_last_5_goals'] = team_matches[f'{team_type}_score'].mean()
            features[f'{team_type}_team_last_5_conceded'] = team_matches[f'{"away" if team_type == "home" else "home"}_score'].mean()
            
            # Win rate
            wins = sum(team_matches[f'{team_type}_win'] == True)
            features[f'{team_type}_team_last_5_winrate'] = wins / len(team_matches)
            
            # Streak (last 3 matches)
            recent_matches = team_matches.tail(3)
            recent_wins = sum(recent_matches[f'{team_type}_win'] == True)
            features[f'{team_type}_team_streak'] = recent_wins
            
            # Clean sheets
            clean_sheets = sum(team_matches[f'{"away" if team_type == "home" else "home"}_score'] == 0)
            features[f'{team_type}_clean_sheet_rate'] = clean_sheets / len(team_matches)
            
            # Goals per game
            features[f'{team_type}_goals_per_game'] = team_matches[f'{team_type}_score'].mean()
            
            # Points
            points = sum(3 if win else 1 if pd.isna(win) else 0 
                        for win in team_matches[f'{team_type}_win'])
            features[f'{team_type}_points_per_game'] = points / len(team_matches)
            
            # Goal difference
            features[f'{team_type}_goal_difference'] = (
                team_matches[f'{team_type}_score'].sum() - 
                team_matches[f'{"away" if team_type == "home" else "home"}_score'].sum()
            )
        else:
            # Default values if no matches found
            features[f'{team_type}_team_last_5_goals'] = 0
            features[f'{team_type}_team_last_5_conceded'] = 0
            features[f'{team_type}_team_last_5_winrate'] = 0
            features[f'{team_type}_team_streak'] = 0
            features[f'{team_type}_clean_sheet_rate'] = 0
            features[f'{team_type}_goals_per_game'] = 0
            features[f'{team_type}_points_per_game'] = 0
            features[f'{team_type}_goal_difference'] = 0
        
        # League position (approximation)
        features[f'{team_type}_league_position'] = len(historical_data[f'{team_type}_team'].unique()) // 2

    # Head-to-head features
    h2h_matches = historical_data[
        ((historical_data['home_team'] == home_team) & (historical_data['away_team'] == away_team)) |
        ((historical_data['home_team'] == away_team) & (historical_data['away_team'] == home_team))
    ]
    
    if len(h2h_matches) > 0:
        features.update({
            'h2h_home_wins': sum(h2h_matches['home_score'] > h2h_matches['away_score']),
            'h2h_away_wins': sum(h2h_matches['home_score'] < h2h_matches['away_score']),
            'h2h_draws': sum(h2h_matches['home_score'] == h2h_matches['away_score']),
            'h2h_total_goals': h2h_matches['home_score'].sum() + h2h_matches['away_score'].sum(),
            'h2h_avg_goals': (h2h_matches['home_score'].sum() + h2h_matches['away_score'].sum()) / len(h2h_matches)
        })
    else:
        features.update({
            'h2h_home_wins': 0,
            'h2h_away_wins': 0,
            'h2h_draws': 0,
            'h2h_total_goals': 0,
            'h2h_avg_goals': 0
        })

    # Time-based features
    current_date = pd.Timestamp.now()
    features.update({
        'is_weekend': 1 if current_date.weekday() >= 5 else 0,
        'month': current_date.month,
        'is_holiday_period': 1 if current_date.month in [12, 1] else 0,
        'match_number_in_season': len(historical_data)
    })
    
    # Create DataFrame with features in correct order
    feature_order = [
        'home_team_last_5_goals', 'home_team_last_5_conceded',
        'away_team_last_5_goals', 'away_team_last_5_conceded',
        'home_team_last_5_winrate', 'away_team_last_5_winrate',
        'home_team_streak', 'away_team_streak',
        'home_clean_sheet_rate', 'away_clean_sheet_rate',
        'home_goals_per_game', 'away_goals_per_game',
        'home_points_per_game', 'away_points_per_game',
        'h2h_home_wins', 'h2h_away_wins', 'h2h_draws',
        'h2h_avg_goals', 'h2h_total_goals',
        'home_goal_difference', 'away_goal_difference',
        'home_league_position', 'away_league_position',
        'is_weekend', 'month', 'is_holiday_period',
        'match_number_in_season'
    ]
    
    return pd.DataFrame([{col: features.get(col, 0) for col in feature_order}])
def main():
    custom_header()
    
    # Load model and data
    model, scaler = load_model()
    
    # Load historical data
    try:
        historical_data = pd.read_csv('data/raw/matches_league_39_2023_20231218.csv')
        historical_data['date'] = pd.to_datetime(historical_data['date'])
    except Exception as e:
        st.error(f"Error loading historical data: {str(e)}")
        return
    
    # Get available teams once
    available_teams = sorted(historical_data['home_team'].unique())
    
    # Styled sidebar
    with st.sidebar:
        st.markdown("""
            <div style="text-align: center; padding: 1rem;">
                <h2 style="color: #3498db;">Navigation</h2>
            </div>
        """, unsafe_allow_html=True)
        page = st.radio("", ["Match Predictions", "Historical Analysis", "Model Performance"])

    if page == "Match Predictions":
        st.markdown("""
            <div class="custom-card">
                <h2 style="color: #3498db; margin-bottom: 1rem;">Match Prediction Engine</h2>
            </div>
        """, unsafe_allow_html=True)
        
        # Team selection with styled columns
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="custom-card">', unsafe_allow_html=True)
            home_team = st.selectbox("Home Team 🏠", available_teams)
            st.markdown('</div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="custom-card">', unsafe_allow_html=True)
            away_teams = [team for team in available_teams if team != home_team]
            away_team = st.selectbox("Away Team ✈️", away_teams)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Styled prediction button
        st.markdown('<div style="padding: 2rem 0;">', unsafe_allow_html=True)
        predict_btn = st.button("🔮 Predict Match Outcome")
        st.markdown('</div>', unsafe_allow_html=True)

        if predict_btn:
            with st.spinner("🎲 Analyzing match data..."):
                try:
                    # Process data and make prediction
                    features = process_teams_data(home_team, away_team, historical_data)
                    features_scaled = scaler.transform(features)
                    prediction_proba = model.predict_proba(features_scaled)[0]
                    
                    # Success message with animation
                    st.markdown("""
                        <div class="custom-card" style="text-align: center; background: linear-gradient(45deg, #2ecc71, #3498db);">
                            <h2 style="color: white;">Prediction Complete! 🎯</h2>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Create styled columns for results
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        styled_metric("Home Win 🏆", f"{prediction_proba[2]:.1%}", f"{home_team}")
                    with col2:
                        styled_metric("Draw 🤝", f"{prediction_proba[1]:.1%}", "Equal chances")
                    with col3:
                        styled_metric("Away Win 🏆", f"{prediction_proba[0]:.1%}", f"{away_team}")
                    
                    # Styled visualization
                    fig = go.Figure(data=[go.Bar(
                        x=['Home Win', 'Draw', 'Away Win'],
                        y=[prediction_proba[2], prediction_proba[1], prediction_proba[0]],
                        text=[f"{x:.1%}" for x in [prediction_proba[2], prediction_proba[1], prediction_proba[0]]],
                        textposition='auto',
                        marker_color=['#2ecc71', '#3498db', '#e74c3c']
                    )])
                    fig = styled_plot(fig)
                    fig.update_layout(
                        title=f"{home_team} vs {away_team} - Prediction Probabilities",
                        yaxis=dict(
                            title="Probability",
                            tickformat='.1%',
                            range=[0, 1]  # Fix y-axis range from 0 to 1
                        ),
                        xaxis=dict(
                            title=""  # Remove x-axis title
                        )
                    )
                    st.markdown('<div class="plot-card">', unsafe_allow_html=True)
                    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Styled feature display
                    if st.checkbox("📊 Show Detailed Analysis"):
                        st.markdown("""
                            <div class="custom-card">
                                <h3 style="color: #3498db;">Feature Analysis</h3>
                            </div>
                        """, unsafe_allow_html=True)
                        feature_importance = pd.DataFrame({
                            'Feature': features.columns,
                            'Value': features.iloc[0]
                        })
                        st.dataframe(feature_importance, use_container_width=True)
                
                except Exception as e:
                    st.error(f"Error making prediction: {str(e)}")

    elif page == "Historical Analysis":
        st.markdown("""
            <div class="custom-card">
                <h2 style="color: #3498db;">Historical Match Analysis</h2>
                <p style="opacity: 0.8;">Comprehensive analysis of past matches and team performance</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Basic statistics with styled metrics
        total_matches = len(historical_data)
        home_wins = sum(historical_data['home_score'] > historical_data['away_score'])
        away_wins = sum(historical_data['home_score'] < historical_data['away_score'])
        draws = sum(historical_data['home_score'] == historical_data['away_score'])
        
        st.markdown('<div style="padding: 1rem 0;">', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            styled_metric(
                "Home Wins 🏠", 
                f"{home_wins} ({home_wins/total_matches:.1%})",
                "Matches won by home teams"
            )
        with col2:
            styled_metric(
                "Draws 🤝", 
                f"{draws} ({draws/total_matches:.1%})",
                "Matches ending in draw"
            )
        with col3:
            styled_metric(
                "Away Wins ✈️", 
                f"{away_wins} ({away_wins/total_matches:.1%})",
                "Matches won by away teams"
            )
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Team performance section
        st.markdown("""
            <div class="custom-card">
                <h3 style="color: #3498db;">Team Performance Analysis</h3>
            </div>
        """, unsafe_allow_html=True)
        
        selected_team = st.selectbox("Select Team to Analyze 🎯", available_teams)
        
        team_matches = historical_data[
            (historical_data['home_team'] == selected_team) |
            (historical_data['away_team'] == selected_team)
        ]
        
        if len(team_matches) > 0:
            team_wins = sum(
                ((team_matches['home_team'] == selected_team) & (team_matches['home_score'] > team_matches['away_score'])) |
                ((team_matches['away_team'] == selected_team) & (team_matches['away_score'] > team_matches['home_score']))
            )
            team_matches_played = len(team_matches)
            
            styled_metric(
                f"{selected_team} Performance 📊",
                f"{team_wins} / {team_matches_played} ({team_wins/team_matches_played:.1%})",
                "Win rate across all matches"
            )
            
            # Form data
            form_data = create_form_data(team_matches, selected_team)
            
            # Display recent form
            st.markdown("""
                <div class="custom-card">
                    <h3 style="color: #3498db;">Recent Form</h3>
                </div>
            """, unsafe_allow_html=True)
            
            st.dataframe(form_data.head(), use_container_width=True)
            
            # Form visualization
            fig = create_form_visualization(form_data, selected_team)
            st.markdown('<div class="plot-card">', unsafe_allow_html=True)
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
            st.markdown('</div>', unsafe_allow_html=True)

    elif page == "Model Performance":
        display_model_performance(model)

if __name__ == "__main__":
    main()
