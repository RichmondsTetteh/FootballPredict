import streamlit as st 
import pandas as pd
import numpy as np
import joblib

# Add to imports (top of Streamlit section)
from sklearn.preprocessing import LabelEncoder  # For fallback
import warnings  # Suppress numpy warnings if needed
warnings.filterwarnings("ignore", message="Mean of empty slice")

# Updated load_resources (drop scalers; add LabelEncoder)
@st.cache_resource
def load_resources():
    try:
        outcome_model = joblib.load(OUTCOME_MODEL_FILENAME)
        goals_model = joblib.load(GOALS_MODEL_FILENAME)
        try:
            label_encoder = joblib.load(LABEL_ENCODER_FILENAME)
        except FileNotFoundError:
            label_encoder = LabelEncoder()
            label_encoder.fit(['A', 'D', 'H'])  # From FTResult: Away=0, Draw=1, Home=2
        return outcome_model, goals_model, label_encoder
    except FileNotFoundError as e:
        st.error(f"Error: {e}")
        return None, None, None

outcome_model, goals_model, label_encoder = load_resources()

# Base columns for user input (exclude derived: EloDiff, ratios, probs, eff, attack, ExpectedGoals)
base_columns = [
    'HomeElo', 'AwayElo', 'Form3Home', 'Form5Home', 'Form3Away', 'Form5Away',
    'HomeShots', 'AwayShots', 'HomeTarget', 'AwayTarget',
    'OddHome', 'OddDraw', 'OddAway', 'HandiSize', 'Over25', 'Under25',
    'C_LTH', 'C_LTA', 'C_VHD', 'C_VAD', 'C_HTB', 'C_PHB'
]
# feature_columns unchanged (28 total; derived auto-added)

# Updated clean_input_data (match notebook: median fill)
def clean_input_data(df):
    df = df.replace([np.inf, -np.inf], np.nan)
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        df[col] = df[col].fillna(df[col].median() if len(df) > 0 else 0.0)
    return df

# Updated create_input_features (exact notebook match; user provides ExpectedGoals as base now)
def create_input_features(df):
    df = df.copy()
    # Elo (bases present)
    df['EloDiff'] = df['HomeElo'] - df['AwayElo']
    df['EloSum'] = df['HomeElo'] + df['AwayElo']
    
    # Form ratios (exact: +0.1)
    df['Form3Ratio'] = df['Form3Home'] / (df['Form3Away'] + 0.1)
    df['Form5Ratio'] = df['Form5Home'] / (df['Form5Away'] + 0.1)
    
    # ExpectedGoals: User input (no history in app; add to base_columns if needed)
    if 'HomeExpectedGoals' not in df.columns:
        df['HomeExpectedGoals'] = 1.5
    if 'AwayExpectedGoals' not in df.columns:
        df['AwayExpectedGoals'] = 1.2
    
    # Probs (exact: always 1/odd; safeguard >0)
    df['HomeWinProbability'] = np.where(df['OddHome'] > 0, 1 / df['OddHome'], 0.5)
    df['DrawProbability'] = np.where(df['OddDraw'] > 0, 1 / df['OddDraw'], 0.25)
    df['AwayWinProbability'] = np.where(df['OddAway'] > 0, 1 / df['OddAway'], 0.25)
    
    # Shot eff (exact: +0.1)
    df['HomeShotEfficiency'] = df['HomeTarget'] / (df['HomeShots'] + 0.1)
    df['AwayShotEfficiency'] = df['AwayTarget'] / (df['AwayShots'] + 0.1)
    
    # Attack (exact)
    df['HomeAttackStrength'] = df['HomeExpectedGoals'] * df['Form5Home'] / 15
    df['AwayAttackStrength'] = df['AwayExpectedGoals'] * df['Form5Away'] / 15
    
    return df

# Simplified preprocess (no scalers; single output)
def preprocess_input_data(input_df, feature_columns):
    # Add missing bases with realistic defaults
    defaults = {
        'HomeShots': 12.0, 'AwayShots': 10.0, 'HomeTarget': 4.0, 'AwayTarget': 3.0,
        'OddHome': 2.0, 'OddDraw': 3.0, 'OddAway': 3.0,  # Realistic → probs ~0.5/0.33
        'HomeExpectedGoals': 1.5, 'AwayExpectedGoals': 1.2,  # Add as bases
        # Extend for others (e.g., Form=0, Cluster=0.2)
    }
    for col in base_columns + ['HomeExpectedGoals', 'AwayExpectedGoals']:
        if col not in input_df.columns:
            input_df[col] = defaults.get(col, 0.0)
    
    cleaned_df = clean_input_data(input_df)
    featured_df = create_input_features(cleaned_df)
    
    try:
        processed_features = featured_df[feature_columns].fillna(0)  # Final safety
        return processed_features
    except KeyError as e:
        st.error(f"Missing column: {e}")
        return None

# In sidebar: Inputs for bases + ExpectedGoals (extend your if-elif chain)
st.sidebar.subheader("Match Statistics")
feature_input_values = {}
for col in base_columns + ['HomeExpectedGoals', 'AwayExpectedGoals']:
    label = col
    default_value = 0.0
    step_size = 0.01
    min_value = None
    max_value = None

    if col == 'HomeElo':
        label = "Home Elo Rating"
        default_value = 1500.0
        step_size = 1.0
    elif col == 'AwayElo':
        label = "Away Elo Rating"
        default_value = 1500.0
        step_size = 1.0
    elif col == 'Form3Home':
        label = "Home Form (last 3 matches)"
        default_value = 0
        min_value = 0
        max_value = 9
        step_size = 1
    elif col == 'Form5Home':
        label = "Home Form (last 5 matches)"
        default_value = 0
        min_value = 0
        max_value = 15
        step_size = 1
    elif col == 'Form3Away':
        label = "Away Form (last 3 matches)"
        default_value = 0
        min_value = 0
        max_value = 9
        step_size = 1
    elif col == 'Form5Away':
        label = "Away Form (last 5 matches)"
        default_value = 0
        min_value = 0
        max_value = 15
        step_size = 1
    elif col == 'HomeShots':
        label = "Home Shots (expected)"
        default_value = 12.0
        min_value = 0
        max_value = 30
        step_size = 1
    elif col == 'AwayShots':
        label = "Away Shots (expected)"
        default_value = 10.0
        min_value = 0
        max_value = 30
        step_size = 1
    elif col == 'HomeTarget':
        label = "Home Shots on Target (expected)"
        default_value = 4.0
        min_value = 0
        max_value = 15
        step_size = 1
    elif col == 'AwayTarget':
        label = "Away Shots on Target (expected)"
        default_value = 3.0
        min_value = 0
        max_value = 15
        step_size = 1
    elif col == 'OddHome':
        label = "Home Win Odds"
        default_value = 2.0
        min_value = 1.0
        step_size = 0.1
    elif col == 'OddDraw':
        label = "Draw Odds"
        default_value = 3.0
        min_value = 1.0
        step_size = 0.1
    elif col == 'OddAway':
        label = "Away Win Odds"
        default_value = 3.0
        min_value = 1.0
        step_size = 0.1
    elif col == 'HandiSize':
        label = "Handicap Size"
        default_value = 0.0
        min_value = -2.0
        max_value = 2.0
        step_size = 0.25
    elif col == 'Over25':
        label = "Over 2.5 Goals Odds"
        default_value = 2.0
        min_value = 1.0
        step_size = 0.1
    elif col == 'Under25':
        label = "Under 2.5 Goals Odds"
        default_value = 2.0
        min_value = 1.0
        step_size = 0.1
    elif col == 'HomeExpectedGoals':
        label = "Home Expected Goals"
        default_value = 1.5
        min_value = 0.0
        max_value = 5.0
        step_size = 0.1
    elif col == 'AwayExpectedGoals':
        label = "Away Expected Goals"
        default_value = 1.2
        min_value = 0.0
        max_value = 5.0
        step_size = 0.1
    elif col in ['C_LTH', 'C_LTA', 'C_VHD', 'C_VAD', 'C_HTB', 'C_PHB']:
        label = f"Cluster Feature: {col}"
        default_value = 0.0
        min_value = 0.0
        max_value = 1.0
        step_size = 0.01

    # Create the number input widget
    feature_input_values[col] = st.sidebar.number_input(
        label=label,
        value=default_value,
        step=step_size,
        min_value=min_value,
        max_value=max_value
    )

# On button:
if st.sidebar.button("Predict"):
    input_data = {'HomeTeam': input_values['HomeTeam'], 'AwayTeam': input_values['AwayTeam']}
    input_data.update(feature_input_values)
    input_df_raw = pd.DataFrame([input_data])
    
    processed_features = preprocess_input_data(input_df_raw, feature_columns)
    
    if processed_features is not None and outcome_model and goals_model:
        outcome_encoded = outcome_model.predict(processed_features)
        goals_pred = goals_model.predict(processed_features)[0]
        
        # Dynamic decoding (robust to class order)
        try:
            predicted_outcome_encoded = outcome_encoded[0]
            predicted_outcome = label_encoder.inverse_transform([predicted_outcome_encoded])[0]
        except:
            # Fallback map if encoder issues
            outcome_map = {0: 'Away Win', 1: 'Draw', 2: 'Home Win'}
            predicted_outcome = outcome_map.get(outcome_encoded[0], 'Unknown')
        
        # Probs with fallback
        outcome_probs = None
        if hasattr(outcome_model, 'predict_proba'):
            outcome_probs = outcome_model.predict_proba(processed_features)[0]
        
        # Results with team context
        home_team = input_values['HomeTeam']
        away_team = input_values['AwayTeam']
        st.subheader(f"Prediction: {home_team} vs. {away_team}")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Outcome", predicted_outcome)
            st.metric("Total Goals", f"{round(goals_pred)}")  # Rounded for simplicity
            # Simple over/under
            over_prob = "Likely Over 2.5" if goals_pred > 2.5 else "Likely Under 2.5"
            st.metric("Over/Under 2.5", over_prob)
        
        if outcome_probs is not None:
            # Series for explicit chart
            probs_series = pd.Series({
                "Home Win": outcome_probs[2],  # Index per classes_ (2=Home)
                "Draw": outcome_probs[1],
                "Away Win": outcome_probs[0]
            })
            with col2:
                st.bar_chart(probs_series)
            st.write(f"Outcome Confidence: {max(outcome_probs):.1%}")
        else:
            with col2:
                st.warning("Probabilities unavailable—using predict only.")
    else:
        st.error("Preprocessing or models failed—check inputs (e.g., Odds >0, no NaNs).")