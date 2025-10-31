import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder  # For fallback
import warnings
warnings.filterwarnings("ignore", message="Mean of empty slice")

# Define constants FIRST (before any functions)
OUTCOME_MODEL_FILENAME = 'Gradient_Boosting_Classifier_outcome_model.joblib'
GOALS_MODEL_FILENAME = 'XGBoost_Regressor_goals_model.joblib'
LABEL_ENCODER_FILENAME = 'label_encoder.joblib'

# Now define the function
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

# Call the function after definition
outcome_model, goals_model, label_encoder = load_resources()

# Define feature columns - MUST MATCH the list used during training
feature_columns = [
    'HomeElo', 'AwayElo', 'EloDiff', 'EloSum', 'Form3Home', 'Form5Home',
    'Form3Away', 'Form5Away', 'Form3Ratio', 'Form5Ratio',
    'HomeExpectedGoals', 'AwayExpectedGoals', 'HomeWinProbability',
    'DrawProbability', 'AwayWinProbability', 'HomeShotEfficiency',
    'AwayShotEfficiency', 'HomeAttackStrength', 'AwayAttackStrength',
    'HandiSize', 'Over25', 'Under25', 'C_LTH', 'C_LTA', 'C_VHD',
    'C_VAD', 'C_HTB', 'C_PHB'
]

# Base columns for user input (exclude derived: EloDiff, ratios, probs, eff, attack, ExpectedGoals)
base_columns = [
    'HomeElo', 'AwayElo', 'Form3Home', 'Form5Home', 'Form3Away', 'Form5Away',
    'HomeShots', 'AwayShots', 'HomeTarget', 'AwayTarget',
    'OddHome', 'OddDraw', 'OddAway', 'HandiSize', 'Over25', 'Under25',
    'C_LTH', 'C_LTA', 'C_VHD', 'C_VAD', 'C_HTB', 'C_PHB'
]

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

# --- Streamlit App Layout and Logic ---
st.title("Football Match Predictor")

# Define input_values early, before model conditional
input_values = {}
input_values['HomeTeam'] = st.sidebar.text_input("Home Team", "Team A")
input_values['AwayTeam'] = st.sidebar.text_input("Away Team", "Team B")

if outcome_model is not None and goals_model is not None:  # Your existing conditional
    # Sidebar stats inputs (feature_input_values loop here)
    st.sidebar.subheader("Match Statistics")
    feature_input_values = {}
    for col in base_columns + ['HomeExpectedGoals', 'AwayExpectedGoals']:
        label = col
        default_value = 0.0  # Default float
        step_size = 0.01     # Default float
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
            default_value = 0     # int
            min_value = 0         # int
            max_value = 9         # int
            step_size = 1         # int
        elif col == 'Form5Home':
            label = "Home Form (last 5 matches)"
            default_value = 0     # int
            min_value = 0         # int
            max_value = 15        # int
            step_size = 1         # int
        elif col == 'Form3Away':
            label = "Away Form (last 3 matches)"
            default_value = 0     # int
            min_value = 0         # int
            max_value = 9         # int
            step_size = 1         # int
        elif col == 'Form5Away':
            label = "Away Form (last 5 matches)"
            default_value = 0     # int
            min_value = 0         # int
            max_value = 15        # int
            step_size = 1         # int
        elif col == 'HomeShots':
            label = "Home Shots (expected)"
            default_value = 12.0
            min_value = 0.0
            max_value = 30.0      # float!
            step_size = 1.0
        elif col == 'AwayShots':
            label = "Away Shots (expected)"
            default_value = 10.0
            min_value = 0.0
            max_value = 30.0      # float!
            step_size = 1.0
        elif col == 'HomeTarget':
            label = "Home Shots on Target (expected)"
            default_value = 4.0
            min_value = 0.0
            max_value = 15.0      # float!
            step_size = 1.0
        elif col == 'AwayTarget':
            label = "Away Shots on Target (expected)"
            default_value = 3.0
            min_value = 0.0
            max_value = 15.0      # float!
            step_size = 1.0
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

        # Create the number input widget (now all consistent types per branch)
        feature_input_values[col] = st.sidebar.number_input(
            label=label,
            value=default_value,
            step=step_size,
            min_value=min_value,
            max_value=max_value
        )

    # On button: (now input_values is defined)
    if st.sidebar.button("Predict"):
        input_data = {'HomeTeam': input_values['HomeTeam'], 'AwayTeam': input_values['AwayTeam']}
        input_data.update(feature_input_values)
        input_df_raw = pd.DataFrame([input_data])
        
        processed_features = preprocess_input_data(input_df_raw, feature_columns)
        
        if processed_features is not None:
            outcome_encoded = outcome_model.predict(processed_features)
            goals_pred = goals_model.predict(processed_features)[0]
            
            # Probs with fallback
            outcome_probs = None
            if hasattr(outcome_model, 'predict_proba'):
                outcome_probs = outcome_model.predict_proba(processed_features)[0]
            
            # DYNAMIC decoding and chart using encoder/model classes (SYNCED!)
            classes = getattr(label_encoder, 'classes_', outcome_model.classes_)  # Prefer encoder, fallback to model
            predicted_outcome_encoded = outcome_encoded[0]
            try:
                predicted_outcome = label_encoder.inverse_transform([predicted_outcome_encoded])[0]
            except:
                # True fallback: Use classes directly
                predicted_outcome = classes[predicted_outcome_encoded]
            
            # Validate: argmax(probs) should == predicted_outcome_encoded
            if outcome_probs is not None:
                argmax_idx = np.argmax(outcome_probs)
                if argmax_idx != predicted_outcome_encoded:
                    st.error(f"⚠️ Model inconsistency: Predict {predicted_outcome_encoded} but max prob at {argmax_idx}. Retrain model!")
                else:
                    st.success("✅ Model consistent.")
            
            # Dynamic probs dict for chart
            probs_dict = {}
            for i, cls in enumerate(classes):
                if cls == 'H':
                    probs_dict["Home Win"] = outcome_probs[i]
                elif cls == 'D':
                    probs_dict["Draw"] = outcome_probs[i]
                elif cls == 'A':
                    probs_dict["Away Win"] = outcome_probs[i]
            
            # Results with team context
            home_team = input_values['HomeTeam']
            away_team = input_values['AwayTeam']
            st.subheader(f"Prediction: {home_team} vs. {away_team}")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Outcome", predicted_outcome)
                st.metric("Total Goals", f"{round(goals_pred)}")
                over_prob = "Likely Over 2.5" if goals_pred > 2.5 else "Likely Under 2.5"
                st.metric("Over/Under 2.5", over_prob)
            
            if outcome_probs is not None:
                probs_series = pd.Series(probs_dict)
                with col2:
                    st.bar_chart(probs_series)
                st.write(f"Outcome Confidence: {max(outcome_probs):.1%}")
                # Temp debug (remove after fix)
                st.write(f"Debug: Classes={classes}, Encoded={predicted_outcome_encoded}, Probs={outcome_probs}")
            else:
                with col2:
                    st.warning("Probabilities unavailable—using predict only.")
        else:
            st.error("Preprocessing failed—check inputs.")
else:
    st.warning("Models could not be loaded...")
    # Optionally disable predict button: st.sidebar.button("Predict", disabled=True)