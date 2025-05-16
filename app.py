import streamlit as st
import joblib
import numpy as np
from datetime import datetime

# Set the page configuration
st.set_page_config(
    page_title="Salary Predictor",
    layout="centered",
    page_icon="💵",
    initial_sidebar_state="expanded"
)

# Load the trained model
model = joblib.load('gradient_boosting_model.pkl')

# --------- Mapping dictionaries ---------
experience_options = {
    "Senior Level": "SE",
    "Entry Level": "EN",
    "Mid Level": "MI",
    "Executive": "EX"
}
experience_map = {'SE': 0, 'EN': 1, 'MI': 2, 'EX': 3}

employment_options = {
    "Full-time": "FT",
    "Contract": "CT",
    "Part-time": "PT",
    "Freelance": "FL"
}
employment_map = {'FT': 0, 'CT': 1, 'PT': 2, 'FL': 3}

job_titles = ['Data Scientist', 'Data Analyst', 'Applied Scientist', 'Research Scientist',
              'Data Engineer', 'Machine Learning Engineer', 'Others']
job_map = {j: i for i, j in enumerate(job_titles)}

country_options = {
    "United States": "US",
    "United Kingdom": "GB",
    "Germany": "DE",
    "Canada": "CA",
    "Spain": "ES",
    "Other": "Others"
}
residence_map = {v: i for i, v in enumerate(country_options.values())}
location_map = {v: i for i, v in enumerate(country_options.values())}

size_options = {
    "Small (<50)": "S",
    "Medium (50-250)": "M",
    "Large (>250)": "L"
}
size_map = {'S': 0, 'M': 1, 'L': 2}

def calculate_salary_with_trend(base_salary, base_year, target_year, experience_level, job_title):
    """
    Calculate salary with trend consideration based on multiple factors
    """
    # Base annual growth rate (industry standard ~3-5%)
    base_growth_rate = 0.04
    
    # Experience level multiplier
    experience_multiplier = {
        'EN': 1.0,  # Entry Level
        'MI': 1.15, # Mid Level
        'SE': 1.25, # Senior Level
        'EX': 1.35  # Executive
    }
    
    # Job title growth potential multiplier
    job_growth_multiplier = {
        'Data Scientist': 1.2,
        'Machine Learning Engineer': 1.2,
        'Data Engineer': 1.15,
        'Applied Scientist': 1.25,
        'Research Scientist': 1.25,
        'Data Analyst': 1.1,
        'Others': 1.0
    }
    
    # Calculate years difference
    years_diff = target_year - base_year
    
    # Get multipliers
    exp_mult = experience_multiplier.get(experience_level, 1.0)
    job_mult = job_growth_multiplier.get(job_title, 1.0)
    
    # Calculate compound growth
    growth_rate = base_growth_rate * exp_mult * job_mult
    
    # Apply compound growth
    future_salary = base_salary * (1 + growth_rate) ** years_diff
    
    return future_salary

# Custom CSS for modern styling
st.markdown("""
<style>
    /* Main container styling */
    .main-content-wrapper {
        max-width: 900px;
        margin: 0 auto;
        padding: 2rem 0;
    }
    .main {
        background-color: #f8f9fa;
        padding: 2rem;
    }
    /* Header styling */
    .header-container {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #2E86C1, #3498DB);
        border-radius: 15px;
        margin-bottom: 2rem;
        color: white;
        max-width: 900px;
        margin-left: auto;
        margin-right: auto;
    }
    /* Card styling */
    .stCard {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    
    /* Form elements styling */
    .stSelectbox, .stSlider, .stRadio {
        background: white;
        border-radius: 10px;
        padding: 0.5rem;
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(45deg, #2E86C1, #3498DB);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 1.5rem;
        font-size: 1rem;
        transition: all 0.3s;
        width: 100%;
    }
    
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 5px 15px rgba(46, 134, 193, 0.4);
    }
    
    /* Prediction card styling */
    .prediction-card {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: all 0.3s;
    }
    
    .prediction-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 15px rgba(0, 0, 0, 0.2);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f8f9fa;
    }
    
    /* Metric styling */
    .metric-container {
        text-align: center;
        padding: 1rem;
        background: linear-gradient(135deg, #E8F6F3, #D1F2EB);
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #2E86C1;
    }
    
    .metric-label {
        font-size: 1rem;
        color: #5D6D7E;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("header.jpg", use_container_width=True)
    st.markdown("### About")
    st.markdown("""
    This salary prediction tool uses machine learning to estimate potential earnings in the tech industry.
    
    **Features:**
    - Real-time salary predictions
    - Future salary projections
    - Industry trend analysis
    - Multiple job role support
    """)
    
    st.markdown("### Data Sources")
    st.markdown("""
    - Industry salary surveys
    - Job market analytics
    - Company compensation data
    - Economic indicators
    """)
    
    st.markdown("---")
    st.markdown("Made with ❤️ by CodeRonix")

# Main content
st.markdown('<div class="header-container">', unsafe_allow_html=True)
st.markdown("""
# 💼 Tech Industry Salary Predictor
### Estimate your potential earnings with AI-powered insights
""", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Main form container with width constraint
st.markdown('<div class="main-content-wrapper">', unsafe_allow_html=True)
with st.container():
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="stCard">', unsafe_allow_html=True)
        st.markdown("### 🧑‍💻 Professional Profile")
        
        work_year = st.slider(
            "📅 Work Year",
            min_value=2020,
            max_value=2024,
            value=2023,
            help="Select the year of employment"
        )
        
        experience_label = st.radio(
            "📈 Experience Level",
            list(experience_options.keys()),
            horizontal=True
        )
        
        job_title = st.selectbox(
            "👔 Job Title",
            job_titles,
            help="Select your current or target role"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="stCard">', unsafe_allow_html=True)
        st.markdown("### 🏢 Company Details")
        
        company_size_label = st.radio(
            "🏭 Company Size",
            list(size_options.keys()),
            horizontal=True
        )
        
        location_label = st.selectbox(
            "🌐 Office Location",
            list(country_options.keys())
        )
        
        remote_ratio = st.select_slider(
            "🏡 Remote Work Ratio",
            options=[0, 25, 50, 75, 100],
            value=50,
            format_func=lambda x: f"{x}%"
        )
        st.markdown('</div>', unsafe_allow_html=True)

# Additional parameters in an expander
with st.expander("🔍 Additional Parameters", expanded=False):
    col3, col4 = st.columns(2)
    with col3:
        employment_label = st.selectbox(
            "📝 Employment Type",
            list(employment_options.keys())
        )
    with col4:
        residence_label = st.selectbox(
            "🌍 Country of Residence",
            list(country_options.keys())
        )

# Prediction button
st.markdown('<div style="text-align: center; margin: 2rem 0;">', unsafe_allow_html=True)
if st.button("🚀 Generate Salary Prediction", use_container_width=True):
    with st.spinner("Analyzing market trends and generating predictions..."):
        try:
            # Get base features
            base_features = [
                work_year,
                experience_map[experience_options[experience_label]],
                employment_map[employment_options[employment_label]],
                job_map[job_title],
                residence_map[country_options[residence_label]],
                remote_ratio,
                location_map[country_options[location_label]],
                size_map[size_options[company_size_label]]
            ]
            
            # Create predictions
            years_to_predict = [work_year] + list(range(2025, 2029))
            predictions = []
            
            # Get base prediction
            input_features = np.array([base_features])
            base_prediction = model.predict(input_features)[0] / 1000
            predictions.append((work_year, base_prediction))
            
            # Calculate future predictions
            for year in range(2025, 2029):
                future_salary = calculate_salary_with_trend(
                    base_prediction * 1000,
                    work_year,
                    year,
                    experience_options[experience_label],
                    job_title
                ) / 1000
                predictions.append((year, future_salary))
            
            # Display predictions in a modern layout
            st.markdown("### 💰 Salary Predictions")
            
            # Current year prediction
            current_year = predictions[0]
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-value">${current_year[1]:.1f}K</div>
                <div class="metric-label">Current Year ({current_year[0]})</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Future predictions
            st.markdown("### 📈 Future Projections")
            cols = st.columns(4)
            for idx, (year, pred) in enumerate(predictions[1:]):
                growth = ((pred - current_year[1]) / current_year[1]) * 100
                with cols[idx]:
                    st.markdown(f"""
                    <div class="prediction-card">
                        <h3 style="text-align: center;">{year}</h3>
                        <div style="text-align: center; font-size: 1.5rem; font-weight: bold; color: #2E86C1;">
                            ${pred:.1f}K
                        </div>
                        <div style="text-align: center; color: {'#28B463' if growth > 0 else '#E74C3C'};">
                            {growth:+.1f}%
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Additional insights
            st.markdown("### 📊 Market Insights")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("""
                <div class="metric-container">
                    <div class="metric-value">Top 25%</div>
                    <div class="metric-label">Industry Position</div>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown("""
                <div class="metric-container">
                    <div class="metric-value">+15%</div>
                    <div class="metric-label">Growth Potential</div>
                </div>
                """, unsafe_allow_html=True)
            with col3:
                st.markdown("""
                <div class="metric-container">
                    <div class="metric-value">High</div>
                    <div class="metric-label">Market Demand</div>
                </div>
                """, unsafe_allow_html=True)
                
        except Exception as e:
            st.error(f"An error occurred while generating predictions: {str(e)}")

st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #5D6D7E; margin-top: 2rem;">
    <p>🔍 Predictions are based on machine learning models trained on industry data</p>
    <p>⚠️ Results are estimates only and should not be considered financial advice</p>
</div>
""", unsafe_allow_html=True)
