import streamlit as st
import joblib
import numpy as np

# Set the page configuration first (only once)
st.set_page_config(page_title="Salary Predictor", layout="centered", page_icon="💵")

# Load the trained model
model = joblib.load('gradient_boosting_model.pkl')

# Optional image header
st.image("header.jpg", use_container_width=True)

# Title and subtitle
st.markdown("""
# 💼 Salary Prediction App
Welcome! Estimate your salary in **thousands of USD (K)** based on your job profile.
""")

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

# --------- Form Layout ---------

# Custom CSS styling
st.markdown("""
<style>
    .header { 
        font-size: 40px !important;
        color: #2E86C1;
        text-align: center;
        padding: 20px;
    }
    .subheader {
        font-size: 20px !important;
        color: #5D6D7E;
        text-align: center;
        margin-bottom: 30px;
    }
    .stButton>button {
        background: linear-gradient(45deg, #2E86C1, #3498DB);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 15px 30px;
        font-size: 18px;
        transition: all 0.3s;
        width: 100%;
    }
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 5px 15px rgba(46, 134, 193, 0.4);
    }
.result-box {
    background: #EBF5FB;
    border-radius: 15px;
    padding: 25px;
    margin: 20px 0;
    text-align: center;
    max-width: 700px; /* Increase the width */
    margin-left: auto;  /* Center align */
    margin-right: auto; /* Center align */
}
</style>
""", unsafe_allow_html=True)

# Header section
st.markdown('<p class="header">💰 Salary Prediction Assistant</p>', unsafe_allow_html=True)
st.markdown('<p class="subheader">Estimate your potential earnings in the tech industry (USD)</p>', unsafe_allow_html=True)

# Form container
with st.container():
    col1, col2 = st.columns(2, gap="large")  # Change gap to "large"

    with col1:
        st.markdown("### 🧑💻 Professional Details")
        work_year = st.slider("**📅 Work Year**", 2020, 2025, 2023,
                            help="Select the year of employment")
        experience_label = st.radio("**📈 Experience Level**", 
                                   list(experience_options.keys()))
        job_title = st.selectbox("**👔 Job Title**", job_titles,
                                help="Select the closest matching role")

    with col2:
        st.markdown("### 🏢 Company Information")
        company_size_label = st.radio("**🏭 Company Size**", 
                                     list(size_options.keys()),
                                     horizontal=True)
        location_label = st.selectbox("**🌐 Office Location**", 
                                    list(country_options.keys()),
                                    help="Physical location of company headquarters")
        remote_ratio = st.select_slider("**🏡 Remote Work Percentage**", 
                                       options=[0, 25, 50, 75, 100],
                                       value=50,
                                       format_func=lambda x: f"{x}%")

# Additional details section
with st.expander("🔍 Additional Parameters", expanded=True):
    col3, col4 = st.columns(2)
    with col3:
        employment_label = st.selectbox("**📝 Contract Type**", 
                                      list(employment_options.keys()),
                                      help="Type of employment contract")
    with col4:
        residence_label = st.selectbox("**🌍 Residence Country**", 
                                      list(country_options.keys()),
                                      help="Your country of residence")

# Prediction section
st.markdown("---")
col_btn, _ = st.columns([1, 2])
with col_btn:
    if st.button("🚀 Predict My Salary"):
        with st.spinner("Analyzing market trends..."):
            input_features = np.array([[ 
                work_year,
                experience_map[experience_options[experience_label]],
                employment_map[employment_options[employment_label]],
                job_map[job_title],
                residence_map[country_options[residence_label]],
                remote_ratio,
                location_map[country_options[location_label]],
                size_map[size_options[company_size_label]]
            ]])

            try:
                prediction = model.predict(input_features)[0] / 1000
                st.markdown(f"""
                <div class="result-box">
                    <h3 style="color:#2E86C1; margin-bottom:15px;">ESTIMATED SALARY</h3>
                    <p style="font-size:36px; color:#28B463; margin:0;">${prediction:.1f}K USD</p>
                    <p style="color:#5D6D7E; margin-top:10px;">Based on current market trends</p>
                </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Prediction error: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #5D6D7E; margin-top: 30px;">
    <p>🔍 Predictions are based on machine learning models trained on industry data</p>
    <p>⚠️ Results are estimates only and should not be considered financial advice</p>
</div>
""", unsafe_allow_html=True)
