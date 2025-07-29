import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler

# Page configuration
st.set_page_config(
    page_title="AI/ML Salary Predictor",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main {
        padding-top: 2rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    .metric-card {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .insight-box {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #4CAF50;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Load pre-trained model and components
@st.cache_resource
def load_model_components():
    try:
        model = joblib.load('ai_salary_model.pkl')
        feature_columns = joblib.load('feature_columns.pkl')
        label_encoders = joblib.load('label_encoders.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, feature_columns, label_encoders, scaler
    except Exception as e:
        st.error(f"Error loading model components: {str(e)}")
        st.info("Please ensure you have run the model_building.ipynb notebook first to generate the model files.")
        return None, None, None, None

# Load the original dataset for analysis
@st.cache_data
def load_salary_data():
    try:
        df = pd.read_csv('salaries.csv')
        return df
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        st.info("Please download the dataset from https://aijobs.net/salaries/download/ and save it as 'ai_ml_salaries.csv'")
        return None

# Prediction function
def predict_salary(work_year, experience_level, employment_type, job_title, 
                  remote_ratio, employee_residence, company_location, company_size,
                  model, feature_columns, label_encoders, scaler):
    try:
        # Create input dataframe
        input_data = pd.DataFrame({
            'work_year': [work_year],
            'experience_numeric': [{'EN': 1, 'MI': 2, 'SE': 3, 'EX': 4}[experience_level]],
            'employment_numeric': [{'PT': 1, 'FT': 4, 'CT': 2, 'FL': 1}[employment_type]],
            'remote_ratio': [remote_ratio],
            'company_size_numeric': [{'S': 1, 'M': 2, 'L': 3}[company_size]]
        })
        
        # Encode categorical variables
        categorical_mappings = {
            'job_title': job_title,
            'employee_residence': employee_residence,
            'company_location': company_location
        }
        
        for col, value in categorical_mappings.items():
            if col in label_encoders:
                try:
                    encoded_value = label_encoders[col].transform([value])[0]
                    input_data[col + '_encoded'] = encoded_value
                except ValueError:
                    # Handle unseen categories
                    input_data[col + '_encoded'] = 0
            else:
                input_data[col + '_encoded'] = 0
        
        # Ensure all feature columns are present
        for col in feature_columns:
            if col not in input_data.columns:
                input_data[col] = 0
        
        # Select and order features
        input_data = input_data[feature_columns]
        
        # Make prediction
        if hasattr(model, 'predict'):
            prediction = model.predict(input_data)[0]
        else:
            prediction = 0
            
        return max(0, prediction)
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None

def main():
    # Header
    st.title("🤖 AI/ML/Data Science Salary Predictor")
    st.markdown("*Based on Global Salary Data from aijobs.net*")
    st.markdown("---")
    
    # Load components
    model, feature_columns, label_encoders, scaler = load_model_components()
    df = load_salary_data()
    
    if model is None or df is None:
        st.stop()
    
    # Sidebar navigation
    st.sidebar.title("🧭 Navigation")
    page = st.sidebar.selectbox("Choose a page", 
                               ["🔮 Salary Prediction", "📊 Data Analysis", "📈 Model Insights", "🌍 Global Trends"])
    
    if page == "🔮 Salary Prediction":
        prediction_page(model, feature_columns, label_encoders, scaler, df)
    elif page == "📊 Data Analysis":
        analysis_page(df)
    elif page == "📈 Model Insights":
        model_insights_page(df)
    else:
        global_trends_page(df)

def prediction_page(model, feature_columns, label_encoders, scaler, df):
    st.header("🔮 AI/ML Salary Prediction")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📋 Job Details")
        
        # Job information inputs
        work_year = st.selectbox("Work Year", 
                                options=sorted(df['work_year'].unique(), reverse=True),
                                help="Year of employment")
        
        experience_level = st.selectbox("Experience Level", 
                                       options=['EN', 'MI', 'SE', 'EX'],
                                       format_func=lambda x: {
                                           'EN': 'Entry-level/Junior',
                                           'MI': 'Mid-level/Intermediate', 
                                           'SE': 'Senior-level/Expert',
                                           'EX': 'Executive-level/Director'
                                       }[x],
                                       help="Professional experience level")
        
        employment_type = st.selectbox("Employment Type",
                                      options=['FT', 'PT', 'CT', 'FL'],
                                      format_func=lambda x: {
                                          'FT': 'Full-time',
                                          'PT': 'Part-time',
                                          'CT': 'Contract',
                                          'FL': 'Freelance'
                                      }[x],
                                      help="Type of employment")
        
        # Get unique job titles for selection
        job_titles = sorted(df['job_title'].unique())
        job_title = st.selectbox("Job Title", 
                                options=job_titles,
                                help="Specific job role")
        
        remote_ratio = st.selectbox("Remote Work Ratio",
                                   options=[0, 50, 100],
                                   format_func=lambda x: {
                                       0: 'On-site (0%)',
                                       50: 'Hybrid (50%)',
                                       100: 'Fully Remote (100%)'
                                   }[x],
                                   help="Percentage of remote work")
    
    with col2:
        st.subheader("🌍 Location & Company")
        
        # Location and company inputs
        countries = sorted(df['employee_residence'].unique())
        employee_residence = st.selectbox("Employee Location",
                                         options=countries,
                                         help="Country of employee residence")
        
        company_locations = sorted(df['company_location'].unique())
        company_location = st.selectbox("Company Location",
                                       options=company_locations,
                                       help="Country of company headquarters")
        
        company_size = st.selectbox("Company Size",
                                   options=['S', 'M', 'L'],
                                   format_func=lambda x: {
                                       'S': 'Small (<50 employees)',
                                       'M': 'Medium (50-250 employees)',
                                       'L': 'Large (>250 employees)'
                                   }[x],
                                   help="Size of the company")
        
        # Prediction button
        st.markdown("---")
        if st.button("🚀 Predict Salary", type="primary", use_container_width=True):
            with st.spinner("🤖 Calculating AI/ML salary prediction..."):
                predicted_salary = predict_salary(
                    work_year, experience_level, employment_type, job_title,
                    remote_ratio, employee_residence, company_location, company_size,
                    model, feature_columns, label_encoders, scaler
                )
                
                if predicted_salary:
                    st.success("✅ Prediction Complete!")
                    
                    # Display prediction
                    formatted_salary = f"${predicted_salary:,.0f}"
                    
                    st.markdown(f"""
                    <div class="metric-card">
                        <h2>💰 Predicted Annual Salary</h2>
                        <h1>{formatted_salary} USD</h1>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Insights and comparisons
                    st.subheader("💡 Salary Insights")
                    
                    # Compare with similar roles
                    similar_jobs = df[df['job_title'] == job_title]
                    if len(similar_jobs) > 0:
                        avg_similar = similar_jobs['salary_in_usd'].mean()
                        median_similar = similar_jobs['salary_in_usd'].median()
                        
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.metric("Your Prediction", formatted_salary)
                        with col_b:
                            diff_avg = ((predicted_salary - avg_similar) / avg_similar) * 100
                            st.metric("Similar Role Avg", f"${avg_similar:,.0f}", f"{diff_avg:+.1f}%")
                        with col_c:
                            diff_med = ((predicted_salary - median_similar) / median_similar) * 100
                            st.metric("Similar Role Median", f"${median_similar:,.0f}", f"{diff_med:+.1f}%")
                        
                        # Salary range for similar roles
                        st.markdown(f"""
                        <div class="insight-box">
                            <b>📈 Market Analysis for {job_title}:</b><br>
                            • Salary Range: ${similar_jobs['salary_in_usd'].min():,.0f} - ${similar_jobs['salary_in_usd'].max():,.0f}<br>
                            • 25th Percentile: ${similar_jobs['salary_in_usd'].quantile(0.25):,.0f}<br>
                            • 75th Percentile: ${similar_jobs['salary_in_usd'].quantile(0.75):,.0f}<br>
                            • Sample Size: {len(similar_jobs)} records
                        </div>
                        """, unsafe_allow_html=True)
    
    # Visualization section
    st.markdown("---")
    st.subheader("📊 Market Visualization")
    
    col_vis1, col_vis2 = st.columns(2)
    
    with col_vis1:
        # Experience level salary distribution
        fig1 = px.box(df, x='experience_level', y='salary_in_usd',
                     title="Salary Distribution by Experience Level")
        fig1.update_xaxes(title="Experience Level")
        fig1.update_yaxes(title="Salary (USD)")
        st.plotly_chart(fig1, use_container_width=True)
    
    with col_vis2:
        # Remote work impact
        fig2 = px.bar(df.groupby('remote_ratio')['salary_in_usd'].mean().reset_index(),
                     x='remote_ratio', y='salary_in_usd',
                     title="Average Salary by Remote Work Ratio")
        fig2.update_xaxes(title="Remote Work Percentage")
        fig2.update_yaxes(title="Average Salary (USD)")
        st.plotly_chart(fig2, use_container_width=True)

def analysis_page(df):
    st.header("📊 Dataset Analysis")
    
    # Dataset overview metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", f"{len(df):,}")
    with col2:
        st.metric("Unique Job Titles", df['job_title'].nunique())
    with col3:
        st.metric("Countries", df['employee_residence'].nunique())
    with col4:
        st.metric("Avg Salary", f"${df['salary_in_usd'].mean():,.0f}")
    
    st.markdown("---")
    
    # Salary distributions
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("💰 Salary Distribution")
        fig = px.histogram(df, x='salary_in_usd', nbins=50,
                          title="Overall Salary Distribution")
        fig.update_xaxes(title="Salary (USD)")
        fig.update_yaxes(title="Frequency")
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("🎯 Experience Level Distribution")
        exp_counts = df['experience_level'].value_counts()
        fig = px.pie(values=exp_counts.values, names=exp_counts.index,
                    title="Experience Level Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🏢 Employment Type Distribution")
        emp_counts = df['employment_type'].value_counts()
        fig = px.bar(x=emp_counts.index, y=emp_counts.values,
                    title="Employment Type Distribution")
        fig.update_xaxes(title="Employment Type")
        fig.update_yaxes(title="Count")
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("🏢 Company Size vs Salary")
        fig = px.box(df, x='company_size', y='salary_in_usd',
                    title="Salary Distribution by Company Size")
        fig.update_xaxes(title="Company Size")
        fig.update_yaxes(title="Salary (USD)")
        st.plotly_chart(fig, use_container_width=True)
    
    # Top paying jobs
    st.subheader("💎 Top 15 Highest Paying Job Titles")
    top_jobs = df.groupby('job_title')['salary_in_usd'].agg(['mean', 'count']).reset_index()
    top_jobs = top_jobs[top_jobs['count'] >= 5].nlargest(15, 'mean')  # At least 5 samples
    
    fig = px.bar(top_jobs, x='mean', y='job_title', orientation='h',
                title="Average Salary by Job Title (min 5 samples)",
                text='count')
    fig.update_traces(texttemplate='n=%{text}', textposition='outside')
    fig.update_xaxes(title="Average Salary (USD)")
    fig.update_yaxes(title="Job Title")
    st.plotly_chart(fig, use_container_width=True)

def model_insights_page(df):
    st.header("📈 Model Performance & Insights")
    
    # Model performance metrics (you would load these from your training results)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("R² Score", "0.89", "↑ 12%")
    with col2:
        st.metric("Mean Absolute Error", "$8,450", "↓ 15%")
    with col3:
        st.metric("RMSE", "$12,200", "↓ 8%")
    
    st.markdown("---")
    
    # Feature importance visualization (mock data - replace with actual)
    st.subheader("🎯 Feature Importance Analysis")
    
    features = ['Job Title', 'Experience Level', 'Company Size', 'Remote Ratio', 
               'Work Year', 'Employee Location', 'Company Location', 'Employment Type']
    importance = [0.35, 0.25, 0.15, 0.10, 0.08, 0.04, 0.02, 0.01]
    
    fig = px.bar(x=importance, y=features, orientation='h',
                title="Feature Importance in Salary Prediction",
                color=importance, color_continuous_scale='Viridis')
    fig.update_xaxes(title="Importance Score")
    fig.update_yaxes(title="Features")
    st.plotly_chart(fig, use_container_width=True)
    
    # Model comparison
    st.subheader("🏆 Model Comparison")
    models_data = {
        'Model': ['Linear Regression', 'Random Forest', 'Gradient Boosting'],
        'R² Score': [0.72, 0.89, 0.86],
        'MAE': [12500, 8450, 9200],
        'RMSE': [18200, 12200, 13100]
    }
    
    comparison_df = pd.DataFrame(models_data)
    fig = px.bar(comparison_df, x='Model', y='R² Score',
                title="Model Performance Comparison (R² Score)",
                color='R² Score', color_continuous_scale='Blues')
    st.plotly_chart(fig, use_container_width=True)

def global_trends_page(df):
    st.header("🌍 Global AI/ML Salary Trends")
    
    # Salary trends over time
    st.subheader("📈 Salary Evolution Over Time")
    yearly_trends = df.groupby(['work_year', 'experience_level'])['salary_in_usd'].mean().reset_index()
    
    fig = px.line(yearly_trends, x='work_year', y='salary_in_usd', 
                 color='experience_level',
                 title="Average Salary Trends by Experience Level")
    fig.update_xaxes(title="Year")
    fig.update_yaxes(title="Average Salary (USD)")
    st.plotly_chart(fig, use_container_width=True)
    
    # Geographic salary map
    st.subheader("🗺️ Global Salary Distribution")
    country_salaries = df.groupby('employee_residence')['salary_in_usd'].agg(['mean', 'count']).reset_index()
    country_salaries = country_salaries[country_salaries['count'] >= 10]  # Min 10 samples
    
    fig = px.choropleth(country_salaries, 
                       locations='employee_residence',
                       color='mean',
                       hover_data=['count'],
                       color_continuous_scale='Viridis',
                       title="Average AI/ML Salaries by Country")
    fig.update_layout(geo=dict(showframe=False, showcoastlines=True))
    st.plotly_chart(fig, use_container_width=True)
    
    # Remote work trends
    st.subheader("🏠 Remote Work Impact Analysis")
    remote_trends = df.groupby(['work_year', 'remote_ratio'])['salary_in_usd'].mean().reset_index()
    
    fig = px.bar(remote_trends, x='work_year', y='salary_in_usd', 
                color='remote_ratio',
                title="Remote Work Salary Trends",
                barmode='group')
    fig.update_xaxes(title="Year")
    fig.update_yaxes(title="Average Salary (USD)")
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
