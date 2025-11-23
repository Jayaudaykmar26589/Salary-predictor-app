import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import pickle
import warnings
from datetime import datetime
import io

warnings.filterwarnings('ignore')
sns.set_style("whitegrid")

# ============================================================================
# CONFIGURATION & UTILITIES
# ============================================================================

class Config:
    CURRENT_YEAR = 2024
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    CV_FOLDS = 5

def create_download_link(obj, filename, link_text):
    """Create a download link for pickle files"""
    output = pickle.dumps(obj)
    b64 = io.BytesIO(output).getvalue()
    return st.download_button(
        label=link_text,
        data=b64,
        file_name=filename,
        mime='application/octet-stream'
    )

# ============================================================================
# DATA LOADING & COMPREHENSIVE CLEANING
# ============================================================================

@st.cache_data
def load_and_clean_data(filepath):
    """Enhanced data loading and cleaning with error handling"""
    try:
        df = pd.read_csv(filepath)
        st.success(f"✅ Data loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
    except FileNotFoundError:
        st.error(f"❌ File not found at: {filepath}")
        return None
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        return None

    initial_rows = len(df)
    
    # 1. SALARY PARSING
    df = df[df['Salary Estimate'] != '-1'].copy()
    salary = df['Salary Estimate'].apply(lambda x: x.split('(')[0])
    minus_Kd = salary.apply(lambda x: x.replace('K','').replace('$',''))
    min_hr = minus_Kd.apply(lambda x: x.lower().replace('per hour','').replace('employer provided salary:','').strip())
    
    try:
        df['min_salary'] = min_hr.apply(lambda x: int(x.split('-')[0]))
        df['max_salary'] = min_hr.apply(lambda x: int(x.split('-')[1]))
        df['avg_salary'] = (df['min_salary'] + df['max_salary']) / 2
        df['salary_range'] = df['max_salary'] - df['min_salary']
    except:
        st.warning("⚠️ Some salary parsing issues detected - rows with issues will be dropped")
        df = df.dropna(subset=['min_salary', 'max_salary'])

    # 2. COMPANY NAME CLEANING
    df['company_txt'] = df.apply(
        lambda x: x['Company Name'] if x['Rating'] < 0 else x['Company Name'][:-3].strip(), 
        axis=1
    )

    # 3. LOCATION PARSING
    df['job_state'] = df['Location'].apply(
        lambda x: x.split(',')[-1].strip() if ',' in x else x.strip()
    )
    df['same_state'] = df.apply(
        lambda x: 1 if x['Location'] == x['Headquarters'] else 0, 
        axis=1
    )

    # 4. COMPANY AGE
    df['age'] = df['Founded'].apply(
        lambda x: Config.CURRENT_YEAR - x if x > 0 else -1
    )

    # 5. JOB DESCRIPTION PARSING - EXPANDED
    skills = {
        'python': ['python'],
        'r': ['r studio', 'r-studio', ' r '],
        'spark': ['spark'],
        'aws': ['aws', 'amazon web services'],
        'excel': ['excel'],
        'sql': ['sql', 'mysql', 'postgresql'],
        'tableau': ['tableau'],
        'hadoop': ['hadoop'],
        'tensorflow': ['tensorflow', 'tf'],
        'pytorch': ['pytorch'],
        'scikit': ['scikit', 'sklearn'],
        'docker': ['docker'],
        'kubernetes': ['kubernetes', 'k8s']
    }
    
    for skill, keywords in skills.items():
        df[f'{skill}_yn'] = df['Job Description'].apply(
            lambda x: 1 if any(kw in x.lower() for kw in keywords) else 0
        )
    
    df['total_skills'] = df[[f'{s}_yn' for s in skills.keys()]].sum(axis=1)

    # 6. JOB TITLE CATEGORIZATION
    def title_simplifier(title):
        title_lower = title.lower()
        if 'data scientist' in title_lower:
            return 'data_scientist'
        elif 'data engineer' in title_lower:
            return 'data_engineer'
        elif 'machine learning' in title_lower or 'ml engineer' in title_lower:
            return 'ml_engineer'
        elif 'analyst' in title_lower:
            return 'analyst'
        elif 'manager' in title_lower:
            return 'manager'
        elif 'director' in title_lower:
            return 'director'
        else:
            return 'other'

    def seniority_extractor(title):
        title_lower = title.lower()
        if any(x in title_lower for x in ['sr', 'senior', 'lead', 'principal', 'staff']):
            return 'senior'
        elif any(x in title_lower for x in ['jr', 'junior', 'entry']):
            return 'junior'
        else:
            return 'mid'

    df['job_simp'] = df['Job Title'].apply(title_simplifier)
    df['seniority'] = df['Job Title'].apply(seniority_extractor)

    # 7. REMOTE WORK DETECTION
    df['is_remote'] = df.apply(
        lambda x: 1 if 'remote' in x['Location'].lower() or 'remote' in x['Job Title'].lower() else 0, 
        axis=1
    )

    # 8. DESCRIPTION LENGTH
    df['desc_len'] = df['Job Description'].apply(lambda x: len(str(x)))

    # 9. HANDLE MISSING VALUES IN CATEGORICAL
    categorical_cols = ['Size', 'Type of ownership', 'Industry', 'Sector', 'Revenue']
    for col in categorical_cols:
        df[col] = df[col].replace('-1', 'Unknown')
        df[col] = df[col].fillna('Unknown')

    # 10. RATING HANDLING
    df['Rating'] = df['Rating'].replace(-1, df['Rating'][df['Rating'] > 0].median())
    df['has_rating'] = df['Rating'].apply(lambda x: 1 if x > 0 else 0)

    rows_removed = initial_rows - len(df)
    st.info(f"📊 Cleaned data: {rows_removed} rows removed, {len(df)} rows remaining")
    
    return df

# ============================================================================
# EXPLORATORY DATA ANALYSIS
# ============================================================================

def perform_eda(df):
    """Comprehensive EDA section"""
    st.header("📊 Exploratory Data Analysis")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Overview", "💰 Salary Analysis", "🔍 Feature Analysis", "📊 Correlations"])
    
    with tab1:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Jobs", len(df))
        col2.metric("Avg Salary", f"${df['avg_salary'].mean():.0f}K")
        col3.metric("Median Salary", f"${df['avg_salary'].median():.0f}K")
        col4.metric("Unique Companies", df['company_txt'].nunique())
        
        st.subheader("Dataset Preview")
        st.dataframe(df.head(10), use_container_width=True)
        
        st.subheader("Statistical Summary")
        st.dataframe(df.describe(), use_container_width=True)
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Salary Distribution")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(df['avg_salary'], kde=True, ax=ax, color='#2ecc71', bins=30)
            ax.axvline(df['avg_salary'].mean(), color='red', linestyle='--', label='Mean')
            ax.axvline(df['avg_salary'].median(), color='blue', linestyle='--', label='Median')
            ax.set_xlabel("Average Salary (K$)")
            ax.set_ylabel("Frequency")
            ax.legend()
            st.pyplot(fig)
            plt.close(fig)
        
        with col2:
            st.subheader("Salary Boxplot")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.boxplot(y=df['avg_salary'], ax=ax, color='#3498db')
            ax.set_ylabel("Average Salary (K$)")
            st.pyplot(fig)
            plt.close(fig)
        
        st.subheader("Salary by Key Features")
        feature_choice = st.selectbox(
            "Select Feature", 
            ['seniority', 'job_simp', 'job_state', 'Size', 'Type of ownership', 'is_remote']
        )
        
        fig, ax = plt.subplots(figsize=(12, 6))
        salary_by_feature = df.groupby(feature_choice)['avg_salary'].mean().sort_values(ascending=False).head(10)
        salary_by_feature.plot(kind='barh', ax=ax, color='#e74c3c')
        ax.set_xlabel("Average Salary (K$)")
        ax.set_title(f"Average Salary by {feature_choice}")
        st.pyplot(fig)
        plt.close(fig)
    
    with tab3:
        st.subheader("Top 10 Job Titles")
        top_titles = df['Job Title'].value_counts().head(10)
        fig, ax = plt.subplots(figsize=(10, 6))
        top_titles.plot(kind='barh', ax=ax, color='#9b59b6')
        ax.set_xlabel("Count")
        st.pyplot(fig)
        plt.close(fig)
        
        st.subheader("Skills Demand")
        skills = ['python_yn', 'r_yn', 'sql_yn', 'aws_yn', 'spark_yn', 'excel_yn', 'tableau_yn']
        skill_counts = df[skills].sum().sort_values(ascending=False)
        skill_counts.index = [s.replace('_yn', '').upper() for s in skill_counts.index]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        skill_counts.plot(kind='bar', ax=ax, color='#1abc9c')
        ax.set_ylabel("Number of Jobs")
        ax.set_title("In-Demand Skills")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        st.pyplot(fig)
        plt.close(fig)
        
        st.subheader("Geographic Distribution")
        top_states = df['job_state'].value_counts().head(15)
        fig, ax = plt.subplots(figsize=(10, 6))
        top_states.plot(kind='bar', ax=ax, color='#f39c12')
        ax.set_xlabel("State")
        ax.set_ylabel("Number of Jobs")
        ax.set_title("Top 15 States by Job Count")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        st.pyplot(fig)
        plt.close(fig)
    
    with tab4:
        st.subheader("Correlation Heatmap")
        numeric_cols = ['avg_salary', 'Rating', 'age', 'total_skills', 'desc_len', 'salary_range']
        skill_cols = [col for col in df.columns if col.endswith('_yn')]
        correlation_cols = numeric_cols + skill_cols[:5]
        
        corr_df = df[correlation_cols].corr()
        
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(corr_df, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax, square=True)
        ax.set_title("Feature Correlation Matrix")
        st.pyplot(fig)
        plt.close(fig)
        
        st.subheader("Salary Correlation Insights")
        salary_corr = corr_df['avg_salary'].sort_values(ascending=False)[1:]
        st.dataframe(salary_corr.to_frame('Correlation with Salary'), use_container_width=True)

# ============================================================================
# MODEL BUILDING & TRAINING
# ============================================================================

def prepare_model_data(df):
    """Prepare data for modeling with feature engineering"""
    
    # Select features for modeling
    feature_cols = [
        'Rating', 'Size', 'Type of ownership', 'Industry', 'Sector', 'Revenue',
        'job_state', 'age', 'python_yn', 'r_yn', 'spark_yn', 'aws_yn', 'excel_yn',
        'sql_yn', 'tableau_yn', 'job_simp', 'seniority', 'is_remote', 
        'total_skills', 'desc_len', 'same_state', 'has_rating', 'salary_range'
    ]
    
    # Filter for available columns
    available_cols = [col for col in feature_cols if col in df.columns]
    model_df = df[available_cols + ['avg_salary']].copy()
    
    # Handle any remaining missing values
    model_df = model_df.dropna(subset=['avg_salary'])
    
    # Get dummies for categorical variables
    categorical_cols = model_df.select_dtypes(include=['object']).columns.tolist()
    df_encoded = pd.get_dummies(model_df, columns=categorical_cols, drop_first=True)
    
    X = df_encoded.drop('avg_salary', axis=1)
    y = df_encoded['avg_salary'].values
    
    return X, y, df_encoded.columns.tolist()

def train_models(X_train, X_test, y_train, y_test):
    """Train multiple models and return results"""
    
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Lasso Regression': Lasso(alpha=0.1),
        'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5),
        'Decision Tree': DecisionTreeRegressor(max_depth=10, random_state=Config.RANDOM_STATE),
        'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=15, random_state=Config.RANDOM_STATE),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=Config.RANDOM_STATE)
    }
    
    results = {}
    
    for name, model in models.items():
        with st.spinner(f"Training {name}..."):
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            
            # Cross-validation score
            cv_scores = cross_val_score(model, X_train, y_train, cv=Config.CV_FOLDS, 
                                       scoring='r2', n_jobs=-1)
            
            results[name] = {
                'model': model,
                'predictions': y_pred,
                'mae': mae,
                'rmse': rmse,
                'r2': r2,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }
    
    return results

def display_model_results(results, y_test):
    """Display comprehensive model comparison"""
    
    st.subheader("📊 Model Performance Comparison")
    
    # Create comparison dataframe
    comparison_data = []
    for name, metrics in results.items():
        comparison_data.append({
            'Model': name,
            'MAE ($K)': f"{metrics['mae']:.2f}",
            'RMSE ($K)': f"{metrics['rmse']:.2f}",
            'R² Score': f"{metrics['r2']:.4f}",
            'CV R² Mean': f"{metrics['cv_mean']:.4f}",
            'CV R² Std': f"{metrics['cv_std']:.4f}"
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True)
    
    # Find best model
    best_model_name = max(results.keys(), key=lambda x: results[x]['r2'])
    st.success(f"🏆 Best Model: **{best_model_name}** (R² = {results[best_model_name]['r2']:.4f})")
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("R² Score Comparison")
        fig, ax = plt.subplots(figsize=(10, 6))
        r2_scores = {name: metrics['r2'] for name, metrics in results.items()}
        colors = ['#2ecc71' if name == best_model_name else '#3498db' for name in r2_scores.keys()]
        ax.barh(list(r2_scores.keys()), list(r2_scores.values()), color=colors)
        ax.set_xlabel("R² Score")
        ax.set_title("Model R² Comparison")
        st.pyplot(fig)
        plt.close(fig)
    
    with col2:
        st.subheader("RMSE Comparison")
        fig, ax = plt.subplots(figsize=(10, 6))
        rmse_scores = {name: metrics['rmse'] for name, metrics in results.items()}
        colors = ['#2ecc71' if name == best_model_name else '#e74c3c' for name in rmse_scores.keys()]
        ax.barh(list(rmse_scores.keys()), list(rmse_scores.values()), color=colors)
        ax.set_xlabel("RMSE ($K)")
        ax.set_title("Model RMSE Comparison")
        st.pyplot(fig)
        plt.close(fig)
    
    # Prediction vs Actual for best model
    st.subheader(f"🎯 {best_model_name}: Predicted vs Actual")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    y_pred_best = results[best_model_name]['predictions']
    
    # Scatter plot
    ax1.scatter(y_test, y_pred_best, alpha=0.5, color='#3498db')
    ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
             'r--', lw=2, label='Perfect Prediction')
    ax1.set_xlabel("Actual Salary ($K)")
    ax1.set_ylabel("Predicted Salary ($K)")
    ax1.set_title("Prediction Scatter Plot")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Residual plot
    residuals = y_test - y_pred_best
    ax2.scatter(y_pred_best, residuals, alpha=0.5, color='#e74c3c')
    ax2.axhline(y=0, color='black', linestyle='--', lw=2)
    ax2.set_xlabel("Predicted Salary ($K)")
    ax2.set_ylabel("Residuals")
    ax2.set_title("Residual Plot")
    ax2.grid(True, alpha=0.3)
    
    st.pyplot(fig)
    plt.close(fig)
    
    return best_model_name, results[best_model_name]['model']

def display_feature_importance(model, feature_names, model_name):
    """Display feature importance for tree-based models"""
    
    if hasattr(model, 'feature_importances_'):
        st.subheader("🔍 Feature Importance Analysis")
        
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        top_n = min(20, len(feature_names))
        top_features = [feature_names[i] for i in indices[:top_n]]
        top_importances = importances[indices[:top_n]]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        colors = plt.cm.viridis(np.linspace(0, 1, top_n))
        ax.barh(range(top_n), top_importances, color=colors)
        ax.set_yticks(range(top_n))
        ax.set_yticklabels(top_features)
        ax.invert_yaxis()
        ax.set_xlabel("Importance")
        ax.set_title(f"Top {top_n} Most Important Features - {model_name}")
        st.pyplot(fig)
        plt.close(fig)
        
        # Feature importance table
        importance_df = pd.DataFrame({
            'Feature': top_features,
            'Importance': top_importances
        }).sort_values('Importance', ascending=False)
        
        st.dataframe(importance_df, use_container_width=True)

# ============================================================================
# PREDICTION INTERFACE
# ============================================================================

def create_prediction_interface(model, feature_names, df):
    """Interactive prediction interface"""
    
    st.header("🎯 Salary Prediction Tool")
    st.write("Enter job details to predict salary:")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        rating = st.slider("Company Rating", 0.0, 5.0, 3.5, 0.1)
        age = st.number_input("Company Age (years)", 0, 150, 20)
        seniority = st.selectbox("Seniority Level", ['junior', 'mid', 'senior'])
        job_type = st.selectbox("Job Type", ['data_scientist', 'data_engineer', 'ml_engineer', 'analyst', 'manager'])
    
    with col2:
        size = st.selectbox("Company Size", df['Size'].unique())
        ownership = st.selectbox("Type of Ownership", df['Type of ownership'].unique())
        revenue = st.selectbox("Revenue", df['Revenue'].unique())
    
    with col3:
        python = st.checkbox("Python", value=True)
        sql = st.checkbox("SQL", value=True)
        aws = st.checkbox("AWS", value=False)
        spark = st.checkbox("Spark", value=False)
        is_remote = st.checkbox("Remote Position", value=False)
    
    if st.button("🔮 Predict Salary", type="primary"):
        # Create input dataframe
        input_data = pd.DataFrame([{
            'Rating': rating,
            'age': age,
            'python_yn': 1 if python else 0,
            'sql_yn': 1 if sql else 0,
            'aws_yn': 1 if aws else 0,
            'spark_yn': 1 if spark else 0,
            'is_remote': 1 if is_remote else 0,
            'total_skills': sum([python, sql, aws, spark]),
            'Size': size,
            'Type of ownership': ownership,
            'Revenue': revenue,
            'seniority': seniority,
            'job_simp': job_type
        }])
        
        # Encode categorical variables
        input_encoded = pd.get_dummies(input_data)
        
        # Align with training features
        for col in feature_names:
            if col not in input_encoded.columns:
                input_encoded[col] = 0
        
        input_encoded = input_encoded[feature_names]
        
        # Predict
        prediction = model.predict(input_encoded)[0]
        
        st.success(f"### Predicted Salary: ${prediction:.2f}K")
        st.info(f"Estimated Range: ${prediction*0.9:.2f}K - ${prediction*1.1:.2f}K")

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    st.set_page_config(
        page_title="DS Salary Predictor Pro",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main { padding: 0rem 1rem; }
    .stTabs [data-baseweb="tab-list"] { gap: 2rem; }
    .stTabs [data-baseweb="tab"] { padding: 1rem 2rem; }
    </style>
    """, unsafe_allow_html=True)
    
    # Title
    st.title("📊 Data Science Salary Predictor Pro")
    st.markdown("### End-to-End ML Pipeline for Salary Analysis & Prediction")
    
    # Sidebar
    st.sidebar.header("⚙️ Configuration")
    
    # File upload
    uploaded_file = st.sidebar.file_uploader("Upload Dataset (CSV)", type=['csv'])
    
    if uploaded_file is not None:
        df = load_and_clean_data(uploaded_file)
    else:
        st.sidebar.info("Upload a CSV file or use default path")
        default_path = st.sidebar.text_input(
            "Default File Path",
            value='/mnt/data/glassdoor_jobs+(1).csv'
        )
        if st.sidebar.button("Load Data"):
            df = load_and_clean_data(default_path)
        else:
            df = None
    
    if df is not None:
        # Navigation
        page = st.sidebar.radio(
            "📑 Navigation",
            ["🏠 Overview", "📊 EDA", "🤖 Model Training", "🎯 Predictions", "📥 Export"]
        )
        
        # Overview Page
        if page == "🏠 Overview":
            st.header("Project Overview")
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("📄 Total Records", f"{len(df):,}")
            col2.metric("💼 Features", df.shape[1])
            col3.metric("💰 Avg Salary", f"${df['avg_salary'].mean():.0f}K")
            col4.metric("🏢 Companies", df['company_txt'].nunique())
            
            st.subheader("Dataset Sample")
            st.dataframe(df.head(20), use_container_width=True)
            
            st.subheader("Column Information")
            col_info = pd.DataFrame({
                'Column': df.columns,
                'Type': df.dtypes.astype(str),
                'Non-Null': df.count().values,
                'Null': df.isnull().sum().values,
                'Unique': df.nunique().values
            })
            st.dataframe(col_info, use_container_width=True)
        
        # EDA Page
        elif page == "📊 EDA":
            perform_eda(df)
        
        # Model Training Page
        elif page == "🤖 Model Training":
            st.header("🤖 Model Training & Evaluation")
            
            # Prepare data
            with st.spinner("Preparing data for modeling..."):
                X, y, feature_names = prepare_model_data(df)
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=Config.TEST_SIZE, random_state=Config.RANDOM_STATE
                )
            
            st.success(f"✅ Data prepared: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")
            st.info(f"📊 Number of features: {X.shape[1]}")
            
            # Train models
            if st.button("🚀 Train All Models", type="primary"):
                results = train_models(X_train, X_test, y_train, y_test)
                
                # Display results
                best_model_name, best_model = display_model_results(results, y_test)
                
                # Feature importance
                display_feature_importance(best_model, X.columns.tolist(), best_model_name)
                
                # Save to session state
                st.session_state['best_model'] = best_model
                st.session_state['feature_names'] = X.columns.tolist()
                st.session_state['all_results'] = results
        
        # Predictions Page
        elif page == "🎯 Predictions":
            if 'best_model' in st.session_state:
                create_prediction_interface(
                    st.session_state['best_model'],
                    st.session_state['feature_names'],
                    df
                )
            else:
                st.warning("⚠️ Please train models first in the Model Training section")
        
        # Export Page
        elif page == "📥 Export":
            st.header("📥 Export Models & Reports")
            
            if 'best_model' in st.session_state:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Download Trained Model")
                    model_bytes = pickle.dumps(st.session_state['best_model'])
                    st.download_button(
                        label="📦 Download Model (.pkl)",
                        data=model_bytes,
                        file_name=f"salary_model_{datetime.now().strftime('%Y%m%d')}.pkl",
                        mime="application/octet-stream"
                    )
                
                with col2:
                    st.subheader("Download Processed Data")
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📊 Download Cleaned Data (.csv)",
                        data=csv,
                        file_name=f"processed_data_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
                
                st.subheader("Model Performance Report")
                if 'all_results' in st.session_state:
                    report_data = []
                    for name, metrics in st.session_state['all_results'].items():
                        report_data.append({
                            'Model': name,
                            'MAE': metrics['mae'],
                            'RMSE': metrics['rmse'],
                            'R2_Score': metrics['r2'],
                            'CV_Mean': metrics['cv_mean'],
                            'CV_Std': metrics['cv_std']
                        })
                    
                    report_df = pd.DataFrame(report_data)
                    report_csv = report_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📄 Download Performance Report (.csv)",
                        data=report_csv,
                        file_name=f"model_report_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
                    
                    st.dataframe(report_df, use_container_width=True)
            else:
                st.warning("⚠️ No trained models available. Please train models first.")
    
    else:
        st.info("👆 Please upload a dataset or specify a file path to begin")
        
        # Display sample usage
        st.markdown("""
        ### 📋 Getting Started
        
        1. **Upload Data**: Use the sidebar to upload your CSV file
        2. **Explore Data**: Navigate to EDA section for insights
        3. **Train Models**: Go to Model Training to build predictive models
        4. **Make Predictions**: Use the Predictions tool for salary estimates
        5. **Export Results**: Download models and reports
        
        ### 📊 Expected CSV Format
        Your dataset should contain columns like:
        - `Salary Estimate`: Salary range (e.g., "$80K-$120K")
        - `Job Title`: Position title
        - `Company Name`: Employer name
        - `Location`: Job location
        - `Rating`: Company rating
        - `Job Description`: Full job description
        - `Size`: Company size
        - `Founded`: Year founded
        - And more...
        """)

# ============================================================================
# ADVANCED FEATURES: HYPERPARAMETER TUNING
# ============================================================================

def hyperparameter_tuning(X_train, y_train, model_type='RandomForest'):
    """Perform hyperparameter tuning with GridSearchCV"""
    
    st.subheader("🔧 Hyperparameter Tuning")
    
    if model_type == 'RandomForest':
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 15, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        model = RandomForestRegressor(random_state=Config.RANDOM_STATE)
    
    elif model_type == 'GradientBoosting':
        param_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'min_samples_split': [2, 5, 10]
        }
        model = GradientBoostingRegressor(random_state=Config.RANDOM_STATE)
    
    with st.spinner(f"Tuning {model_type}... This may take a while."):
        grid_search = GridSearchCV(
            model, param_grid, cv=3, scoring='r2', n_jobs=-1, verbose=1
        )
        grid_search.fit(X_train, y_train)
    
    st.success(f"✅ Best Parameters: {grid_search.best_params_}")
    st.info(f"📊 Best CV Score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

# ============================================================================
# RUN APPLICATION
# ============================================================================

if __name__ == "__main__":
    main()
