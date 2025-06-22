#!/usr/bin/env python3
"""
Streamlit Frontend for RD Rating System
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import json
from datetime import datetime
import yaml
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Page configuration
st.set_page_config(
    page_title="RD Rating System",
    page_icon="🥗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load configuration
def load_config():
    config_path = project_root / "configs" / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

config = load_config()

# API configuration
API_BASE_URL = "http://localhost:8000"

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .score-display {
        font-size: 2rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
    }
    .strength-item {
        background-color: #d4edda;
        padding: 0.5rem;
        border-radius: 0.25rem;
        margin: 0.25rem 0;
    }
    .improvement-item {
        background-color: #f8d7da;
        padding: 0.5rem;
        border-radius: 0.25rem;
        margin: 0.25rem 0;
    }
</style>
""", unsafe_allow_html=True)

def check_api_health():
    """Check if the API is running."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def score_transcript(transcript, rd_name=None, session_date=None):
    """Score a single transcript via API."""
    try:
        payload = {
            "transcript": transcript,
            "rd_name": rd_name,
            "session_date": session_date
        }
        response = requests.post(f"{API_BASE_URL}/score", json=payload, timeout=30)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error scoring transcript: {str(e)}")
        return None

def score_batch_transcripts(transcripts_data):
    """Score multiple transcripts via API."""
    try:
        payload = {"transcripts": transcripts_data}
        response = requests.post(f"{API_BASE_URL}/score/batch", json=payload, timeout=60)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error in batch scoring: {str(e)}")
        return None

def create_radar_chart(scores):
    """Create a radar chart for scores."""
    categories = list(scores.keys())
    values = list(scores.values())
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='RD Scores',
        line_color='#1f77b4'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 5]
            )),
        showlegend=False,
        title="RD Performance Radar Chart"
    )
    
    return fig

def create_bar_chart(scores):
    """Create a bar chart for scores."""
    fig = px.bar(
        x=list(scores.keys()),
        y=list(scores.values()),
        title="RD Performance Scores",
        labels={'x': 'Dimension', 'y': 'Score'},
        color=list(scores.values()),
        color_continuous_scale='Blues'
    )
    fig.update_layout(yaxis_range=[0, 5])
    return fig

def main():
    """Main application function."""
    
    # Header
    st.markdown('<h1 class="main-header">🥗 RD Rating System</h1>', unsafe_allow_html=True)
    
    # Check API health
    api_healthy = check_api_health()
    
    if not api_healthy:
        st.error("⚠️ API server is not running. Please start the vLLM server first.")
        st.info("Run: `python src/deployment/start_server.py`")
        return
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Single Transcript", "Batch Processing", "Upload CSV", "Reports", "Configuration"]
    )
    
    if page == "Single Transcript":
        show_single_transcript_page()
    elif page == "Batch Processing":
        show_batch_processing_page()
    elif page == "Upload CSV":
        show_csv_upload_page()
    elif page == "Reports":
        show_reports_page()
    elif page == "Configuration":
        show_configuration_page()

def show_single_transcript_page():
    """Single transcript scoring page."""
    st.header("📝 Single Transcript Scoring")
    
    # Input form
    with st.form("transcript_form"):
        rd_name = st.text_input("Registered Dietitian Name (Optional)")
        session_date = st.date_input("Session Date (Optional)")
        
        transcript = st.text_area(
            "Telehealth Session Transcript",
            height=300,
            placeholder="Paste the telehealth session transcript here..."
        )
        
        submitted = st.form_submit_button("Score Transcript", type="primary")
    
    if submitted and transcript.strip():
        with st.spinner("Analyzing transcript..."):
            result = score_transcript(
                transcript, 
                rd_name if rd_name else None,
                session_date.isoformat() if session_date else None
            )
        
        if result:
            display_scoring_results(result)

def show_batch_processing_page():
    """Batch processing page."""
    st.header("📊 Batch Transcript Processing")
    
    st.info("Enter multiple transcripts for batch processing.")
    
    # Initialize session state for transcripts
    if 'transcripts' not in st.session_state:
        st.session_state.transcripts = []
    
    # Add new transcript
    with st.expander("Add New Transcript", expanded=True):
        with st.form("batch_transcript_form"):
            rd_name = st.text_input("RD Name")
            session_date = st.date_input("Session Date")
            transcript = st.text_area("Transcript", height=200)
            
            if st.form_submit_button("Add to Batch"):
                if transcript.strip():
                    st.session_state.transcripts.append({
                        "rd_name": rd_name,
                        "session_date": session_date.isoformat() if session_date else None,
                        "transcript": transcript
                    })
                    st.success("Transcript added to batch!")
                    st.rerun()
    
    # Display current batch
    if st.session_state.transcripts:
        st.subheader(f"Current Batch ({len(st.session_state.transcripts)} transcripts)")
        
        for i, transcript_data in enumerate(st.session_state.transcripts):
            with st.expander(f"Transcript {i+1}: {transcript_data['rd_name'] or 'Unnamed RD'}"):
                st.write(f"**RD Name:** {transcript_data['rd_name'] or 'Not specified'}")
                st.write(f"**Session Date:** {transcript_data['session_date'] or 'Not specified'}")
                st.write(f"**Transcript:** {transcript_data['transcript'][:200]}...")
                
                if st.button(f"Remove Transcript {i+1}", key=f"remove_{i}"):
                    st.session_state.transcripts.pop(i)
                    st.rerun()
        
        # Process batch
        if st.button("Process Batch", type="primary"):
            with st.spinner("Processing batch..."):
                results = score_batch_transcripts(st.session_state.transcripts)
            
            if results:
                display_batch_results(results)

def show_csv_upload_page():
    """CSV upload page."""
    st.header("📁 Upload CSV File")
    
    st.info("Upload a CSV file with transcripts for batch processing.")
    st.markdown("""
    **CSV Format Requirements:**
    - Required column: `transcript`
    - Optional columns: `rd_name`, `session_date`
    """)
    
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="Upload a CSV file with transcript data"
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"File uploaded successfully! Found {len(df)} rows.")
            
            # Display preview
            st.subheader("Data Preview")
            st.dataframe(df.head())
            
            # Validate columns
            required_columns = ['transcript']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                st.error(f"Missing required columns: {missing_columns}")
            else:
                if st.button("Process CSV", type="primary"):
                    with st.spinner("Processing CSV file..."):
                        # Prepare data for API
                        transcripts_data = []
                        for _, row in df.iterrows():
                            transcript_data = {
                                "transcript": row['transcript'],
                                "rd_name": row.get('rd_name'),
                                "session_date": row.get('session_date')
                            }
                            transcripts_data.append(transcript_data)
                        
                        # Send to API
                        try:
                            response = requests.post(
                                f"{API_BASE_URL}/upload/csv",
                                files={"file": uploaded_file.getvalue()},
                                timeout=120
                            )
                            response.raise_for_status()
                            results = response.json()
                            
                            if results:
                                display_batch_results(results['results'])
                                
                                # Display summary report
                                if 'report' in results:
                                    display_summary_report(results['report'])
                                    
                        except Exception as e:
                            st.error(f"Error processing CSV: {str(e)}")
                            
        except Exception as e:
            st.error(f"Error reading CSV file: {str(e)}")

def show_reports_page():
    """Reports page."""
    st.header("📈 Reports & Analytics")
    
    st.info("This page will show historical reports and analytics.")
    st.warning("Reports functionality is under development.")

def show_configuration_page():
    """Configuration page."""
    st.header("⚙️ Configuration")
    
    st.subheader("Current Configuration")
    
    # Display scoring criteria
    st.write("**Scoring Criteria:**")
    scoring_criteria = config.get('scoring', {}).get('dimensions', {})
    
    for dimension, criteria in scoring_criteria.items():
        with st.expander(f"{dimension.title()} Criteria"):
            st.write(f"**Weight:** {criteria.get('weight', 0.25)}")
            st.write(f"**Description:** {criteria.get('description', '')}")
            st.write("**Criteria:**")
            for criterion in criteria.get('criteria', []):
                st.write(f"- {criterion}")
    
    # Display model configuration
    st.write("**Model Configuration:**")
    model_config = config.get('model', {})
    st.json(model_config)

def display_scoring_results(result):
    """Display scoring results for a single transcript."""
    st.success("✅ Transcript scored successfully!")
    
    # Create columns for layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📊 Performance Scores")
        
        # Create radar chart
        radar_fig = create_radar_chart(result['scores'])
        st.plotly_chart(radar_fig, use_container_width=True)
        
        # Create bar chart
        bar_fig = create_bar_chart(result['scores'])
        st.plotly_chart(bar_fig, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Overall Score")
        st.markdown(f'<div class="score-display">{result["overall_score"]}/5.0</div>', unsafe_allow_html=True)
        
        st.subheader("📈 Individual Scores")
        for dimension, score in result['scores'].items():
            st.metric(dimension.title(), f"{score}/5")
        
        st.subheader("🎯 Confidence")
        st.progress(result['confidence'])
        st.write(f"{result['confidence']:.1%}")
    
    # Detailed analysis
    st.subheader("📝 Detailed Analysis")
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.write("**Reasoning:**")
        st.write(result['reasoning'])
    
    with col4:
        st.write("**Strengths:**")
        for strength in result['strengths']:
            st.markdown(f'<div class="strength-item">✅ {strength}</div>', unsafe_allow_html=True)
        
        st.write("**Areas for Improvement:**")
        for area in result['areas_for_improvement']:
            st.markdown(f'<div class="improvement-item">🔧 {area}</div>', unsafe_allow_html=True)
    
    # Export options
    st.subheader("💾 Export Results")
    
    col5, col6, col7 = st.columns(3)
    
    with col5:
        if st.button("Export as JSON"):
            st.download_button(
                label="Download JSON",
                data=json.dumps(result, indent=2),
                file_name=f"rd_score_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    with col6:
        if st.button("Export as CSV"):
            df = pd.DataFrame([result])
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"rd_score_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

def display_batch_results(results):
    """Display results for batch processing."""
    st.success(f"✅ Successfully processed {len(results)} transcripts!")
    
    # Create summary dataframe
    summary_data = []
    for result in results:
        summary_data.append({
            'RD Name': result['rd_name'] or 'Unknown',
            'Overall Score': result['overall_score'],
            'Empathy': result['scores']['empathy'],
            'Clarity': result['scores']['clarity'],
            'Accuracy': result['scores']['accuracy'],
            'Professionalism': result['scores']['professionalism'],
            'Confidence': result['confidence']
        })
    
    df = pd.DataFrame(summary_data)
    
    # Display summary table
    st.subheader("📊 Batch Results Summary")
    st.dataframe(df, use_container_width=True)
    
    # Create summary charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Average scores by dimension
        avg_scores = {
            'Empathy': df['Empathy'].mean(),
            'Clarity': df['Clarity'].mean(),
            'Accuracy': df['Accuracy'].mean(),
            'Professionalism': df['Professionalism'].mean()
        }
        
        fig = px.bar(
            x=list(avg_scores.keys()),
            y=list(avg_scores.values()),
            title="Average Scores by Dimension",
            labels={'x': 'Dimension', 'y': 'Average Score'}
        )
        fig.update_layout(yaxis_range=[0, 5])
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Overall score distribution
        fig = px.histogram(
            df, 
            x='Overall Score',
            title="Overall Score Distribution",
            nbins=10
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Export batch results
    st.subheader("💾 Export Batch Results")
    
    col3, col4 = st.columns(2)
    
    with col3:
        if st.button("Export as CSV"):
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    with col4:
        if st.button("Export Full Results as JSON"):
            st.download_button(
                label="Download JSON",
                data=json.dumps(results, indent=2),
                file_name=f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

def display_summary_report(report):
    """Display summary report."""
    st.subheader("📋 Summary Report")
    
    summary = report.get('summary', {})
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Evaluations", summary.get('total_evaluations', 0))
    
    with col2:
        avg_overall = summary.get('average_scores', {}).get('overall', 0)
        st.metric("Average Overall Score", f"{avg_overall:.2f}")
    
    with col3:
        top_performers = summary.get('top_performers', [])
        if top_performers:
            best_score = top_performers[0]['score']
            st.metric("Best Score", f"{best_score:.2f}")
    
    # Top performers
    if summary.get('top_performers'):
        st.write("**🏆 Top Performers:**")
        for performer in summary['top_performers']:
            st.write(f"{performer['rank']}. {performer['rd_name']} - Score: {performer['score']:.2f}")

if __name__ == "__main__":
    main() 