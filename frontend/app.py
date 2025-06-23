#!/usr/bin/env python3
"""
Streamlit Frontend for HCP Rating System
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
import random

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Page configuration
st.set_page_config(
    page_title="HCP Rating System",
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
    .backend-info {
        background-color: #e7f3ff;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #007bff;
        margin-bottom: 1rem;
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

def check_ollama_availability():
    """Check if Ollama is available."""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        return response.status_code == 200
    except:
        return False

def get_hcp_scorer():
    """Get HCP Scorer instance for direct use."""
    try:
        from src.inference.hcp_scorer import HCPScorer
        return HCPScorer()
    except Exception as e:
        st.error(f"Failed to initialize HCP Scorer: {e}")
        return None

def score_transcript(transcript, hcp_name=None, session_date=None):
    """Score a single transcript via API or direct Ollama."""
    # Try API first
    try:
        payload = {
            "transcript": transcript,
            "hcp_name": hcp_name,
            "session_date": session_date
        }
        response = requests.post(f"{API_BASE_URL}/score", json=payload, timeout=30)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        # Fallback to direct Ollama
        st.warning("API server not available, using Ollama directly...")
        return score_transcript_direct(transcript, hcp_name, session_date)

def score_transcript_direct(transcript, hcp_name=None, session_date=None):
    """Score transcript using direct backend."""
    hcp_scorer = get_hcp_scorer()
    if not hcp_scorer:
        return None
    
    try:
        result = hcp_scorer.score_transcript(transcript, hcp_name)
        return {
            "hcp_name": hcp_name,
            "session_date": session_date,
            "scores": {
                'empathy': result.empathy,
                'clarity': result.clarity,
                'accuracy': result.accuracy,
                'professionalism': result.professionalism
            },
            "overall_score": result.overall_score,
            "confidence": result.confidence,
            "reasoning": result.reasoning,
            "strengths": result.strengths,
            "areas_for_improvement": result.areas_for_improvement,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        st.error(f"Scoring Error: {e}")
        return None

def score_batch_transcripts(transcripts_data):
    """Score multiple transcripts via API or direct Ollama."""
    # Try API first
    try:
        payload = {"transcripts": transcripts_data}
        response = requests.post(f"{API_BASE_URL}/score/batch", json=payload, timeout=60)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        # Fallback to direct Ollama
        st.warning("API server not available, using Ollama directly...")
        return score_batch_transcripts_direct(transcripts_data)

def score_batch_transcripts_direct(transcripts_data):
    """Score multiple transcripts directly using HCP Scorer."""
    hcp_scorer = get_hcp_scorer()
    if not hcp_scorer:
        return None
    
    try:
        # Prepare transcripts for batch processing
        transcripts = [(t["transcript"], t.get("hcp_name")) for t in transcripts_data]
        
        # Score transcripts
        results = hcp_scorer.batch_score_transcripts(transcripts)
        
        # Convert to API response format
        responses = []
        for i, (request_item, result) in enumerate(zip(transcripts_data, results)):
            response = {
                "hcp_name": request_item.get("hcp_name"),
                "session_date": request_item.get("session_date"),
                "scores": {
                    'empathy': result.empathy,
                    'clarity': result.clarity,
                    'accuracy': result.accuracy,
                    'professionalism': result.professionalism
                },
                "overall_score": result.overall_score,
                "confidence": result.confidence,
                "reasoning": result.reasoning,
                "strengths": result.strengths,
                "areas_for_improvement": result.areas_for_improvement,
                "timestamp": datetime.now().isoformat()
            }
            responses.append(response)
        
        return responses
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
        name='HCP Scores',
        line_color='#1f77b4'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 5]
            )),
        showlegend=False,
        title="HCP Performance Radar Chart"
    )
    
    return fig

def create_bar_chart(scores):
    """Create a bar chart for scores."""
    fig = px.bar(
        x=list(scores.keys()),
        y=list(scores.values()),
        title="HCP Performance Scores",
        labels={'x': 'Dimension', 'y': 'Score'},
        color=list(scores.values()),
        color_continuous_scale='Blues'
    )
    fig.update_layout(yaxis_range=[0, 5])
    return fig

def main():
    """Main application function."""
    
    # Header
    st.markdown('<h1 class="main-header">🥗 HCP Rating System</h1>', unsafe_allow_html=True)
    
    # Check backend availability
    api_healthy = check_api_health()
    ollama_available = check_ollama_availability()
    
    # Show backend status
    if api_healthy:
        st.success("✅ API server is running")
    elif ollama_available:
        st.info("ℹ️ Using Ollama backend directly (API server not available)")
    else:
        st.error("❌ No backend available. Please start Ollama or the API server.")
        st.info("""
        **To start Ollama:**
        1. Install Ollama: Visit https://ollama.ai or run `brew install ollama`
        2. Start Ollama: `ollama serve`
        3. Pull model: `ollama pull mistral`
        
        **To start API server:**
        Run: `python src/deployment/start_server.py`
        """)
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
    
    # Sample transcripts for random generation
    sample_transcripts = [
        "Patient: I'm really worried about my test results.\nHCP: I can see this is causing you a lot of anxiety. Let me explain what these results mean and what our next steps will be. We're going to work through this together.\nPatient: Thank you, that makes me feel better.\nHCP: You're welcome. Remember, I'm here to support you throughout this process. Do you have any questions about what we discussed?",
        "Patient: My back has been hurting for a week.\nHCP: Let me examine you. Can you describe the pain?\nPatient: It's sharp and gets worse when I move.\nHCP: I'll order some tests and prescribe pain medication. Come back in a week.",
        "Patient: I'm feeling very depressed lately.\nHCP: Take these pills. Next patient.\nPatient: But I have questions about side effects.\nHCP: Read the label. Next!",
        "Patient: I have a headache.\nHCP: How long have you had it? Can you describe the pain?\nPatient: About two days, it's a dull ache.\nHCP: Let me check your blood pressure and ask about any recent changes in your routine.",
        "Patient: I'm concerned about my medication.\nHCP: I understand your concern. Let's review your current treatment and discuss any side effects you're experiencing.\nPatient: I've been feeling dizzy.\nHCP: That's important to know. Let me adjust your dosage and monitor your response.",
        "Patient: My symptoms are getting worse.\nHCP: I need to examine you more thoroughly. Let's run some additional tests to understand what's happening.\nPatient: What do you think it could be?\nHCP: I want to rule out several possibilities. The tests will help us determine the best treatment approach.",
        "Patient: I've been having trouble sleeping.\nHCP: I understand how frustrating that can be. Can you tell me more about your sleep patterns and what might be contributing to this?\nPatient: I'm stressed about work.\nHCP: Stress can definitely impact sleep. Let's discuss some strategies to help you relax and establish a better sleep routine.",
        "Patient: I think I have an infection.\nHCP: Let me examine the area and ask about your symptoms.\nPatient: It's red and swollen.\nHCP: I can see the inflammation. Let me prescribe an antibiotic and give you instructions for care.",
        "Patient: I'm worried about my child's fever.\nHCP: I understand your concern. Let me check their temperature and symptoms.\nPatient: It's been high for two days.\nHCP: Let me examine them thoroughly and determine if we need to run any tests.",
        "Patient: My medication isn't working.\nHCP: I'm sorry to hear that. Let's review your current treatment and see what adjustments we can make.\nPatient: I'm still in pain.\nHCP: Let me explore alternative options and work with you to find a solution that provides relief."
    ]

    # Session state for transcript text
    if 'transcript_text' not in st.session_state:
        st.session_state.transcript_text = ""

    # Button to generate a random transcript
    if st.button("🎲 Generate Random Transcript"):
        st.session_state.transcript_text = random.choice(sample_transcripts)

    # Input form
    with st.form("transcript_form"):
        hcp_name = st.text_input("Healthcare Provider Name (Optional)")
        session_date = st.date_input("Session Date (Optional)")
        
        transcript = st.text_area(
            "Telehealth Session Transcript",
            height=300,
            value=st.session_state.transcript_text,
            placeholder="Paste the telehealth session transcript here..."
        )
        
        submitted = st.form_submit_button("Score Transcript", type="primary")
    
    if submitted and transcript.strip():
        with st.spinner("Analyzing transcript..."):
            result = score_transcript(
                transcript, 
                hcp_name if hcp_name else None,
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
            hcp_name = st.text_input("HCP Name")
            session_date = st.date_input("Session Date")
            transcript = st.text_area("Transcript", height=200)
            
            if st.form_submit_button("Add to Batch"):
                if transcript.strip():
                    st.session_state.transcripts.append({
                        "hcp_name": hcp_name,
                        "session_date": session_date.isoformat() if session_date else None,
                        "transcript": transcript
                    })
                    st.success("Transcript added to batch!")
                    st.rerun()
    
    # Display current batch
    if st.session_state.transcripts:
        st.subheader(f"Current Batch ({len(st.session_state.transcripts)} transcripts)")
        
        for i, transcript_data in enumerate(st.session_state.transcripts):
            with st.expander(f"Transcript {i+1}: {transcript_data['hcp_name'] or 'Unnamed HCP'}"):
                st.write(f"**HCP Name:** {transcript_data['hcp_name'] or 'Not specified'}")
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
    - Optional columns: `hcp_name`, `session_date`
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
                                "hcp_name": row.get('hcp_name'),
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
                file_name=f"hcp_score_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    with col6:
        if st.button("Export as CSV"):
            df = pd.DataFrame([result])
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"hcp_score_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

def display_batch_results(results):
    """Display results for batch processing."""
    st.success(f"✅ Successfully processed {len(results)} transcripts!")
    
    # Create summary dataframe
    summary_data = []
    for result in results:
        summary_data.append({
            'HCP Name': result['hcp_name'] or 'Unknown',
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
            st.write(f"{performer['rank']}. {performer['hcp_name']} - Score: {performer['score']:.2f}")

if __name__ == "__main__":
    main() 