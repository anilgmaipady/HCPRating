# HCP Rating System - Usage Guide

## Introduction

Welcome to the HCP Rating System! This guide will walk you through using the application, from scoring single transcripts to providing feedback for model improvement.

## 🚀 Quick Start

1.  **Start the application**:
    ```bash
    python run.py
    ```
2.  **Open the web interface**: Navigate to `http://localhost:8501` in your browser.

## 🖥️ Using the Web Interface

### Single Transcript Scoring

This is the primary feature for evaluating a single telehealth session.

1.  **Navigate to the Page**: Select "Single Transcript" from the sidebar navigation.

2.  **Generate or Enter a Transcript**:
    *   Click the **"🎲 Generate Random Transcript"** button to instantly populate the text area with a realistic example for quick testing.
    *   Alternatively, you can paste or type your own transcript into the text area.

3.  **Enter HCP Information (Optional)**:
    *   You can add the Healthcare Provider's Name and the Session Date for tracking purposes.

4.  **Score the Transcript**:
    *   Click the **"Score Transcript"** button.
    *   The system will analyze the text and display a detailed evaluation, including scores, charts, and textual analysis.

### Providing Feedback on Results

To continuously improve the model, you can provide feedback on any evaluation you believe is incorrect.

1.  **Locate the Feedback Section**: After a transcript is scored, scroll to the bottom of the results page to find the **"📝 Provide Feedback"** section.

2.  **Open the Correction Form**: Click on the expander `"Click here to correct this evaluation"` to open the feedback form.

3.  **Correct the Evaluation**:
    *   **Scores**: Adjust the 1-5 scores for each dimension (`Empathy`, `Clarity`, etc.).
    *   **Text Analysis**: Edit the text for "Corrected Reasoning", "Corrected Strengths", and "Corrected Areas for Improvement".

4.  **Submit Your Feedback**: Click the **"Submit Feedback"** button. Your correction will be saved and used in future training cycles to improve the model's accuracy.

### Batch Processing & CSV Upload

For evaluating multiple transcripts at once:

1.  **Navigate**: Go to the "Batch Processing" or "Upload CSV" page from the sidebar.
2.  **Add Transcripts**:
    *   On the "Batch Processing" page, add transcripts one by one using the form.
    *   On the "Upload CSV" page, upload a file with a `transcript` column. You can use `data/demo_transcripts.csv` as a template.
3.  **Process**: Click the "Process Batch" or "Process CSV" button to get a summary report for all transcripts.

## ⚙️ Using the Command Line Interface (CLI)

The CLI is useful for scripting and programmatic access.

1.  **Score a Single Transcript**:
    ```bash
    python src/cli.py score "HCP: Hello, how are you feeling today? Patient: I'm struggling..."
    ```

2.  **Process a Batch File**:
    ```bash
    python src/cli.py batch --input-file data/sample_transcripts.csv
    ```

3.  **See all options**:
    ```bash
    python src/cli.py --help
    ```

## 🔌 Using the API

For integration with other services, the API offers powerful endpoints.

1.  **Start the API Server**:
    ```bash
    python src/deployment/start_server.py
    ```
2.  **Access API Documentation**: Open `http://localhost:8000/docs` in your browser for an interactive API specification.

### Example: Score a Transcript via API

```bash
curl -X POST "http://localhost:8000/score" \
-H "Content-Type: application/json" \
-d '{
  "transcript": "HCP: Hello, how are you feeling today? Patient: I am not feeling well.",
  "hcp_name": "Dr. Feelgood"
}'
```

---

This guide covers the primary uses of the HCP Rating System. For details on how to retrain the model with the feedback you've collected, please see the [**Training Guide**](TRAINING_GUIDE.md). 