# Cynbit-Studio-Intelligent-AI-Coding-Mentor

## 1. Project Overview

Cynbit Studio – Intelligent AI Coding Mentor is an AI-powered coding education and interview practice proof of concept designed to support students and aspiring developers in improving their programming skills.

The application combines Python-based code analysis, a locally running Large Language Model (LLM), personalized exercise recommendations, coding interview practice, and learning progress tracking.

This project demonstrates how AI and machine learning could enhance the coding education experience through automated feedback, personalized learning, and structured practice.

## 2. Key Features

### 2.1 Interactive Dashboard
- Displays coding practice and learning progress.
- Provides an overview of student activity.
- Presents performance metrics and visualizations.

### 2.2 Python Code Analyzer
- Analyzes Python source code using the Abstract Syntax Tree (AST).
- Identifies functions, loops, conditional statements, classes, and imports.
- Provides structural insights into submitted code.

### 2.3 AI Coding Mentor
- Uses Ollama with the Qwen2.5-Coder 3B model.
- Explains Python code and programming concepts.
- Provides AI-assisted coding guidance and improvement suggestions.
- Runs the language model locally.

### 2.4 Coding Interview Practice
- Provides coding problems in a LeetCode-inspired interface.
- Includes problem descriptions and Python code editing.
- Supports sample test-case execution and result display.
- Provides AI-assisted code review.
- Tracks solved problems using SQLite.

### 2.5 Personalized Exercise Recommendations
- Uses TF-IDF vectorization and cosine similarity.
- Recommends relevant coding exercises based on available exercise data.
- Helps learners identify suitable practice topics.

### 2.6 Learning Progress Tracking
- Stores learning activity and solved problem records in SQLite.
- Tracks coding practice progress.
- Supports progress monitoring through the dashboard.

## 3. Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Frontend | Streamlit |
| AI Model | Qwen2.5-Coder 3B |
| Local LLM Runtime | Ollama |
| Code Analysis | Python AST |
| Machine Learning | Scikit-learn |
| Recommendation System | TF-IDF, Cosine Similarity |
| Database | SQLite |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn, Plotly |
| API Communication | Requests |
| Model/Data Storage | Joblib |

## 4. Project Structure

```text
CYNBIT_project/
│
├── app.py
├── requirements.txt
├── README.md
│
├── exercise_catalog.csv
├── vectorizer.joblib
├── exercise_vectors.joblib
├── project_config.json
│
└── coding_mentor.db
```

**File descriptions:**

- `app.py` – Main Streamlit application.
- `requirements.txt` – Python dependencies.
- `exercise_catalog.csv` – Coding exercise dataset.
- `vectorizer.joblib` – Saved TF-IDF vectorizer.
- `exercise_vectors.joblib` – Saved exercise feature vectors.
- `project_config.json` – Optional project configuration.
- `coding_mentor.db` – SQLite database for progress and solved problems.

The database and generated files may be created or updated when the application runs.

## 5. Installation and Setup

### Prerequisites

- Python 3.9 or later.
- Ollama installed.
- At least 8 GB RAM recommended for this local prototype.
- Internet connection for initial package and model downloads.

### Step 1: Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd CYNBIT_project
```

Replace `YOUR_GITHUB_REPOSITORY_URL` with your actual repository URL.

### Step 2: Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Install and Configure Ollama

Download Ollama:

https://ollama.com/download

Pull the coding model:

```bash
ollama pull qwen2.5-coder:3b
```

Start the model:

```bash
ollama run qwen2.5-coder:3b
```

Keep Ollama running while using the AI Coding Mentor.

### Step 5: Run the Application

Open a terminal in the project directory and execute:

```bash
streamlit run app.py
```

The application will be available at:

- Local URL: `http://localhost:8501`
- Network URL: `http://192.168.1.79:8501`

**Access note:** These are runtime network addresses, not permanent public URLs. The local URL works on the host machine while Streamlit is running. The network URL generally requires access to the same local network and may change.

## 6. System Architecture

The application follows a modular architecture:

1. **User Interface:** Streamlit provides the interactive dashboard and application pages.
2. **Code Analysis:** Python AST parses submitted source code and extracts structural information.
3. **AI Assistance:** Ollama communicates with the locally running Qwen2.5-Coder model to generate coding explanations and feedback.
4. **Recommendation Engine:** TF-IDF and cosine similarity identify relevant exercises from the available dataset.
5. **Database Layer:** SQLite stores learning progress and solved problem records.
6. **Visualization:** Data visualization libraries display learning and performance information.

## 7. How the Application Works

1. The user opens the Streamlit application.
2. The dashboard displays available learning and coding progress information.
3. The user submits Python code for structural analysis.
4. The AI Coding Mentor provides explanations and suggestions using the local language model.
5. The user practices coding interview problems and executes sample test cases.
6. The recommendation engine suggests relevant exercises.
7. Learning activity and solved problems are stored in SQLite.

## 8. Current Limitations

- The coding interview question bank currently contains a limited set of problems, including Two Sum, Valid Anagram, and Binary Search.
- Test cases are locally configured and are not hidden assessment tests.
- AST-based analysis identifies code structure but does not independently prove algorithmic correctness.
- AI-generated feedback may occasionally be inaccurate and should be reviewed.
- Exercise recommendations depend on the available exercise dataset.
- This implementation is a proof of concept, not a production-grade secure online coding judge.

## 9. Future Enhancements

- Expand the coding interview question bank.
- Add more programming languages.
- Introduce difficulty-based learning paths.
- Improve personalized recommendations using learner performance.
- Add more detailed code quality and complexity analysis.
- Introduce instructor and administrator dashboards.
- Improve evaluation and progress analytics.

## 10. Assessment Context

This project was developed as a proof of concept for the AIML Engineer Intern assessment.

**Proposed AI opportunity:** Enhance the coding education experience through AI-powered code analysis, personalized exercise recommendations, coding explanations, and interview practice.

The project demonstrates a possible extension to Cynbit Studio. It does not claim that these prototype features are already part of Cynbit's production platform.

## 11. Application Access

When the application is running on the development machine:

```text
Local URL: http://localhost:8501
Network URL: http://192.168.1.79:8501
```

