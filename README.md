# FinGPT-Advisor

FinGPT-Advisor is a full-stack web application that leverages generative AI to provide easy-to-understand financial insights. Users can input financial questions or data, and the application, powered by Google's Gemini AI, will generate simplified summaries, analyses, and recommendations.

**🚀 Live Demo:** [https://fingpt-advisor.onrender.com](https://fingpt-advisor.onrender.com)

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [License](#license)

## Project Overview
FinGPT-Advisor is a web-based financial analysis tool designed to provide concise, data-driven insights into stock market performance. It acts as an intelligent financial advisor by integrating multiple Large Language Models (LLMs) to analyze real-time and historical data from the `yfinance` library. The application leverages a flexible orchestration architecture, allowing it to select and utilize the best available LLM to answer a user's query, providing a robust and resilient service.

## Key Features
* **Multi-LLM Integration:** Seamlessly uses a variety of LLMs (Gemini, Groq, Deepseek) to answer user prompts, ensuring high-quality and reliable responses.
* **Real-time Financial Data:** Fetches up-to-date and historical stock data from Yahoo Finance, including prices, P/E ratios, market caps, and more.
* **Dynamic Data Extraction:** Automatically identifies stock tickers, data types (e.g., historical data, balance sheets), and time periods from natural language prompts.
* **Intelligent Response Orchestration:** Evaluates responses from multiple LLMs and selects the most comprehensive or best-suited answer.
* **User-friendly Interface:** Provides a clean and simple web interface for users to submit queries and view responses.
* **Database Logging:** Logs all user prompts and AI responses to a local SQLite database for historical tracking and analysis.

## Core Technologies
* **Python:** The primary programming language for the backend logic.
* **Flask:** A lightweight web framework to serve the application and handle API requests.
* **Google Gemini API:** Integrated for powerful, large-scale language model capabilities.
* **Groq API:** Used for high-speed inference of open-source models like Llama.
* **Deepseek API:** Integrated for its unique language model, accessible via an OpenAI-compatible client.
* **yfinance:** A Python library to fetch financial market data from Yahoo Finance.
* **sqlite3:** The standard Python library for interacting with a local SQLite database.
* **Deployment**: Render

### Prerequisites

Before you begin, ensure you have the following installed:

* Python 3.8+
* Google Gemini API Key
* Groq API Key
* Deepseek API Key
* OpenAI API Key 

### Installation

1.  **Set up your Python environment and install dependencies:**

    **Option A: Using `venv` (Recommended for most users)**
    ```bash
    # Create a virtual environment
    python -m venv venv

    # Activate the virtual environment
    # On macOS/Linux:
    source venv/bin/activate
    # On Windows:
    # .\venv\Scripts\activate

    # Install dependencies from requirements.txt
    pip install -r requirements.txt
    ```

    **Option B: Using `conda` (Alternative)**
    ```bash
    # Create the conda environment from the file
    conda env create -f environment.yml

    # Activate the new environment
    conda activate fingpt-advisor
    ```

2.  **Set your API keys as environment variables:**
    ```bash
    export GOOGLE_API_KEY="YOUR_GEMINI_API_KEY_HERE"
    export GROQ_API_KEY="YOUR_GROQ_API_KEY_HERE"
    export DEEPSEEK_API_KEY="YOUR_DEEPSEEK_API_KEY_HERE"
    export OPENAI_API_KEY="YOUR_OPENAI_API_KEY_HERE"
    ```

## Usage

1.  **Ensure your virtual/conda environment is activated.**

2.  **Run the Flask application:**
    ```bash
    python app.py
    ```

3.  Open your web browser and navigate to `http://127.0.0.1:5000`.

4.  Enter your financial query in the input box and click "Ask" to receive a response from the AI.

## License

This project is open-sourced under the MIT License. Please refer to **[LICENSE](/LICENSE.md)** for more information.