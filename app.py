import os
import sqlite3
import pandas as pd
import yfinance as yf
import google.generativeai as genai
from openai import OpenAI
from groq import Groq
import re
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import io
import PyPDF2
from docx import Document

app = Flask(__name__)

@app.route('/healthcheck')
def healthcheck():
    return 'OK', 200
    
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

llm_clients = {}
if GOOGLE_API_KEY:
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        llm_clients['gemini'] = genai.GenerativeModel('gemini-1.5-pro')
    except Exception as e:
        print(f"Failed to configure Gemini API: {e}")
if OPENAI_API_KEY:
    try:
        llm_clients['openai'] = OpenAI(api_key=OPENAI_API_KEY)
    except Exception as e:
        print(f"Failed to configure OpenAI API: {e}")
if GROQ_API_KEY:
    try:
        llm_clients['groq'] = Groq(api_key=GROQ_API_KEY)
    except Exception as e:
        print(f"Failed to configure Groq API: {e}")
if DEEPSEEK_API_KEY:
    try:
        llm_clients['deepseek'] = OpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url="https://api.deepseek.com/v1"
        )
    except Exception as e:
        print(f"Failed to configure Deepseek API: {e}")

def create_db():
    conn = sqlite3.connect('fingpt_advisor.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS interactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_prompt TEXT NOT NULL,
            ai_response TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()
create_db()

def get_financial_data(ticker, data_type, period='1y'):
    """Fetches financial data for a given ticker using yfinance."""
    try:
        stock = yf.Ticker(ticker)
        data_context = ""
        info = stock.info
        long_name = info.get('longName', ticker)

        if data_type == 'historical':
            historical_data = stock.history(period=period)
            if historical_data.empty:
                return f"No historical data found for {ticker} over the last {period}."
            
            start_price = historical_data['Close'].iloc[0]
            end_price = historical_data['Close'].iloc[-1]
            roi = ((end_price - start_price) / start_price) * 100 if start_price != 0 else 0
            
            data_context = (
                f"Financial data for {long_name} (Ticker: {ticker}):\n"
                f"- Time Period: {period}\n"
                f"- Latest Price: ${info.get('currentPrice', end_price):.2f}\n"
                f"- Start Price: ${start_price:.2f}\n"
                f"- End Price: ${end_price:.2f}\n"
                f"- Return on Investment (ROI) over the period: {roi:.2f}%\n"
                f"- Market Cap: ${info.get('marketCap', 'N/A'):,}\n"
                f"- Industry: {info.get('industry', 'N/A')}\n"
            )
            
        elif data_type == 'balance_sheet':
            balance_sheet = stock.balance_sheet
            if balance_sheet.empty:
                return f"No balance sheet data found for {ticker}."
            data_context = f"Latest balance sheet for {long_name}:\n{balance_sheet.to_string()}"

        elif data_type == 'income_statement':
            income_statement = stock.income_stmt
            if income_statement.empty:
                return f"No income statement data found for {ticker}."
            data_context = f"Latest income statement for {long_name}:\n{income_statement.to_string()}"

        elif data_type == 'key_metrics':
            data_context = (
                f"Key metrics for {long_name} (Ticker: {ticker}):\n"
                f"- P/E Ratio: {info.get('trailingPE', 'N/A'):.2f}\n"
                f"- Forward P/E: {info.get('forwardPE', 'N/A'):.2f}\n"
                f"- PEG Ratio: {info.get('pegRatio', 'N/A'):.2f}\n"
                f"- Dividend Yield: {info.get('dividendYield', 0) * 100:.2f}%\n"
                f"- 52-Week High: ${info.get('fiftyTwoWeekHigh', 'N/A'):.2f}\n"
                f"- 52-Week Low: ${info.get('fiftyTwoWeekLow', 'N/A'):.2f}\n"
            )
            
        return data_context

    except Exception as e:
        return f"Error fetching data for {ticker}: {e}"

def extract_info(prompt):
    """Extracts ticker, data type, and period from a user prompt."""
    company_name_to_ticker = {
        'tesla': 'TSLA', 'apple': 'AAPL', 'microsoft': 'MSFT',
        'google': 'GOOGL', 'alphabet': 'GOOGL', 'amazon': 'AMZN',
        'nvidia': 'NVDA', 'meta': 'META', 'netflix': 'NFLX'
    }

    prompt_lower = prompt.lower()
    ticker = None
    data_type = 'historical'

    ticker_match = re.search(r'\b[A-Z]{1,5}\b', prompt)
    if ticker_match:
        ticker = ticker_match.group(0)
    else:
        for name, tick in company_name_to_ticker.items():
            if name in prompt_lower:
                ticker = tick
                break

    if any(keyword in prompt_lower for keyword in ['balance sheet', 'balance']):
        data_type = 'balance_sheet'
    elif any(keyword in prompt_lower for keyword in ['income statement', 'income', 'revenue', 'profit', 'earnings']):
        data_type = 'income_statement'
    elif any(keyword in prompt_lower for keyword in ['metrics', 'pe ratio', 'p/e', 'dividend', 'yield']):
        data_type = 'key_metrics'

    period = '1y' 
    if data_type == 'historical':
        period_match = re.search(r'(1|5|10)\s*year[s]?', prompt_lower)
        if period_match:
            period = period_match.group(1) + 'y'
    
    return ticker, data_type, period

def process_uploaded_file(file):
    """
    Processes an uploaded file (CSV, Excel, PDF, DOCX) and extracts its content.
    Returns the extracted content as a string or an error message.
    """
    filename = secure_filename(file.filename)
    file_extension = os.path.splitext(filename)[1].lower()
    
    extracted_text = ""
    try:
        if file_extension == '.csv':
            df = pd.read_csv(io.StringIO(file.read().decode('utf-8')))
            extracted_text = df.to_string()
        elif file_extension in ['.xls', '.xlsx']:
            df = pd.read_excel(io.BytesIO(file.read()))
            extracted_text = df.to_string()
        elif file_extension == '.pdf':
            reader = PyPDF2.PdfReader(io.BytesIO(file.read()))
            for page_num in range(len(reader.pages)):
                extracted_text += reader.pages[page_num].extract_text() + "\n"
        elif file_extension == '.docx':
            document = Document(io.BytesIO(file.read()))
            for paragraph in document.paragraphs:
                extracted_text += paragraph.text + "\n"
        else:
            return "Unsupported file type. Please upload a CSV, Excel, PDF, or DOCX file."
        
        if not extracted_text.strip():
            return f"No readable content found in '{filename}'."

        return f"User-provided data from '{filename}':\n{extracted_text}"
    except Exception as e:
        return f"Error processing uploaded file '{filename}': {e}"

def get_model_response(model_name, client, full_prompt):
    """
    Sends a prompt to a specific LLM and returns the response text.
    Handles different API client formats.
    """
    try:
        if model_name == 'gemini':
            print(f"Calling Gemini...")
            response = client.generate_content(full_prompt)
            return response.text
        elif model_name in ['openai', 'deepseek']:
            print(f"Calling {model_name.capitalize()}...")
            model_to_use = "gpt-3.5-turbo" if model_name == 'openai' else "deepseek-chat"
            response = client.chat.completions.create(
                model=model_to_use,
                messages=[{"role": "user", "content": full_prompt}]
            )
            return response.choices[0].message.content
        elif model_name == 'groq':
            print(f"Calling Groq (Llama)...")
            response = client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[{"role": "user", "content": full_prompt}]
            )
            return response.choices[0].message.content
    except Exception as e:
        print(f"Error with {model_name} client: {e}")
        return None

def evaluate_responses(responses):
    """
    A simple heuristic to select the 'best' response.
    Chooses the longest response.
    """
    best_response = ""
    for response in responses:
        if response and len(response) > len(best_response):
            best_response = response
    
    if not best_response:
        return "Sorry, all models failed to provide a valid response."
    
    return best_response

@app.route('/')
def index():
    """Renders the main HTML page."""
    return render_template('index.html')

@app.route('/ask_gemini', methods=['POST'])
def ask_gemini():
    """Handles user prompts and file uploads, orchestrates LLM responses."""
    user_prompt = request.form.get('prompt')
    uploaded_file = request.files.get('file')

    if not user_prompt and not uploaded_file:
        return jsonify({'error': 'No prompt or file provided.'}), 400

    financial_data_context = ""
    user_file_data_context = ""

    if uploaded_file:
        user_file_data_context = process_uploaded_file(uploaded_file)
        if "Error" in user_file_data_context or "Unsupported" in user_file_data_context:
            return jsonify({'error': user_file_data_context}), 400
        print(f"Processed uploaded file data: {user_file_data_context[:200]}...") 

    ticker, data_type, period = extract_info(user_prompt)
    if ticker and not uploaded_file: 
        financial_data_context = get_financial_data(ticker, data_type, period)
        print(f"Fetched yfinance data: {financial_data_context[:200]}...") 

    full_prompt = (
        "You are a financial advisor. Provide concise, easy-to-understand insights. "
        "Use the provided data to answer the user's request. "
        "Here is the user's prompt: " + user_prompt
    )

    if financial_data_context:
        full_prompt += "\n\nAdditional financial data for context:\n" + financial_data_context
    if user_file_data_context:
        full_prompt += "\n\nUser-provided file data for context:\n" + user_file_data_context

    responses = []
    if not llm_clients:
        return jsonify({'error': 'No LLM API keys configured. Please set GOOGLE_API_KEY, OPENAI_API_KEY, GROQ_API_KEY, or DEEPSEEK_API_KEY environment variables.'}), 500

    for model_name, client in llm_clients.items():
        response = get_model_response(model_name, client, full_prompt)
        responses.append(response)
    
    final_ai_response = evaluate_responses(responses)

    try:
        conn = sqlite3.connect('fingpt_advisor.db')
        cursor = conn.cursor()
        cursor.execute(
            'INSERT INTO interactions (user_prompt, ai_response) VALUES (?, ?)',
            (user_prompt, final_ai_response)
        )
        conn.commit()
        conn.close()
    except Exception as db_e:
        print(f"Database error occurred: {db_e}")
        final_ai_response += "\n\n(Note: A database error occurred while saving this interaction.)"

    return jsonify({'response': final_ai_response})
    
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
