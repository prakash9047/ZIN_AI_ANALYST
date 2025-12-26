import streamlit as st
import pandas as pd
import plotly.express as px
import random
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_core.messages import SystemMessage, HumanMessage
import pandasql as ps
import json

# Load environment variables
load_dotenv()

# HARDCODED API KEY (Replace with your actual key)
HARDCODED_API_KEY = ""  # Replace this with your actual API key

# Supported languages with their native names
SUPPORTED_LANGUAGES = {
    "English": "en",
    "हिंदी (Hindi)": "hi"
}

# Language name mapping for LLM
LANGUAGE_NAMES = {
    "en": "English",
    "hi": "Hindi"
}

# Base English text templates
BASE_TRANSLATIONS = {
    "page_title": "CSV Analyzer",
    "language":  "Language",
    "api_configured": "✅ API Key Configured",
    "api_not_found": "❌ API Key Not Found",
    "upload_csv": "Upload CSV files",
    "upload_instruction": "Drag and drop CSV files here or click Browse files",
    "file_limit": "Limit 200MB per file • CSV format only",
    "select_file": "Select a CSV file",
    "data_summary": "### Data Summary",
    "rows":  "Rows",
    "columns": "Columns",
    "column_list": "### Columns",
    "type":  "Type",
    "unique": "Unique",
    "nulls": "Nulls",
    "min": "Min",
    "max": "Max",
    "mean": "Mean",
    "data_preview": "## Data Preview",
    "query_data": "## Query Data",
    "natural_language": "Natural Language",
    "sql_query": "SQL Query",
    "dashboard": "Dashboard",
    "ask_question": "Ask a question:",
    "analyze":  "Analyze",
    "analyzing": "Analyzing.. .",
    "answer": "### Answer",
    "history": "### History",
    "input_method": "Input method:",
    "ai_generated": "AI Generated SQL",
    "write_directly": "Write SQL Directly",
    "describe_query": "Describe your query:",
    "generate_sql": "Generate SQL",
    "generating_sql": "Generating SQL.. .",
    "generated_sql": "### Generated SQL",
    "sql_query_label": "SQL Query:",
    "execute":  "Execute",
    "clear":  "Clear",
    "executing": "Executing...",
    "returned_rows": "Returned {0} rows",
    "download_results": "Download Results",
    "query_history": "### Query History",
    "sample_charts": "### Sample Charts",
    "custom_chart": "### Custom Chart",
    "chart_type": "Chart Type:",
    "color_by": "Color By:",
    "x_axis":  "X-axis:",
    "y_axis":  "Y-axis:",
    "generate_chart": "Generate Chart",
    "generate_sample":  "Generate Sample Charts",
    "welcome": "### Welcome to CSV Analyzer",
    "get_started": "Upload a CSV file to get started",
    "enter_question": "Please enter a question",
    "describe_query_prompt": "Please describe what you want to query",
    "enter_sql":  "Please enter a SQL query",
    "could_not_read": "Could not read the CSV file",
    "could_not_generate": "Could not generate {0} chart",
    "error":  "Error:  {0}",
    "none": "None",
    "api_key_info": "💡 Add OPENAI_API_KEY to your . env file or Streamlit secrets",
    "invalid_api_key": "❌ Invalid API Key.  Please check your OPENAI_API_KEY",
    "quota_exceeded": "❌ API quota exceeded. Please check your OpenAI account",
    "browse_files": "Browse files"
}

# Page configuration
st.set_page_config(layout='wide', page_title="CSV Analyzer")

# Enhanced API key function with hardcoded option
def get_api_key():
    """Get OpenAI API key from multiple sources with proper error handling"""
    api_key = None
    
    # Try hardcoded API key first (if not empty)
    if HARDCODED_API_KEY and HARDCODED_API_KEY.strip():
        api_key = HARDCODED_API_KEY
    
    # Try Streamlit secrets (for deployment)
    if not api_key:
        try:
            if hasattr(st, 'secrets'):
                if "OPENAI_API_KEY" in st.secrets:
                    api_key = st.secrets["OPENAI_API_KEY"]
                elif "openai" in st.secrets and "api_key" in st. secrets["openai"]:
                    api_key = st.secrets["openai"]["api_key"]
        except Exception:
            pass
    
    # Try environment variable (for local development)
    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY")
    
    # Try . env file explicitly
    if not api_key: 
        try:
            from dotenv import dotenv_values
            env_values = dotenv_values(". env")
            api_key = env_values.get("OPENAI_API_KEY")
        except:
            pass
    
    # Clean the API key
    if api_key:
        api_key = api_key.strip().strip('"').strip("'")
    
    return api_key

openai_api_key = get_api_key()

# Initialize session state
def initialize_session_state():
    """Initialize session state variables"""
    if 'language' not in st.session_state:
        st.session_state.language = 'en'
    if 'translations_cache' not in st.session_state:
        st.session_state.translations_cache = {}
    if 'nl_history' not in st.session_state:
        st.session_state.nl_history = []
    if 'sql_history' not in st.session_state:
        st.session_state.sql_history = []

initialize_session_state()

# Dynamic translation function using LLM
def translate_text_llm(text, target_lang="en", _api_key=None):
    """Translate text using OpenAI LLM with caching"""
    if not _api_key or target_lang == "en":
        return text
    
    if target_lang not in LANGUAGE_NAMES:
        return text
        
    # Check cache first
    cache_key = f"{text}_{target_lang}"
    if cache_key in st.session_state.translations_cache:
        return st.session_state. translations_cache[cache_key]
    
    try:
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.1,
            api_key=_api_key,
            max_tokens=500
        )
        
        target_language_name = LANGUAGE_NAMES[target_lang]
        
        response = llm.invoke([
            SystemMessage(content=f"""You are a professional translator.  Translate the following text to {target_language_name}. 
Rules:
1. Only return the translation, nothing else
2. Maintain the original formatting (like ### for headers)
3. Keep placeholders like {{0}} unchanged
4. Preserve technical terms appropriately
5. Use natural, fluent language"""),
            HumanMessage(content=text)
        ])
        
        translated_text = response.content. strip()
        
        # Cache the translation
        st.session_state.translations_cache[cache_key] = translated_text
        
        return translated_text
    except Exception: 
        return text

def get_text(key, lang="en"):
    """Get translated text based on selected language"""
    base_text = BASE_TRANSLATIONS. get(key, key)
    
    if lang == "en" or not openai_api_key:
        return base_text
    
    return translate_text_llm(base_text, lang, openai_api_key)

# Language selector - FIXED TO UPDATE IMMEDIATELY
col_lang, col_empty = st.columns([1, 5])
with col_lang:
    # Get current language for proper indexing
    current_lang_index = 0
    if st.session_state.language == "hi":
        current_lang_index = 1
    
    selected_language = st.selectbox(
        "Language",  # Keep this in English for consistency
        options=list(SUPPORTED_LANGUAGES.keys()),
        index=current_lang_index,
        key="language_selector"
    )
    
    # Update session state immediately
    new_lang = SUPPORTED_LANGUAGES[selected_language]
    if new_lang != st.session_state.language:
        st. session_state.language = new_lang
        st.rerun()  # Force rerun to update translations

lang = st.session_state.language

def execute_sql_query(df, sql_query):
    """Execute SQL query on pandas dataframe"""
    try:
        sql_query = sql_query.strip()
        if sql_query.endswith(';'):
            sql_query = sql_query[:-1]
        
        # Replace common table name variations
        replacements = ['FROM data', 'from data', 'FROM Data', 'FROM df', 'from df']
        for replacement in replacements:
            sql_query = sql_query.replace(replacement, 'FROM df')
        
        result = ps.sqldf(sql_query, locals())
        return result, None
    except Exception as e: 
        return None, f"SQL Error: {str(e)}"

def chat_with_csv_natural_language(df, prompt, language="en"):
    """Analyze CSV data using natural language with dynamic translation"""
    if not openai_api_key:
        return get_text("api_key_info", language), None
    
    try:
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            api_key=openai_api_key,
            max_tokens=1000
        )
        
        # Translate prompt to English if needed
        english_prompt = prompt
        if language == "hi": 
            english_prompt = translate_text_llm(prompt, "en", openai_api_key)
        
        try:
            # Create pandas agent - UPDATED FOR CURRENT LANGCHAIN VERSION
            agent = create_pandas_dataframe_agent(
                llm,
                df,
                verbose=False,
                allow_dangerous_code=True,
                agent_type="openai-tools",  # Updated agent type
                max_iterations=5,
                max_execution_time=30,
                handle_parsing_errors=True
            )
            
            result = agent.invoke({"input": english_prompt})
            
            answer = ""
            if isinstance(result, dict):
                answer = result.get('output', str(result))
            else:
                answer = str(result)
            
            # Translate answer back to Hindi if needed
            if language == "hi": 
                answer = translate_text_llm(answer, "hi", openai_api_key)
            
            return answer, None
                
        except Exception: 
            # Fallback method
            df_info = f"""
DataFrame Information:
- Shape: {df.shape[0]} rows, {df.shape[1]} columns
- Columns: {', '.join(df.columns. tolist())}

Statistical Summary:
{df.describe().to_string()}

First 5 rows:
{df.head().to_string()}
"""
            
            fallback_prompt = f"""{df_info}

Question: {english_prompt}

Please provide a detailed answer based on the data above."""
            
            response = llm.invoke([
                SystemMessage(content="You are a data analyst. Provide clear, accurate answers based on the provided dataset."),
                HumanMessage(content=fallback_prompt)
            ])
            
            answer = response.content
            
            # Translate answer back if needed
            if language == "hi":
                answer = translate_text_llm(answer, "hi", openai_api_key)
            
            return answer, None
        
    except Exception as e: 
        error_msg = str(e)
        if "401" in error_msg or "invalid_api_key" in error_msg or "Incorrect API key" in error_msg: 
            return get_text("invalid_api_key", language), error_msg
        elif "insufficient_quota" in error_msg: 
            return get_text("quota_exceeded", language), error_msg
        else:
            return get_text("error", language).format(error_msg), error_msg

def generate_sql_query(df, prompt, language="en"):
    """Generate SQL query from natural language with translation support"""
    if not openai_api_key:
        return get_text("api_key_info", language), None
    
    try:
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            api_key=openai_api_key,
            max_tokens=500
        )
        
        # Translate prompt to English if needed
        english_prompt = prompt
        if language == "hi":
            english_prompt = translate_text_llm(prompt, "en", openai_api_key)
        
        # Generate column information
        column_info = []
        for col in df.columns:
            dtype = str(df[col].dtype)
            sample_values = df[col].dropna().head(3).tolist()
            column_info.append(f"  - {col} ({dtype}): {sample_values}")
        
        columns_description = "\n".join(column_info)
        
        sql_prompt = f"""Generate ONLY a SQL query (no explanation) that answers this question. 

Question: {english_prompt}

Table name: df
Columns: 
{columns_description}

Return only the SQL query without any formatting or explanation."""

        response = llm.invoke([
            SystemMessage(content="You are a SQL expert. Generate only clean SQL queries without explanation, code blocks, or formatting. "),
            HumanMessage(content=sql_prompt)
        ])
        
        sql_query = response.content.strip()
        
        # Clean up the SQL query
        if "```sql" in sql_query:
            sql_query = sql_query.split("```sql")[1].split("```")[0].strip()
        elif "```" in sql_query:
            sql_query = sql_query.split("```")[1].split("```")[0].strip()
        
        # Remove comments and clean lines
        lines = sql_query.split('\n')
        sql_lines = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith('--') and not line.startswith('/*'):
                sql_lines.append(line)
        
        sql_query = ' '.join(sql_lines) if sql_lines else sql_query
        
        return sql_query, None
        
    except Exception as e:
        return f"Error: {str(e)}", str(e)

def generate_sample_charts(data):
    """Generate sample chart configurations"""
    charts = []
    numeric_columns = data.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.tolist()
    categorical_columns = data.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

    if len(numeric_columns) >= 2:
        x = numeric_columns[0]
        y = numeric_columns[1]
        charts.append(("Scatter", x, y))
        if len(numeric_columns) > 2:
            charts.append(("Line", numeric_columns[0], numeric_columns[2]))

    if len(categorical_columns) >= 1 and len(numeric_columns) >= 1:
        x = categorical_columns[0]
        y = numeric_columns[0]
        charts.append(("Bar", x, y))

    return charts[: 3]

# Main layout
col1, col2 = st.columns([1, 3])

# FIXED SIDEBAR - Now properly translates
with col1:
    st. title(get_text("page_title", lang))
    st.markdown("---")
    
    # API Key Status
    if openai_api_key:
        st.success(get_text("api_configured", lang))
    else:
        st.error(get_text("api_not_found", lang))
        st.info(get_text("api_key_info", lang))

    # FIXED FILE UPLOADER with custom instructions
    st.markdown(f"**{get_text('upload_csv', lang)}**")
    
    # Add custom instruction text in the selected language
    if lang == "hi":
        st. markdown("📁 *CSV फाइलों को यहाँ खींचें और छोड़ें या Browse files पर क्लिक करें*")
        st.markdown("*अधिकतम 200MB प्रति फाइल • केवल CSV फॉर्मेट*")
    else:
        st.markdown("📁 *Drag and drop CSV files here or click Browse files*")
        st.markdown("*Limit 200MB per file • CSV format only*")
    
    # The actual file uploader (Streamlit's built-in text will remain in English)
    input_csvs = st.file_uploader("", type=['csv'], accept_multiple_files=True, label_visibility="collapsed")

    if input_csvs: 
        selected_file = st.selectbox(get_text("select_file", lang), [file.name for file in input_csvs])
        selected_index = [file.name for file in input_csvs].index(selected_file)
        
        # Enhanced CSV reading with multiple encodings and delimiters
        encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'windows-1252']
        delimiters = [',', ';', '\t', '|']
        data = None
        
        for encoding in encodings:
            for delimiter in delimiters:
                try:
                    input_csvs[selected_index].seek(0)
                    data = pd.read_csv(
                        input_csvs[selected_index], 
                        encoding=encoding, 
                        delimiter=delimiter,
                        on_bad_lines='skip',
                        encoding_errors='ignore'
                    )
                    if not data.empty and len(data. columns) > 1:
                        break
                except: 
                    continue
            if data is not None and not data.empty:
                break
        
        if data is None or data.empty:
            st.error(get_text("could_not_read", lang))
            st.stop()

        # Data summary
        st.markdown(get_text("data_summary", lang))
        st.write(f"**{get_text('rows', lang)}:** {data.shape[0]: ,}")
        st.write(f"**{get_text('columns', lang)}:** {data.shape[1]}")
        
        # Column information
        st.markdown(get_text("column_list", lang))
        for column in data.columns:
            with st.expander(column):
                col_type = str(data[column].dtype)
                unique_values = data[column].nunique()
                null_count = data[column].isnull().sum()
                
                st.write(f"**{get_text('type', lang)}:** {col_type}")
                st.write(f"**{get_text('unique', lang)}:** {unique_values}")
                st.write(f"**{get_text('nulls', lang)}:** {null_count}")
                
                # Show statistics for numeric columns
                if pd.api.types.is_numeric_dtype(data[column]):
                    try:
                        st.write(f"**{get_text('min', lang)}:** {data[column].min()}")
                        st.write(f"**{get_text('max', lang)}:** {data[column].max()}")
                        mean_val = data[column].mean()
                        if pd.notna(mean_val):
                            st.write(f"**{get_text('mean', lang)}:** {mean_val:.2f}")
                    except: 
                        pass

# Main content area
with col2:
    if 'data' in locals() and data is not None:
        st.markdown(get_text("data_preview", lang))
        st.dataframe(data. head(100), use_container_width=True)

        st.markdown(get_text("query_data", lang))
        
        tab1, tab2, tab3 = st.tabs([
            get_text("natural_language", lang),
            get_text("sql_query", lang),
            get_text("dashboard", lang)
        ])
        
        # Natural Language Tab
        with tab1:
            nl_input = st.text_area(get_text("ask_question", lang), height=100, key="nl_input")
            
            if st.button(get_text("analyze", lang), key="nl_button"):
                if nl_input. strip():
                    with st.spinner(get_text("analyzing", lang)):
                        result, error = chat_with_csv_natural_language(data, nl_input, lang)
                        
                        st.markdown(get_text("answer", lang))
                        st. write(result)
                        
                        # Add to history
                        st.session_state.nl_history.append({
                            'question': nl_input,
                            'answer':  result,
                            'language': lang
                        })
                else:
                    st.warning(get_text("enter_question", lang))
            
            # Show history
            if st.session_state.nl_history:
                st.markdown("---")
                st.markdown(get_text("history", lang))
                for i, item in enumerate(reversed(st.session_state.nl_history[-5:])):
                    with st.expander(f"Q{len(st.session_state.nl_history)-i}:  {item['question'][: 50]}..."):
                        st.write(f"**Q:** {item['question']}")
                        st.write(f"**A:** {item['answer']}")
        
        # SQL Query Tab
        with tab2:
            query_input_method = st.radio(
                get_text("input_method", lang),
                [get_text("ai_generated", lang), get_text("write_directly", lang)],
                key="sql_method"
            )
            
            if query_input_method == get_text("ai_generated", lang):
                nl_query = st.text_area(get_text("describe_query", lang), height=80, key="sql_nl_input")
                
                if st. button(get_text("generate_sql", lang), key="gen_sql_button"):
                    if nl_query.strip():
                        with st.spinner(get_text("generating_sql", lang)):
                            sql_query, error = generate_sql_query(data, nl_query, lang)
                            
                            if not error:
                                st.session_state.generated_sql = sql_query
                                st.markdown(get_text("generated_sql", lang))
                                st. code(sql_query, language="sql")
                            else:
                                st.error(sql_query)
                    else: 
                        st.warning(get_text("describe_query_prompt", lang))
            
            # SQL Editor
            if 'generated_sql' in st.session_state:
                default_sql = st.session_state.generated_sql
            else:
                default_sql = "SELECT * FROM df LIMIT 10"
            
            sql_query_input = st.text_area(get_text("sql_query_label", lang), value=default_sql, height=120, key="sql_editor")
            
            col_exec, col_clear = st.columns([1, 4])
            with col_exec: 
                execute_btn = st.button(get_text("execute", lang), key="exec_sql_button", use_container_width=True)
            with col_clear:
                if st.button(get_text("clear", lang), key="clear_sql_button"):
                    if 'generated_sql' in st. session_state:
                        del st.session_state.generated_sql
                    st.rerun()
            
            if execute_btn:
                if sql_query_input.strip():
                    with st.spinner(get_text("executing", lang)):
                        result_df, error = execute_sql_query(data, sql_query_input)
                        
                        if not error and result_df is not None: 
                            st.success(get_text("returned_rows", lang).format(len(result_df)))
                            st.dataframe(result_df, use_container_width=True)
                            
                            csv = result_df.to_csv(index=False)
                            st. download_button(
                                label=get_text("download_results", lang),
                                data=csv,
                                file_name="query_results.csv",
                                mime="text/csv"
                            )
                            
                            # Add to history
                            st.session_state.sql_history.append({
                                'query': sql_query_input,
                                'success': True,
                                'rows': len(result_df)
                            })
                        else:
                            st.error(error or "Unknown error occurred")
                            st.session_state.sql_history.append({
                                'query': sql_query_input,
                                'success': False,
                                'rows': 0
                            })
                else: 
                    st.warning(get_text("enter_sql", lang))
            
            # Show SQL history
            if st.session_state.sql_history:
                st.markdown("---")
                st.markdown(get_text("query_history", lang))
                for i, item in enumerate(reversed(st.session_state.sql_history[-5:])):
                    status = "✅" if item['success'] else "❌"
                    with st. expander(f"{status} Query {len(st.session_state.sql_history)-i}"):
                        st.code(item['query'], language="sql")
                        if item['success']:
                            st.write(get_text("returned_rows", lang).format(item['rows']))
        
        # Dashboard Tab
        with tab3:
            if 'sample_charts' not in st.session_state or st.button(get_text("generate_sample", lang)):
                st.session_state.sample_charts = generate_sample_charts(data)

            # Sample charts
            if st.session_state.sample_charts:
                st.markdown(get_text("sample_charts", lang))
                for i, (chart_type, x, y) in enumerate(st.session_state.sample_charts):
                    try:
                        fig = None
                        if chart_type == "Scatter":
                            fig = px.scatter(data, x=x, y=y, title=f"{x} vs {y}")
                        elif chart_type == "Line":
                            fig = px.line(data, x=x, y=y, title=f"{x} vs {y}")
                        elif chart_type == "Bar":
                            agg_data = data.groupby(x)[y].mean().reset_index()
                            fig = px.bar(agg_data, x=x, y=y, title=f"{y} by {x}")
                        
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                    except Exception: 
                        st.warning(get_text("could_not_generate", lang).format(chart_type))

            # Custom chart builder
            st.markdown("---")
            st.markdown(get_text("custom_chart", lang))
            
            col_chart1, col_chart2 = st.columns(2)
            
            with col_chart1:
                chart_type = st. selectbox(get_text("chart_type", lang), 
                    ["Bar", "Line", "Scatter", "Pie", "Area", "Box", "Histogram"])
            
            with col_chart2:
                color_col = st. selectbox(get_text("color_by", lang), 
                    [get_text("none", lang)] + data.columns.tolist(), key="color_select")
            
            col_axis1, col_axis2 = st.columns(2)
            
            with col_axis1:
                x_axis = st.selectbox(get_text("x_axis", lang), data.columns, key="x_axis_dash")
            
            with col_axis2:
                y_axis = st.selectbox(get_text("y_axis", lang), data.columns, key="y_axis_dash")

            if st.button(get_text("generate_chart", lang), key="gen_chart_button"):
                try:
                    fig = None
                    color_param = None if color_col == get_text("none", lang) else color_col
                    
                    if chart_type == "Bar":
                        if data[x_axis]. dtype == 'object':
                            agg_data = data. groupby(x_axis)[y_axis].mean().reset_index()
                            fig = px.bar(agg_data, x=x_axis, y=y_axis, color=color_param)
                        else:
                            fig = px.bar(data, x=x_axis, y=y_axis, color=color_param)
                    
                    elif chart_type == "Line":
                        fig = px.line(data, x=x_axis, y=y_axis, color=color_param)
                    
                    elif chart_type == "Scatter": 
                        fig = px.scatter(data, x=x_axis, y=y_axis, color=color_param)
                    
                    elif chart_type == "Pie":
                        if data[x_axis].dtype == 'object':
                            pie_data = data.groupby(x_axis)[y_axis].sum().reset_index()
                            fig = px.pie(pie_data, names=x_axis, values=y_axis)
                        else:
                            fig = px.pie(data, names=x_axis, values=y_axis)
                    
                    elif chart_type == "Area": 
                        fig = px.area(data, x=x_axis, y=y_axis, color=color_param)
                    
                    elif chart_type == "Box":
                        fig = px.box(data, x=x_axis, y=y_axis, color=color_param)
                    
                    elif chart_type == "Histogram": 
                        fig = px.histogram(data, x=x_axis, color=color_param)
                    
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(get_text("error", lang).format(str(e)))

    else:
        st.markdown(get_text("welcome", lang))
        st.write(get_text("get_started", lang))
