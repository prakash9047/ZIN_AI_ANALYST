import streamlit as st
import pandas as pd
import plotly.express as px
import random
import json
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType
from langchain_core.messages import SystemMessage, HumanMessage
import pandasql as ps

# Load environment variables from .env file
load_dotenv()

# Page configuration
st.set_page_config(layout='wide', page_title="CSV Analyzer", page_icon="📊")

# Get API key from either Streamlit secrets or environment variable
def get_api_key():
    """Get OpenAI API key from secrets or environment"""
    try:
        if hasattr(st, 'secrets') and "OPENAI_API_KEY" in st.secrets:
            return st.secrets["OPENAI_API_KEY"]
        elif os.getenv("OPENAI_API_KEY"):
            return os.getenv("OPENAI_API_KEY")
        else:
            return None
    except:
        return None

openai_api_key = get_api_key()

def execute_sql_query(df, sql_query):
    """Execute SQL query on pandas dataframe using pandasql"""
    try:
        # Clean the SQL query
        sql_query = sql_query.strip()
        if sql_query.endswith(';'):
            sql_query = sql_query[:-1]
        
        # Replace common table name variations with 'df'
        sql_query = sql_query.replace('FROM data', 'FROM df')
        sql_query = sql_query. replace('from data', 'from df')
        sql_query = sql_query.replace('FROM Data', 'FROM df')
        
        # Execute query
        result = ps.sqldf(sql_query, locals())
        return result, None
    except Exception as e: 
        return None, f"SQL Error: {str(e)}"

def chat_with_csv_natural_language(df, prompt):
    """Use LangChain with OpenAI to analyze CSV data - Natural Language Mode"""
    if not openai_api_key:
        return "⚠️ Please add your OPENAI_API_KEY to Streamlit secrets or . env file", None
    
    try: 
        # Initialize the LLM
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            api_key=openai_api_key
        )
        
        # Try with OPENAI_FUNCTIONS agent first (most reliable)
        try:
            agent = create_pandas_dataframe_agent(
                llm,
                df,
                verbose=False,
                allow_dangerous_code=True,
                agent_type=AgentType. OPENAI_FUNCTIONS,
                max_iterations=5,
                max_execution_time=30,
                handle_parsing_errors=True
            )
            
            result = agent.invoke({"input": prompt})
            
            if isinstance(result, dict):
                return result.get('output', str(result)), None
            else: 
                return str(result), None
                
        except Exception as agent_error:
            # Fallback:  Direct analysis with LLM
            df_info = f"""
DataFrame Information:
- Shape: {df.shape[0]} rows, {df.shape[1]} columns
- Columns: {', '.join(df.columns. tolist())}
- Data Types: {df.dtypes.to_dict()}

Statistical Summary:
{df.describe().to_string()}

First 5 rows:
{df.head().to_string()}

Last 5 rows:
{df.tail().to_string()}
"""
            
            fallback_prompt = f"""{df_info}

Question: {prompt}

Please provide a detailed answer based on the data above.  If calculations are needed, show your work."""
            
            response = llm.invoke([
                SystemMessage(content="You are a data analyst. Provide clear, accurate answers based on the provided dataset."),
                HumanMessage(content=fallback_prompt)
            ])
            
            return response.content, None
        
    except Exception as e: 
        error_msg = str(e)
        
        if "401" in error_msg or "invalid_api_key" in error_msg: 
            return "❌ Invalid API Key.  Please check your OpenAI API key.", error_msg
        elif "insufficient_quota" in error_msg:
            return "❌ OpenAI API quota exceeded. Please check your billing.", error_msg
        else:
            return f"❌ Error: {error_msg}", error_msg

def generate_sql_query(df, prompt):
    """Generate SQL query based on natural language prompt"""
    if not openai_api_key: 
        return "⚠️ Please add your OPENAI_API_KEY to Streamlit secrets or .env file", None
    
    try:
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            api_key=openai_api_key
        )
        
        # Get column information
        column_info = []
        for col in df.columns:
            dtype = str(df[col].dtype)
            sample_values = df[col].dropna().head(3).tolist()
            column_info.append(f"  - {col} ({dtype}): Sample values:  {sample_values}")
        
        columns_description = "\n".join(column_info)
        
        sql_prompt = f"""You are a SQL expert. Generate ONLY a SQL query (no explanation, no markdown) that answers this question. 

Question: {prompt}

Table name: df
Columns and their details:
{columns_description}

DataFrame shape: {df.shape[0]} rows, {df.shape[1]} columns

Requirements:
1. Use table name 'df' (not 'data')
2. Return ONLY the SQL query
3. No markdown formatting, no ```sql```, no explanations
4. Use proper SQL syntax (SELECT, FROM, WHERE, GROUP BY, ORDER BY, etc.)
5. For aggregations, use functions like COUNT, SUM, AVG, MAX, MIN
6. Ensure column names are exactly as shown above

SQL Query:"""

        response = llm.invoke([
            SystemMessage(content="You are a SQL expert. Generate only SQL queries without any explanation or markdown formatting."),
            HumanMessage(content=sql_prompt)
        ])
        
        sql_query = response.content. strip()
        
        # Clean up the SQL query
        if "```sql" in sql_query: 
            sql_query = sql_query.split("```sql")[1].split("```")[0].strip()
        elif "```" in sql_query:
            sql_query = sql_query. split("```")[1].split("```")[0].strip()
        
        # Remove any explanatory text before or after the query
        lines = sql_query.split('\n')
        sql_lines = []
        for line in lines:
            line = line.strip()
            if line and (line.upper().startswith('SELECT') or 
                        line.upper().startswith('FROM') or 
                        line. upper().startswith('WHERE') or 
                        line.upper().startswith('GROUP') or 
                        line.upper().startswith('ORDER') or 
                        line.upper().startswith('HAVING') or
                        line.upper().startswith('LIMIT') or
                        (sql_lines and not line.upper().startswith('--'))):
                sql_lines.append(line)
        
        sql_query = ' '.join(sql_lines) if sql_lines else sql_query
        
        return sql_query, None
        
    except Exception as e: 
        error_msg = str(e)
        return f"Error generating SQL: {error_msg}", error_msg

def generate_sample_charts(data):
    """Generate sample chart configurations"""
    charts = []
    numeric_columns = data.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.tolist()
    categorical_columns = data.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

    # Add scatter and line charts for numeric data
    if len(numeric_columns) >= 2:
        x = numeric_columns[0]
        y = numeric_columns[1] if len(numeric_columns) > 1 else numeric_columns[0]
        charts.append(("Scatter", x, y))
        if len(numeric_columns) > 2:
            charts.append(("Line", numeric_columns[0], numeric_columns[2]))

    # Add bar and box charts for categorical vs numeric
    if len(categorical_columns) >= 1 and len(numeric_columns) >= 1:
        x = categorical_columns[0]
        y = numeric_columns[0]
        charts.append(("Bar", x, y))

    return charts[: 3]  # Return at most 3 charts

# Custom CSS
st.markdown("""
    <style>
    .stButton button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        border: none;
    }
    .stButton button:hover {
        background-color:  #45a049;
    }
    .column-header {
        font-weight: bold;
        font-size: 1.2em;
        margin-bottom: 5px;
        color: #1f77b4;
    }
    .column-info {
        margin-left: 10px;
        font-size: 0.9em;
    }
    .info-box {
        background-color: #e7f3ff;
        padding: 1rem;
        border-radius:  5px;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .success-box {
        background-color:  #d4edda;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    .error-box {
        background-color:  #f8d7da;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #dc3545;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Main layout
col1, col2 = st.columns([1, 3])

with col1:
    # Try to load logo
    try:
        st.image("icon.jpg", width=150)
    except:
        st.markdown("### 📊")
    
    st.title("CSV Analyzer")
    st.markdown("---")
    
    # API Key Status
    if openai_api_key:
        st.success("✅ OpenAI API Key Configured")
    else:
        st.error("❌ OpenAI API Key Not Found")
        st.info("Add OPENAI_API_KEY to Streamlit secrets or .env file")

    # File uploader
    input_csvs = st.file_uploader("Upload CSV files", type=['csv'], accept_multiple_files=True)

    if input_csvs: 
        selected_file = st.selectbox("Select a CSV file", [file.name for file in input_csvs])
        selected_index = [file.name for file in input_csvs].index(selected_file)
        
        # Try multiple encodings and delimiters
        encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'windows-1252']
        delimiters = [',', ';', '\t', '|']
        data = None
        
        for encoding in encodings:
            for delimiter in delimiters:
                try:
                    input_csvs[selected_index]. seek(0)
                    data = pd.read_csv(
                        input_csvs[selected_index], 
                        encoding=encoding, 
                        delimiter=delimiter,
                        on_bad_lines='skip',
                        encoding_errors='ignore'
                    )
                    if not data.empty and len(data.columns) > 1:
                        break
                except: 
                    continue
            if data is not None and not data.empty:
                break
        
        if data is None or data.empty:
            st. error("Could not read the CSV file. Please check the file format.")
            st.stop()

        # Data summary
        st.markdown("### 📋 Data Summary")
        st.info(f"**Rows:** {data.shape[0]: ,}  \n**Columns:** {data.shape[1]}")
        
        # Column overview
        st.markdown("### 📊 Column Overview")
        for column in data.columns:
            with st.expander(f"📌 {column}"):
                st.markdown(f"<div class='column-header'>{column}</div>", unsafe_allow_html=True)
                
                col_type = str(data[column].dtype)
                unique_values = data[column].nunique()
                null_count = data[column].isnull().sum()
                
                st. markdown(f"<div class='column-info'><b>Type:</b> {col_type}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='column-info'><b>Unique values:</b> {unique_values}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='column-info'><b>Null values:</b> {null_count}</div>", unsafe_allow_html=True)
                
                if unique_values < 20:
                    st.markdown("<div class='column-info'><b>Categories:</b></div>", unsafe_allow_html=True)
                    value_counts = data[column].value_counts().head(10)
                    for value, count in value_counts. items():
                        st.markdown(f"<div class='column-info'>  • {value}: {count}</div>", unsafe_allow_html=True)
                elif data[column].dtype in ['int64', 'float64', 'int32', 'float32']:
                    st.markdown(f"<div class='column-info'><b>Min:</b> {data[column]. min()}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='column-info'><b>Max:</b> {data[column].max()}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='column-info'><b>Mean:</b> {data[column].mean():.2f}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='column-info'><b>Median:</b> {data[column]. median():.2f}</div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div class='column-info'><b>Sample values:</b></div>", unsafe_allow_html=True)
                    sample_values = data[column].dropna().sample(min(5, len(data[column]. dropna()))).tolist()
                    for value in sample_values:
                        st.markdown(f"<div class='column-info'>  • {value}</div>", unsafe_allow_html=True)

with col2:
    if 'data' in locals() and data is not None:
        st.markdown("## 📁 Data Preview")
        st.dataframe(data.head(100), use_container_width=True)

        st.markdown("## 💬 Query Your Data")
        
        # Mode selection with tabs
        tab1, tab2, tab3 = st.tabs(["🗣️ Natural Language", "🔍 SQL Query", "📊 Dashboard"])
        
        # ==================== NATURAL LANGUAGE MODE ====================
        with tab1:
            st.markdown("### Ask Questions in Plain English")
            
            # Example questions
            with st.expander("💡 Example Questions"):
                st. markdown("""
                **Statistical Questions:**
                - What is the average/mean of [column name]?
                - What is the maximum/minimum value in [column name]?
                - How many unique values are in [column name]? 
                - What is the total sum of [column name]?
                
                **Data Exploration:**
                - Show me the top 5 rows
                - How many rows are in this dataset?
                - Which column has the most null values?
                - What are the data types of all columns?
                
                **Analysis Questions:**
                - What is the correlation between [column1] and [column2]? 
                - Find all rows where [column name] is greater than [value]
                - Group by [column name] and show the average [another column]
                - What are the most common values in [column name]?
                """)
            
            nl_input = st.text_area("Ask a question about your data:", height=100, key="nl_input")
            
            if st.button("🔍 Analyze", key="nl_button"):
                if not nl_input. strip():
                    st.warning("⚠️ Please enter a question about your data.")
                else:
                    with st.spinner("🤔 Analyzing your question..."):
                        result, error = chat_with_csv_natural_language(data, nl_input)
                        
                        if error:
                            st.markdown(f"<div class='error-box'>{result}</div>", unsafe_allow_html=True)
                        else:
                            st.markdown(f"<div class='success-box'><b>Answer:</b><br>{result}</div>", unsafe_allow_html=True)
                        
                        # Store in session state
                        if 'nl_history' not in st.session_state:
                            st.session_state.nl_history = []
                        st.session_state.nl_history.append({
                            'question': nl_input,
                            'answer':  result
                        })
            
            # Show history
            if 'nl_history' in st. session_state and st.session_state.nl_history:
                st.markdown("---")
                st.markdown("### 📜 Query History")
                for i, item in enumerate(reversed(st.session_state.nl_history[-5:])):
                    with st.expander(f"Q{len(st.session_state.nl_history)-i}:  {item['question'][: 50]}..."):
                        st. markdown(f"**Question:** {item['question']}")
                        st.markdown(f"**Answer:** {item['answer']}")
        
        # ==================== SQL QUERY MODE ====================
        with tab2:
            st.markdown("### Generate and Execute SQL Queries")
            
            # Example SQL queries
            with st.expander("💡 Example SQL Questions"):
                st.markdown("""
                **Basic Queries:**
                - Show all columns for the first 10 rows
                - Select all rows where [column] equals [value]
                - Count the number of rows in the dataset
                
                **Aggregation:**
                - Calculate the average of [column] grouped by [another column]
                - Find the maximum value in [column]
                - Count unique values in [column]
                
                **Filtering & Sorting:**
                - Show top 10 rows sorted by [column] descending
                - Filter rows where [column] is greater than [value]
                - Get rows where [column] contains [text]
                """)
            
            query_input_method = st.radio("Choose input method:", ["Natural Language (AI Generated)", "Write SQL Directly"], key="sql_method")
            
            if query_input_method == "Natural Language (AI Generated)":
                nl_query = st.text_area("Describe what you want to query:", height=80, key="sql_nl_input")
                
                if st. button("🤖 Generate SQL Query", key="gen_sql_button"):
                    if not nl_query.strip():
                        st.warning("⚠️ Please describe what you want to query.")
                    else:
                        with st.spinner("🤖 Generating SQL query..."):
                            sql_query, error = generate_sql_query(data, nl_query)
                            
                            if error: 
                                st.markdown(f"<div class='error-box'>{sql_query}</div>", unsafe_allow_html=True)
                            else:
                                st.session_state.generated_sql = sql_query
                                st.markdown(f"<div class='success-box'><b>Generated SQL Query:</b></div>", unsafe_allow_html=True)
                                st. code(sql_query, language="sql")
            
            # SQL query editor
            if 'generated_sql' in st. session_state:
                default_sql = st.session_state.generated_sql
            else:
                default_sql = "SELECT * FROM df LIMIT 10"
            
            sql_query_input = st.text_area("SQL Query (edit if needed):", value=default_sql, height=120, key="sql_editor")
            
            col_exec, col_clear = st.columns([1, 4])
            with col_exec:
                execute_btn = st.button("▶️ Execute Query", key="exec_sql_button", use_container_width=True)
            with col_clear: 
                if st.button("🗑️ Clear", key="clear_sql_button"):
                    if 'generated_sql' in st. session_state:
                        del st.session_state.generated_sql
                    st.rerun()
            
            if execute_btn:
                if not sql_query_input.strip():
                    st.warning("⚠️ Please enter a SQL query.")
                else:
                    with st.spinner("⚙️ Executing query..."):
                        result_df, error = execute_sql_query(data, sql_query_input)
                        
                        if error: 
                            st.markdown(f"<div class='error-box'>{error}</div>", unsafe_allow_html=True)
                            st.info("💡 Tip: Make sure to use 'df' as the table name and check column names are correct.")
                        else:
                            st.markdown(f"<div class='success-box'><b>Query executed successfully!</b> Returned {len(result_df)} rows.</div>", unsafe_allow_html=True)
                            st.dataframe(result_df, use_container_width=True)
                            
                            # Download option
                            csv = result_df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Results as CSV",
                                data=csv,
                                file_name="query_results.csv",
                                mime="text/csv"
                            )
                        
                        # Store in history
                        if 'sql_history' not in st.session_state:
                            st.session_state.sql_history = []
                        st.session_state.sql_history.append({
                            'query': sql_query_input,
                            'success': error is None,
                            'rows':  len(result_df) if result_df is not None else 0
                        })
            
            # SQL History
            if 'sql_history' in st.session_state and st.session_state.sql_history:
                st.markdown("---")
                st.markdown("### 📜 SQL Query History")
                for i, item in enumerate(reversed(st.session_state.sql_history[-5:])):
                    status = "✅" if item['success'] else "❌"
                    with st.expander(f"{status} Query {len(st.session_state.sql_history)-i}: {item['query'][:40]}..."):
                        st. code(item['query'], language="sql")
                        if item['success']:
                            st.success(f"Returned {item['rows']} rows")
        
        # ==================== DASHBOARD MODE ====================
        with tab3:
            st.markdown("### Interactive Data Dashboard")
            
            # Generate sample charts on first load
            if 'sample_charts' not in st.session_state or st.button("🔄 Generate New Sample Charts"):
                st.session_state. sample_charts = generate_sample_charts(data)

            # Sample charts
            if st.session_state.sample_charts:
                st.markdown("#### 📈 Sample Visualizations")
                for i, (chart_type, x, y) in enumerate(st.session_state.sample_charts):
                    try:
                        fig = None
                        if chart_type == "Scatter":
                            fig = px.scatter(data, x=x, y=y, title=f"Scatter Plot:  {x} vs {y}")
                        elif chart_type == "Line":
                            fig = px. line(data, x=x, y=y, title=f"Line Plot: {x} vs {y}")
                        elif chart_type == "Bar":
                            agg_data = data.groupby(x)[y].mean().reset_index()
                            fig = px.bar(agg_data, x=x, y=y, title=f"Bar Plot: Average {y} by {x}")
                        
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.warning(f"⚠️ Could not generate {chart_type} chart: {str(e)}")

            st.markdown("---")
            st.markdown("#### 🎨 Create Custom Chart")
            
            # Chart configuration
            col_chart1, col_chart2 = st.columns(2)
            
            with col_chart1:
                chart_type = st.selectbox("Chart Type:", 
                    ["Bar", "Line", "Scatter", "Pie", "Area", "Box", "Violin", "Histogram", "Heatmap"])
            
            with col_chart2:
                color_col = st.selectbox("Color By (optional):", 
                    ["None"] + data.columns.tolist(), key="color_select")
            
            col_axis1, col_axis2 = st.columns(2)
            
            with col_axis1:
                x_axis = st.selectbox("X-axis:", data.columns, key="x_axis_dash")
            
            with col_axis2:
                y_axis = st.selectbox("Y-axis:", data.columns, key="y_axis_dash")
            
            # Special handling for heatmap
            if chart_type == "Heatmap": 
                numeric_cols = data.select_dtypes(include=['number']).columns
                if len(numeric_cols) > 0:
                    z_axis = st.selectbox("Z-axis (values):", numeric_cols, key="z_axis_dash")
                else:
                    st.warning("⚠️ No numeric columns available for heatmap values")
                    z_axis = None

            # Filters section
            with st.expander("🔧 Apply Filters"):
                filtered_data = data.copy()
                
                # Filter for X-axis
                st.markdown(f"**Filter {x_axis}:**")
                if data[x_axis].dtype == 'object':
                    x_categories = st.multiselect(f"Select {x_axis} values:", 
                        data[x_axis].unique(), key="x_filter_dash")
                    if x_categories:
                        filtered_data = filtered_data[filtered_data[x_axis].isin(x_categories)]
                else:
                    x_min = float(data[x_axis].min())
                    x_max = float(data[x_axis].max())
                    if x_min != x_max:
                        x_range = st.slider(f"{x_axis} range:", x_min, x_max, (x_min, x_max), key="x_slider_dash")
                        filtered_data = filtered_data[(filtered_data[x_axis] >= x_range[0]) & 
                                                     (filtered_data[x_axis] <= x_range[1])]
                
                # Filter for Y-axis (if not heatmap)
                if chart_type != "Histogram": 
                    st.markdown(f"**Filter {y_axis}:**")
                    if data[y_axis].dtype == 'object':
                        y_categories = st.multiselect(f"Select {y_axis} values:", 
                            data[y_axis].unique(), key="y_filter_dash")
                        if y_categories:
                            filtered_data = filtered_data[filtered_data[y_axis]. isin(y_categories)]
                    else:
                        y_min = float(data[y_axis].min())
                        y_max = float(data[y_axis].max())
                        if y_min != y_max:
                            y_range = st.slider(f"{y_axis} range:", y_min, y_max, (y_min, y_max), key="y_slider_dash")
                            filtered_data = filtered_data[(filtered_data[y_axis] >= y_range[0]) & 
                                                         (filtered_data[y_axis] <= y_range[1])]
                
                st.info(f"📊 Filtered data: {len(filtered_data):,} rows (from {len(data):,} total)")

            if st.button("🎨 Generate Custom Chart", key="gen_chart_button"):
                if filtered_data.empty:
                    st.warning("⚠️ No data matches the current filters.  Please adjust your filter settings.")
                else:
                    try:
                        fig = None
                        color_param = None if color_col == "None" else color_col
                        
                        if chart_type == "Bar":
                            if filtered_data[x_axis].dtype == 'object':
                                agg_data = filtered_data. groupby(x_axis)[y_axis].mean().reset_index()
                                fig = px.bar(agg_data, x=x_axis, y=y_axis, color=color_param,
                                           title=f"Bar Chart: {y_axis} by {x_axis}")
                            else:
                                fig = px.bar(filtered_data, x=x_axis, y=y_axis, color=color_param,
                                           title=f"Bar Chart:  {x_axis} vs {y_axis}")
                        
                        elif chart_type == "Line": 
                            fig = px.line(filtered_data, x=x_axis, y=y_axis, color=color_param,
                                        title=f"Line Chart: {x_axis} vs {y_axis}")
                        
                        elif chart_type == "Scatter":
                            fig = px. scatter(filtered_data, x=x_axis, y=y_axis, color=color_param,
                                           title=f"Scatter Plot: {x_axis} vs {y_axis}")
                        
                        elif chart_type == "Pie":
                            if filtered_data[x_axis].dtype == 'object':
                                pie_data = filtered_data.groupby(x_axis)[y_axis].sum().reset_index()
                                fig = px.pie(pie_data, names=x_axis, values=y_axis,
                                           title=f"Pie Chart: {y_axis} by {x_axis}")
                            else: 
                                fig = px.pie(filtered_data, names=x_axis, values=y_axis,
                                           title=f"Pie Chart: {x_axis} vs {y_axis}")
                        
                        elif chart_type == "Area":
                            fig = px.area(filtered_data, x=x_axis, y=y_axis, color=color_param,
                                        title=f"Area Chart: {x_axis} vs {y_axis}")
                        
                        elif chart_type == "Box": 
                            fig = px.box(filtered_data, x=x_axis, y=y_axis, color=color_param,
                                       title=f"Box Plot: {y_axis} by {x_axis}")
                        
                        elif chart_type == "Violin": 
                            fig = px.violin(filtered_data, x=x_axis, y=y_axis, color=color_param,
                                          title=f"Violin Plot: {y_axis} by {x_axis}")
                        
                        elif chart_type == "Histogram": 
                            fig = px.histogram(filtered_data, x=x_axis, color=color_param,
                                             title=f"Histogram:  {x_axis}")
                        
                        elif chart_type == "Heatmap" and z_axis:
                            pivot_data = filtered_data.pivot_table(
                                index=y_axis, columns=x_axis, values=z_axis, aggfunc='mean')
                            fig = px.imshow(pivot_data, 
                                          title=f"Heatmap:  {z_axis} by {x_axis} and {y_axis}",
                                          labels=dict(color=z_axis),
                                          aspect="auto")
                        
                        if fig:
                            fig.update_layout(height=500)
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Show filtered data preview
                            with st.expander("📋 View Filtered Data"):
                                st.dataframe(filtered_data. head(50), use_container_width=True)
                                st.write(f"Showing first 50 of {len(filtered_data):,} rows")
                                
                                # Download filtered data
                                csv = filtered_data.to_csv(index=False)
                                st.download_button(
                                    label="📥 Download Filtered Data as CSV",
                                    data=csv,
                                    file_name="filtered_data.csv",
                                    mime="text/csv"
                                )
                        
                    except Exception as e:
                        st.error(f"❌ Error generating chart: {str(e)}")
                        st.info("💡 Try selecting different columns or adjusting your filters.")

    else:
        st.markdown("""
        <div class='info-box'>
            <h3>👋 Welcome to CSV Analyzer! </h3>
            <p>Upload a CSV file to get started with: </p>
            <ul>
                <li>🗣️ <b>Natural Language Queries</b> - Ask questions in plain English</li>
                <li>🔍 <b>SQL Queries</b> - Generate and execute SQL queries</li>
                <li>📊 <b>Interactive Dashboard</b> - Create beautiful visualizations</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Built with Streamlit 🎈 | Powered by OpenAI 🤖 | Made with ❤️</p>
</div>
""", unsafe_allow_html=True)
