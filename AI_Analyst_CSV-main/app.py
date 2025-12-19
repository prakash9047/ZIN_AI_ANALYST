import streamlit as st
import pandas as pd
import plotly.express as px
import random
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.agents. agent_types import AgentType
from langchain_core.messages import SystemMessage, HumanMessage
import pandasql as ps

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(layout='wide', page_title="CSV Analyzer")

# Get API key
def get_api_key():
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
    """Execute SQL query on pandas dataframe"""
    try:
        sql_query = sql_query.strip()
        if sql_query.endswith(';'):
            sql_query = sql_query[:-1]
        
        sql_query = sql_query.replace('FROM data', 'FROM df')
        sql_query = sql_query. replace('from data', 'from df')
        sql_query = sql_query.replace('FROM Data', 'FROM df')
        
        result = ps.sqldf(sql_query, locals())
        return result, None
    except Exception as e:   
        return None, f"SQL Error: {str(e)}"

def generate_chart_from_data(df, prompt):
    """Generate chart based on the question"""
    if not openai_api_key:
        return None
    
    try:
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            api_key=openai_api_key
        )
        
        numeric_cols = df.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
        
        chart_prompt = f"""Based on this question, determine if a chart is needed and specify chart details.  

Question: {prompt}

Available columns:  
Numeric: {', '.join(numeric_cols)}
Categorical: {', '.join(categorical_cols)}

Respond in this exact format:  
CHART:  yes/no
TYPE: bar/line/scatter/pie/histogram/box/none
X_COLUMN: column_name or none
Y_COLUMN: column_name or none
TITLE: chart title or none
"""

        response = llm.invoke([
            SystemMessage(content="You are a data visualization expert. "),
            HumanMessage(content=chart_prompt)
        ])
        
        response_text = response.content. strip()
        lines = response_text.split('\n')
        
        chart_config = {}
        for line in lines: 
            if ': ' in line:
                key, value = line.split(':', 1)
                chart_config[key. strip()] = value.strip()
        
        if chart_config.get('CHART', 'no').lower() != 'yes':
            return None
        
        chart_type = chart_config.get('TYPE', 'none').lower()
        x_col = chart_config.get('X_COLUMN', 'none')
        y_col = chart_config.get('Y_COLUMN', 'none')
        title = chart_config. get('TITLE', 'Chart')
        
        if chart_type == 'none' or x_col == 'none':  
            return None
        
        if x_col not in df.columns:
            return None
        if y_col != 'none' and y_col not in df.columns:
            return None
        
        fig = None
        
        if chart_type == 'bar':
            if y_col != 'none':
                if df[x_col].dtype == 'object':
                    agg_data = df.groupby(x_col)[y_col].mean().reset_index()
                    fig = px.bar(agg_data, x=x_col, y=y_col, title=title)
                else:
                    fig = px.bar(df, x=x_col, y=y_col, title=title)
            else:
                value_counts = df[x_col].value_counts().reset_index()
                value_counts.columns = [x_col, 'count']
                fig = px.bar(value_counts, x=x_col, y='count', title=title)
        
        elif chart_type == 'line':
            if y_col != 'none':  
                fig = px.line(df, x=x_col, y=y_col, title=title)
        
        elif chart_type == 'scatter':
            if y_col != 'none': 
                fig = px.scatter(df, x=x_col, y=y_col, title=title)
        
        elif chart_type == 'pie': 
            if df[x_col].dtype == 'object':
                if y_col != 'none':  
                    pie_data = df.groupby(x_col)[y_col].sum().reset_index()
                    fig = px.pie(pie_data, names=x_col, values=y_col, title=title)
                else:
                    value_counts = df[x_col].value_counts().reset_index()
                    value_counts. columns = [x_col, 'count']
                    fig = px.pie(value_counts, names=x_col, values='count', title=title)
        
        elif chart_type == 'histogram':
            fig = px.histogram(df, x=x_col, title=title)
        
        elif chart_type == 'box':  
            if y_col != 'none':  
                fig = px.box(df, x=x_col, y=y_col, title=title)
            else:
                fig = px. box(df, y=x_col, title=title)
        
        return fig
        
    except Exception as e:  
        return None

def chat_with_csv_natural_language(df, prompt):
    """Analyze CSV data using natural language"""
    if not openai_api_key:  
        return "⚠️ Please add your OPENAI_API_KEY", None
    
    try:  
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            api_key=openai_api_key
        )
        
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
            df_info = f"""
DataFrame Information:
- Shape: {df.shape[0]} rows, {df.shape[1]} columns
- Columns: {', '.join(df.columns. tolist())}

Statistical Summary:
{df.describe().to_string()}

First 5 rows:
{df. head().to_string()}
"""
            
            fallback_prompt = f"""{df_info}

Question: {prompt}

Please provide a detailed answer based on the data above."""
            
            response = llm.invoke([
                SystemMessage(content="You are a data analyst.  Provide clear, accurate answers based on the provided dataset."),
                HumanMessage(content=fallback_prompt)
            ])
            
            return response.content, None
        
    except Exception as e:  
        error_msg = str(e)
        if "401" in error_msg or "invalid_api_key" in error_msg:  
            return "❌ Invalid API Key", error_msg
        elif "insufficient_quota" in error_msg:  
            return "❌ API quota exceeded", error_msg
        else:  
            return f"❌ Error: {error_msg}", error_msg

def generate_sql_query(df, prompt):
    """Generate SQL query from natural language"""
    if not openai_api_key:  
        return "⚠️ Please add your OPENAI_API_KEY", None
    
    try:
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            api_key=openai_api_key
        )
        
        column_info = []
        for col in df.columns:
            dtype = str(df[col].dtype)
            sample_values = df[col].dropna().head(3).tolist()
            column_info.append(f"  - {col} ({dtype}): {sample_values}")
        
        columns_description = "\n".join(column_info)
        
        sql_prompt = f"""Generate ONLY a SQL query (no explanation) that answers this question.  

Question: {prompt}

Table name: df
Columns: 
{columns_description}

Return only the SQL query."""

        response = llm.invoke([
            SystemMessage(content="You are a SQL expert. Generate only SQL queries without explanation."),
            HumanMessage(content=sql_prompt)
        ])
        
        sql_query = response.content.strip()
        
        if "```sql" in sql_query:  
            sql_query = sql_query.split("```sql")[1].split("```")[0].strip()
        elif "```" in sql_query:
            sql_query = sql_query. split("```")[1].split("```")[0].strip()
        
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
                sql_lines. append(line)
        
        sql_query = ' '.join(sql_lines) if sql_lines else sql_query
        
        return sql_query, None
        
    except Exception as e:  
        return f"Error:  {str(e)}", str(e)

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
col1, col2 = st. columns([1, 3])

with col1:
    st.title("CSV Analyzer")
    st.markdown("---")
    
    if openai_api_key:
        st.success("✅ API Key Configured")
    else:
        st.error("❌ API Key Not Found")

    input_csvs = st.file_uploader("Upload CSV files", type=['csv'], accept_multiple_files=True)

    if input_csvs:
        selected_file = st.selectbox("Select a CSV file", [file.name for file in input_csvs])
        selected_index = [file.name for file in input_csvs].index(selected_file)
        
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
                    if not data.empty and len(data. columns) > 1:
                        break
                except:   
                    continue
            if data is not None and not data.empty:
                break
        
        if data is None or data.empty:
            st. error("Could not read the CSV file")
            st.stop()

        st.markdown("### Data Summary")
        st.write(f"**Rows:** {data.shape[0]: ,}")
        st.write(f"**Columns:** {data.shape[1]}")
        
        st.markdown("### Columns")
        for column in data.columns:
            with st.expander(column):
                col_type = str(data[column].dtype)
                unique_values = data[column].nunique()
                null_count = data[column].isnull().sum()
                
                st.write(f"**Type:** {col_type}")
                st.write(f"**Unique:** {unique_values}")
                st.write(f"**Nulls:** {null_count}")
                
                if data[column].dtype in ['int64', 'float64', 'int32', 'float32']:
                    st.write(f"**Min:** {data[column].min()}")
                    st.write(f"**Max:** {data[column].max()}")
                    st.write(f"**Mean:** {data[column].mean():.2f}")

with col2:
    if 'data' in locals() and data is not None:
        st. markdown("## Data Preview")
        st.dataframe(data.head(100), use_container_width=True)

        st.markdown("## Query Data")
        
        tab1, tab2, tab3 = st.tabs(["Natural Language", "SQL Query", "Dashboard"])
        
        # Natural Language Mode
        with tab1:
            nl_input = st.text_area("Ask a question:", height=100, key="nl_input")
            
            if st.button("Analyze", key="nl_button"):
                if nl_input. strip():
                    with st.spinner("Analyzing..."):
                        result, error = chat_with_csv_natural_language(data, nl_input)
                        
                        st.markdown("### Answer")
                        st.write(result)
                        
                        if 'nl_history' not in st.session_state:
                            st.session_state. nl_history = []
                        st.session_state.nl_history.append({
                            'question': nl_input,
                            'answer': result
                        })
                else:
                    st.warning("Please enter a question")
            
            if 'nl_history' in st. session_state and st.session_state.nl_history:
                st.markdown("---")
                st.markdown("### History")
                for i, item in enumerate(reversed(st.session_state.nl_history[-5:])):
                    with st.expander(f"Q{len(st.session_state.nl_history)-i}:  {item['question'][: 50]}..."):
                        st. write(f"**Q:** {item['question']}")
                        st.write(f"**A:** {item['answer']}")
        
        # SQL Query Mode
        with tab2:
            query_input_method = st.radio("Input method:", ["AI Generated SQL", "Write SQL Directly"], key="sql_method")
            
            if query_input_method == "AI Generated SQL":
                nl_query = st.text_area("Describe your query:", height=80, key="sql_nl_input")
                
                if st. button("Generate SQL", key="gen_sql_button"):
                    if nl_query.strip():
                        with st.spinner("Generating SQL..."):
                            sql_query, error = generate_sql_query(data, nl_query)
                            
                            if not error:
                                st.session_state.generated_sql = sql_query
                                st.markdown("### Generated SQL")
                                st.code(sql_query, language="sql")
                            else:
                                st.error(sql_query)
                    else:  
                        st.warning("Please describe what you want to query")
            
            if 'generated_sql' in st.session_state:
                default_sql = st.session_state.generated_sql
            else:
                default_sql = "SELECT * FROM df LIMIT 10"
            
            sql_query_input = st.text_area("SQL Query:", value=default_sql, height=120, key="sql_editor")
            
            col_exec, col_clear = st.columns([1, 4])
            with col_exec:
                execute_btn = st.button("Execute", key="exec_sql_button", use_container_width=True)
            with col_clear:  
                if st.button("Clear", key="clear_sql_button"):
                    if 'generated_sql' in st. session_state:
                        del st.session_state.generated_sql
                    st.rerun()
            
            if execute_btn:  
                if sql_query_input. strip():
                    with st. spinner("Executing... "):
                        result_df, error = execute_sql_query(data, sql_query_input)
                        
                        if not error:
                            st.success(f"Returned {len(result_df)} rows")
                            st.dataframe(result_df, use_container_width=True)
                            
                            csv = result_df.to_csv(index=False)
                            st.download_button(
                                label="Download Results",
                                data=csv,
                                file_name="query_results.csv",
                                mime="text/csv"
                            )
                        else:  
                            st.error(error)
                        
                        if 'sql_history' not in st.session_state:
                            st.session_state.sql_history = []
                        st.session_state.sql_history.append({
                            'query': sql_query_input,
                            'success': error is None,
                            'rows':  len(result_df) if result_df is not None else 0
                        })
                else:
                    st.warning("Please enter a SQL query")
            
            if 'sql_history' in st.session_state and st.session_state.sql_history:
                st.markdown("---")
                st.markdown("### Query History")
                for i, item in enumerate(reversed(st.session_state.sql_history[-5:])):
                    status = "✅" if item['success'] else "❌"
                    with st. expander(f"{status} Query {len(st.session_state.sql_history)-i}"):
                        st.code(item['query'], language="sql")
                        if item['success']:
                            st.write(f"Returned {item['rows']} rows")
        
        # Dashboard Mode
        with tab3:
            st.markdown("### Chart Generation Method")
            chart_method = st.radio("Select method:", ["Natural Language", "Manual Selection"], key="chart_method")
            
            if chart_method == "Natural Language":
                nl_chart_input = st.text_area("Describe the chart you want:", height=100, key="nl_chart_input")
                
                if st.button("Generate Chart from Question", key="nl_chart_button"):
                    if nl_chart_input.strip():
                        with st.spinner("Generating chart..."):
                            # Get answer
                            result, error = chat_with_csv_natural_language(data, nl_chart_input)
                            
                            st.markdown("### Answer")
                            st.write(result)
                            
                            # Generate chart
                            if not error:
                                fig = generate_chart_from_data(data, nl_chart_input)
                                if fig:
                                    st. markdown("### Chart")
                                    st.plotly_chart(fig, use_container_width=True)
                                else:
                                    st.info("No chart generated for this question")
                    else:
                        st.warning("Please describe the chart you want")
            
            else:  # Manual Selection
                if 'sample_charts' not in st.session_state or st.button("Generate Sample Charts"):
                    st.session_state.sample_charts = generate_sample_charts(data)

                if st.session_state.sample_charts:
                    st.markdown("### Sample Charts")
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
                        except Exception as e:
                            st.warning(f"Could not generate {chart_type} chart")

                st.markdown("---")
                st.markdown("### Custom Chart")
                
                col_chart1, col_chart2 = st.columns(2)
                
                with col_chart1:
                    chart_type = st.selectbox("Chart Type:", 
                        ["Bar", "Line", "Scatter", "Pie", "Area", "Box", "Histogram"])
                
                with col_chart2:
                    color_col = st.selectbox("Color By:", 
                        ["None"] + data.columns.tolist(), key="color_select")
                
                col_axis1, col_axis2 = st.columns(2)
                
                with col_axis1:
                    x_axis = st.selectbox("X-axis:", data.columns, key="x_axis_dash")
                
                with col_axis2:
                    y_axis = st.selectbox("Y-axis:", data.columns, key="y_axis_dash")

                if st.button("Generate Chart", key="gen_chart_button"):
                    try:
                        fig = None
                        color_param = None if color_col == "None" else color_col
                        
                        if chart_type == "Bar":
                            if data[x_axis].dtype == 'object':   
                                agg_data = data.groupby(x_axis)[y_axis].mean().reset_index()
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
                        st.error(f"Error:  {str(e)}")

    else:  
        st.markdown("### Welcome to CSV Analyzer")
        st.write("Upload a CSV file to get started")
