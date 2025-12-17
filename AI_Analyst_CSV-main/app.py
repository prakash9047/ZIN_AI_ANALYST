import streamlit as st
import pandas as pd
import plotly.express as px
import random
from openai import OpenAI
import json
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

if "OPENAI_API_KEY" in st.secrets:
    openai_api_key = st.secrets["OPENAI_API_KEY"]
else:
    openai_api_key = os.getenv("OPENAI_API_KEY")
def chat_with_csv(df, prompt):
    """Use OpenAI to analyze CSV data"""
    if not openai_api_key:
        return "Please add your OPENAI_API_KEY to the .env file", "API key not configured"
    
    client = OpenAI(api_key=openai_api_key)
    
    # Prepare data context
    data_summary = {
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "shape": df.shape,
        "sample_data": df.head(3).to_dict('records'),
        "statistics": {}
    }
    
    # Add statistics for numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns
    if len(numeric_cols) > 0:
        stats_df = df[numeric_cols].describe()
        data_summary["statistics"] = stats_df.to_dict()
    
    # Create system message with data context
    system_message = f"""You are a data analyst assistant. Here's information about the dataset:
- Columns: {', '.join(data_summary['columns'])}
- Shape: {data_summary['shape'][0]} rows, {data_summary['shape'][1]} columns
- Data types: {json.dumps(data_summary['dtypes'], indent=2)}

Provide clear, concise answers about the data."""

    try:
        # Get natural language response using GPT-4o Mini
        nl_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        nl_result = nl_response.choices[0].message.content
        
        # Get SQL query suggestion
        sql_prompt = f"Based on this question about the data: '{prompt}', generate ONLY a SQL query (no explanation) that would answer this question. Assume the table name is 'data'."
        sql_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": sql_prompt}
            ],
            temperature=0.3,
            max_tokens=200
        )
        sql_result = sql_response.choices[0].message.content
        
        return nl_result, sql_result
    except Exception as e:
        return f"Error: {str(e)}", "Unable to generate SQL query"

def generate_sample_charts(data):
    """Generate sample chart configurations"""
    charts = []
    numeric_columns = data.select_dtypes(include=['int64', 'float64']).columns
    categorical_columns = data.select_dtypes(include=['object']).columns

    if len(numeric_columns) >= 2:
        x = random.choice(numeric_columns)
        y = random.choice([col for col in numeric_columns if col != x])
        charts.append(("Scatter", x, y))
        charts.append(("Line", x, y))

    if len(categorical_columns) >= 1 and len(numeric_columns) >= 1:
        x = random.choice(categorical_columns)
        y = random.choice(numeric_columns)
        charts.append(("Bar", x, y))
        charts.append(("Box", x, y))

    if len(categorical_columns) >= 2:
        x = random.choice(categorical_columns)
        y = random.choice([col for col in categorical_columns if col != x])
        charts.append(("Heatmap", x, y))

    return charts[:3]  # Return at most 3 charts

st.set_page_config(layout='wide', page_title="CSV Analyzer")

st.markdown("""
    <style>
    .stButton button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    .stTextInput > div > div > input {
        background-color: #e0e0e0;
    }
    .column-header {
        font-weight: bold;
        font-size: 1.2em;
        margin-bottom: 5px;
    }
    .column-info {
        margin-left: 10px;
        font-size: 0.9em;
    }
    </style>
""", unsafe_allow_html=True)

col1, col2 = st.columns([1, 3])

with col1:
    try:
        st.image("icon.jpg", width=150)
    except:
        st.markdown("### 📊")
    
    st.title("CSV Analyzer")
    st.markdown("---")

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
                    input_csvs[selected_index].seek(0)
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
            st.error("Could not read the CSV file. Please check the file format.")
            st.stop()

        st.markdown("### Column Overview")
        for column in data.columns:
            with st.expander(column):
                st.markdown(f"<div class='column-header'>{column}</div>", unsafe_allow_html=True)
                unique_values = data[column].nunique()
                st.markdown(f"<div class='column-info'>Unique values: {unique_values}</div>", unsafe_allow_html=True)
                
                if unique_values < 20:
                    st.markdown("<div class='column-info'>Categories:</div>", unsafe_allow_html=True)
                    categories = data[column].unique()
                    for category in categories:
                        st.markdown(f"<div class='column-info'>- {category}</div>", unsafe_allow_html=True)
                elif data[column].dtype == 'object':
                    st.markdown("<div class='column-info'>Sample values:</div>", unsafe_allow_html=True)
                    sample_values = data[column].sample(min(5, len(data))).tolist()
                    for value in sample_values:
                        st.markdown(f"<div class='column-info'>- {value}</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='column-info'>Range: {data[column].min()} to {data[column].max()}</div>", unsafe_allow_html=True)

with col2:
    if 'data' in locals() and data is not None:
        st.markdown("## Data Preview")
        st.dataframe(data.head(100), use_container_width=True)

        st.markdown("## Chat with Your Data")
        query_type = st.radio("Select query type:", ["Natural Language", "SQL Query", "Dashboard"])

        if query_type in ["Natural Language", "SQL Query"]:
            user_input = st.text_area("Ask a question about your data:", height=100)
            if st.button("Analyze"):
                with st.spinner("Processing your request..."):
                    nl_result, sql_result = chat_with_csv(data, user_input)
                    st.markdown("### Natural Language Result")
                    st.write(nl_result)
                    st.markdown("### SQL Query")
                    st.code(sql_result, language="sql")
                    
                    if 'chat_history' not in st.session_state:
                        st.session_state.chat_history = []
                    st.session_state.chat_history.append({
                        'query': user_input,
                        'nl_response': nl_result,
                        'sql_response': sql_result
                    })
        else:
            if 'sample_charts' not in st.session_state:
                st.session_state.sample_charts = generate_sample_charts(data)

            st.markdown("### Sample Charts")
            for i, (chart_type, x, y) in enumerate(st.session_state.sample_charts):
                fig = None
                try:
                    if chart_type == "Scatter":
                        fig = px.scatter(data, x=x, y=y, title=f"Sample Scatter Plot: {x} vs {y}")
                    elif chart_type == "Line":
                        fig = px.line(data, x=x, y=y, title=f"Sample Line Plot: {x} vs {y}")
                    elif chart_type == "Bar":
                        fig = px.bar(data, x=x, y=y, title=f"Sample Bar Plot: {x} vs {y}")
                    elif chart_type == "Box":
                        fig = px.box(data, x=x, y=y, title=f"Sample Box Plot: {x} vs {y}")
                    elif chart_type == "Heatmap":
                        numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
                        if len(numeric_cols) > 0:
                            pivot_data = pd.pivot_table(data, values=numeric_cols[0], index=x, columns=y, aggfunc='mean')
                            fig = px.imshow(pivot_data, title=f"Sample Heatmap: {x} vs {y}")

                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not generate {chart_type} chart: {str(e)}")

            st.markdown("### Modify Charts")
            chart_type = st.selectbox("Select chart type:", ["Bar", "Line", "Scatter", "Pie", "Area", "Doughnut", "Heatmap"])
            x_axis = st.selectbox("Select X-axis:", data.columns)
            y_axis = st.selectbox("Select Y-axis:", data.columns)

            if chart_type == "Heatmap":
                numeric_cols = data.select_dtypes(include=['number']).columns
                if len(numeric_cols) > 0:
                    z_axis = st.selectbox("Select Z-axis (values):", numeric_cols)
                else:
                    st.warning("No numeric columns available for heatmap values")
                    z_axis = None

            st.markdown("### Filters")
            filtered_data = data.copy()
            
            # X-axis filter
            if data[x_axis].dtype == 'object':
                x_categories = st.multiselect(f"Filter {x_axis}:", data[x_axis].unique(), key="x_filter")
                if x_categories:
                    filtered_data = filtered_data[filtered_data[x_axis].isin(x_categories)]
            else:
                x_min, x_max = st.slider(f"Filter {x_axis}:", 
                                         float(data[x_axis].min()), 
                                         float(data[x_axis].max()), 
                                         (float(data[x_axis].min()), float(data[x_axis].max())), 
                                         key="x_slider")
                filtered_data = filtered_data[(filtered_data[x_axis] >= x_min) & (filtered_data[x_axis] <= x_max)]
            
            # Y-axis filter
            if data[y_axis].dtype == 'object':
                y_categories = st.multiselect(f"Filter {y_axis}:", data[y_axis].unique(), key="y_filter")
                if y_categories:
                    filtered_data = filtered_data[filtered_data[y_axis].isin(y_categories)]
            else:
                y_min, y_max = st.slider(f"Filter {y_axis}:", 
                                         float(data[y_axis].min()), 
                                         float(data[y_axis].max()), 
                                         (float(data[y_axis].min()), float(data[y_axis].max())), 
                                         key="y_slider")
                filtered_data = filtered_data[(filtered_data[y_axis] >= y_min) & (filtered_data[y_axis] <= y_max)]

            if st.button("Generate Modified Chart"):
                if filtered_data.empty:
                    st.warning("No data matches the current filters. Please adjust your filter settings.")
                else:
                    try:
                        fig = None
                        if chart_type == "Bar":
                            fig = px.bar(filtered_data, x=x_axis, y=y_axis)
                        elif chart_type == "Line":
                            fig = px.line(filtered_data, x=x_axis, y=y_axis)
                        elif chart_type == "Scatter":
                            fig = px.scatter(filtered_data, x=x_axis, y=y_axis)
                        elif chart_type == "Pie":
                            fig = px.pie(filtered_data, names=x_axis, values=y_axis)
                        elif chart_type == "Area":
                            fig = px.area(filtered_data, x=x_axis, y=y_axis)
                        elif chart_type == "Doughnut":
                            fig = px.pie(filtered_data, names=x_axis, values=y_axis, hole=0.3)
                        elif chart_type == "Heatmap" and z_axis:
                            pivot_data = filtered_data.pivot_table(index=y_axis, columns=x_axis, values=z_axis, aggfunc='mean')
                            fig = px.imshow(pivot_data, labels=dict(color=z_axis))
                        
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                        
                        st.subheader("Filtered Data Preview")
                        st.dataframe(filtered_data.head(10))
                        st.write(f"Showing {len(filtered_data)} out of {len(data)} total rows")
                    except Exception as e:
                        st.error(f"Error generating chart: {str(e)}")
                        st.info("Try selecting different columns or adjusting your filters.")

    else:

        st.warning("Please upload a CSV file to begin analysis.")
