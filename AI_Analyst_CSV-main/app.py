import streamlit as st
import pandas as pd
import plotly.express as px
import random
from openai import OpenAI
import json
import os
from dotenv import load_dotenv

# -------------------------------------------------
# Load .env for LOCAL only (safe in cloud)
# -------------------------------------------------
if os.path.exists(".env"):
    load_dotenv()

# -------------------------------------------------
# SAFE OpenAI client loader
# -------------------------------------------------
def get_openai_client():
    if "OPENAI_API_KEY" in st.secrets:
        api_key = st.secrets["OPENAI_API_KEY"]
    else:
        api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not configured")

    return OpenAI(api_key=api_key.strip())

# -------------------------------------------------
# Chat with CSV
# -------------------------------------------------
def chat_with_csv(df, prompt):
    try:
        client = get_openai_client()
    except Exception as e:
        return f"Error: {str(e)}", "API key not configured"

    data_summary = {
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "shape": df.shape,
        "sample_data": df.head(3).to_dict("records"),
        "statistics": {}
    }

    numeric_cols = df.select_dtypes(include=["number"]).columns
    if len(numeric_cols) > 0:
        data_summary["statistics"] = df[numeric_cols].describe().to_dict()

    system_message = f"""
You are a data analyst assistant.

Dataset info:
- Columns: {', '.join(data_summary['columns'])}
- Shape: {data_summary['shape'][0]} rows, {data_summary['shape'][1]} columns
- Data types: {json.dumps(data_summary['dtypes'], indent=2)}

Answer clearly and concisely.
"""

    try:
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

        sql_prompt = (
            f"Based on this question: '{prompt}', "
            f"generate ONLY a SQL query (no explanation). "
            f"Assume the table name is 'data'."
        )

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

# -------------------------------------------------
# Sample charts
# -------------------------------------------------
def generate_sample_charts(data):
    charts = []
    numeric_columns = data.select_dtypes(include=["int64", "float64"]).columns
    categorical_columns = data.select_dtypes(include=["object"]).columns

    if len(numeric_columns) >= 2:
        x = random.choice(numeric_columns)
        y = random.choice([c for c in numeric_columns if c != x])
        charts.append(("Scatter", x, y))
        charts.append(("Line", x, y))

    if len(categorical_columns) >= 1 and len(numeric_columns) >= 1:
        charts.append(("Bar", random.choice(categorical_columns), random.choice(numeric_columns)))
        charts.append(("Box", random.choice(categorical_columns), random.choice(numeric_columns)))

    if len(categorical_columns) >= 2:
        x = random.choice(categorical_columns)
        y = random.choice([c for c in categorical_columns if c != x])
        charts.append(("Heatmap", x, y))

    return charts[:3]

# -------------------------------------------------
# Page config & styles
# -------------------------------------------------
st.set_page_config(layout="wide", page_title="CSV Analyzer")

st.markdown("""
<style>
.stButton button { background-color:#4CAF50; color:white; font-weight:bold; }
.stTextInput > div > div > input { background-color:#e0e0e0; }
.column-header { font-weight:bold; font-size:1.2em; margin-bottom:5px; }
.column-info { margin-left:10px; font-size:0.9em; }
</style>
""", unsafe_allow_html=True)

col1, col2 = st.columns([1, 3])

# -------------------------------------------------
# LEFT PANEL
# -------------------------------------------------
with col1:
    try:
        st.image("icon.jpg", width=150)
    except:
        st.markdown("### 📊")

    st.title("CSV Analyzer")
    st.markdown("---")

    input_csvs = st.file_uploader(
        "Upload CSV files", type=["csv"], accept_multiple_files=True
    )

    if input_csvs:
        selected_file = st.selectbox(
            "Select a CSV file", [f.name for f in input_csvs]
        )
        selected_index = [f.name for f in input_csvs].index(selected_file)

        encodings = ["utf-8", "latin-1", "iso-8859-1", "cp1252", "windows-1252"]
        delimiters = [",", ";", "\t", "|"]
        data = None

        for encoding in encodings:
            for delimiter in delimiters:
                try:
                    input_csvs[selected_index].seek(0)
                    data = pd.read_csv(
                        input_csvs[selected_index],
                        encoding=encoding,
                        delimiter=delimiter,
                        on_bad_lines="skip",
                        encoding_errors="ignore"
                    )
                    if not data.empty and len(data.columns) > 1:
                        break
                except:
                    continue
            if data is not None and not data.empty:
                break

        if data is None or data.empty:
            st.error("Could not read the CSV file.")
            st.stop()

        st.markdown("### Column Overview")
        for column in data.columns:
            with st.expander(column):
                st.markdown(f"<div class='column-header'>{column}</div>", unsafe_allow_html=True)
                st.markdown(
                    f"<div class='column-info'>Unique values: {data[column].nunique()}</div>",
                    unsafe_allow_html=True
                )

# -------------------------------------------------
# RIGHT PANEL
# -------------------------------------------------
with col2:
    if "data" in locals():
        st.markdown("## Data Preview")
        st.dataframe(data.head(100), use_container_width=True)

        st.markdown("## Chat with Your Data")
        query_type = st.radio("Select query type:", ["Natural Language", "SQL Query", "Dashboard"])

        if query_type in ["Natural Language", "SQL Query"]:
            user_input = st.text_area("Ask a question about your data:", height=100)

            if st.button("Analyze"):
                with st.spinner("Processing..."):
                    nl_result, sql_result = chat_with_csv(data, user_input)

                st.markdown("### Natural Language Result")
                st.write(nl_result)

                st.markdown("### SQL Query")
                st.code(sql_result, language="sql")

                if "chat_history" not in st.session_state:
                    st.session_state.chat_history = []

                st.session_state.chat_history.append({
                    "query": user_input,
                    "nl_response": nl_result,
                    "sql_response": sql_result
                })

        else:
            if "sample_charts" not in st.session_state:
                st.session_state.sample_charts = generate_sample_charts(data)

            st.markdown("### Sample Charts")
            for chart_type, x, y in st.session_state.sample_charts:
                try:
                    if chart_type == "Scatter":
                        fig = px.scatter(data, x=x, y=y)
                    elif chart_type == "Line":
                        fig = px.line(data, x=x, y=y)
                    elif chart_type == "Bar":
                        fig = px.bar(data, x=x, y=y)
                    elif chart_type == "Box":
                        fig = px.box(data, x=x, y=y)
                    elif chart_type == "Heatmap":
                        pivot = pd.pivot_table(data, index=x, columns=y, values=data.select_dtypes("number").columns[0], aggfunc="mean")
                        fig = px.imshow(pivot)
                    else:
                        continue

                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.warning(f"Chart error: {e}")

    else:
        st.warning("Please upload a CSV file to begin analysis.")
