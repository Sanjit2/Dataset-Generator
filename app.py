import streamlit as st
import pandas as pd
import numpy as np
import requests
from faker import Faker
import re
from io import StringIO

fake = Faker()

# Replace this with your actual Hugging Face token
HF_API_KEY = st.secrets["general"]["huggingface_api_key"]
HF_MODEL = "microsoft/phi-2"

headers = {"Authorization": f"Bearer {HF_API_KEY}"}

# Available domains
DOMAINS = [
    "Insurance", "Stock Market", "Job", "Car Price", "Healthcare", "Education"
]

# Hugging Face column suggestion
def suggest_columns(domain, user_columns):
    prompt = (
        "You are a data science assistant. Your task is to suggest column names for a dataset.\n"
        f"The domain is: {domain}\n"
        f"Existing columns are: {', '.join(user_columns)}\n"
        "Suggest 5 additional, relevant column names.\n"
        "RESPONSE FORMAT: Your response must be ONLY a comma-separated list of column names, and nothing else.\n"
        "EXAMPLE: new_col_1, new_col_2, new_col_3, new_col_4, new_col_5"
    )
    response = requests.post(
        f"https://api-inference.huggingface.co/models/{HF_MODEL}",
        headers=headers,
        json={"inputs": prompt}
    )
    response.raise_for_status()
    result = response.json()
    
    if isinstance(result, list) and "generated_text" in result[0]:
        output = result[0]["generated_text"]
    elif isinstance(result, dict) and "generated_text" in result:
        output = result["generated_text"]
    else:
        output = result[0] if isinstance(result, list) else str(result)
    
    # Take the last part of the response, as the model might repeat the prompt
    if len(output) > len(prompt):
        output = output[len(prompt):]

    if ':' in output:
        output = output.split(':')[-1]

    potential_columns = re.split(r'[,\n]', output)
    
    columns = []
    # Keywords that often appear in conversational, non-column text
    stop_words = ['encourage', 'unfortunately', 'sure', 'here', 'are', 'is', 'not', 'etc']

    for col in potential_columns:
        # Clean up the column name
        cleaned_col = re.sub(r'^\s*\d+[\.\)]\s*|^\s*[\*\-]\s*', '', col)
        cleaned_col = cleaned_col.strip().strip('\'"`)')
        
        if cleaned_col.lower().startswith('and '):
            cleaned_col = cleaned_col[4:]
        elif cleaned_col.lower().startswith('or '):
            cleaned_col = cleaned_col[3:]

        cleaned_col = cleaned_col.strip()
        cleaned_col = cleaned_col.rstrip('.,;')

        # Validation checks
        is_valid = True
        # 1. Must be between 1 and 4 words
        if not (0 < len(cleaned_col.split()) <= 4 and cleaned_col):
            is_valid = False
        # 2. Should not contain characters that are unusual for column names
        if is_valid and re.search(r'[^\w\s_]', cleaned_col):
            is_valid = False
        # 3. Should not be a common conversational word
        if is_valid and cleaned_col.lower() in stop_words:
            is_valid = False
            
        if is_valid:
            columns.append(cleaned_col)
            
    return columns

# Smart data filler (for code-based generation)
def generate_fake_value(col_name):
    col = col_name.lower().replace("_", " ")
    if "id" in col:
        return fake.uuid4()
    elif "name" in col and "company" not in col:
        return fake.name()
    elif "email" in col:
        return fake.email()
    elif "address" in col:
        return fake.address()
    elif "city" in col:
        return fake.city()
    elif "country" in col:
        return fake.country()
    elif "zip" in col or "postcode" in col:
        return fake.zipcode()
    elif "phone" in col:
        return fake.phone_number()
    elif "date" in col or "time" in col:
        return fake.date()
    elif any(keyword in col for keyword in ["price", "amount", "salary", "income", "cost", "value", "revenue", "budget"]):
        return round(np.random.uniform(1000, 100000), 2)
    elif any(keyword in col for keyword in ["score", "rating"]):
        return round(np.random.uniform(1, 10), 2)
    elif "age" in col:
        return np.random.randint(18, 75)
    elif "year" in col:
        return np.random.randint(1990, 2024)
    elif "mileage" in col:
        return np.random.randint(1000, 200000)
    elif any(keyword in col for keyword in ["job", "title"]):
        return fake.job()
    elif "company" in col:
        return fake.company()
    elif any(keyword in col for keyword in ["ticker", "symbol"]):
        return fake.lexify(text="???").upper()
    elif "sector" in col:
        return fake.random_element(['Technology', 'Healthcare', 'Finance', 'Retail', 'Manufacturing', 'Energy', 'Automotive', 'Services'])
    elif "status" in col:
        return fake.random_element(["Approved", "Pending", "Rejected", "Active", "Inactive", "Closed", "Open"])
    elif any(keyword in col for keyword in ["type", "category"]):
        return fake.random_element(["Type A", "Type B", "Category 1", "Category 2", "Basic", "Premium", "Standard"])
    elif any(keyword in col for keyword in ["color", "colour"]):
        return fake.color_name()
    elif any(keyword in col for keyword in ["url", "website"]):
        return fake.url()
    elif any(keyword in col for keyword in ["description", "comment", "text"]):
        return fake.sentence()
    else:
        return fake.catch_phrase() # More sensible fallback

# Create data by learning from AI-generated samples
def generate_data_from_samples(columns, num_rows, sample_df):
    if num_rows <= 0:
        return pd.DataFrame(columns=columns)
        
    data = []
    for _ in range(num_rows):
        row = {}
        for col in columns:
            # If no samples are available for a column, fall back to the generic generator
            if col not in sample_df.columns or sample_df[col].dropna().empty:
                row[col] = generate_fake_value(col)
                continue

            samples = sample_df[col].dropna()
            
            try:
                # Attempt to convert to numeric to see if it's a number column
                numeric_samples = pd.to_numeric(samples)
                min_val, max_val = numeric_samples.min(), numeric_samples.max()
                
                # Check if original samples look like integers
                if (numeric_samples.astype(int) == numeric_samples).all():
                    min_val, max_val = int(min_val), int(max_val)
                    row[col] = np.random.randint(min_val, max_val + 1 if min_val < max_val else min_val + 2)
                else:
                    row[col] = np.random.uniform(min_val, max_val)

            except (ValueError, TypeError):
                # If conversion fails, treat as string/object data
                unique_samples = samples.unique()
                # If there are few unique values compared to total samples, treat as categorical
                if len(unique_samples) / len(samples) < 0.6 and len(unique_samples) > 0:
                     row[col] = np.random.choice(unique_samples)
                else:
                    # Otherwise, it's likely a high-cardinality string column (like names)
                    row[col] = generate_fake_value(col)
        data.append(row)
    return pd.DataFrame(data)

# Hybrid data generation
def generate_hybrid_dataset(columns, num_rows, domain):
    ai_rows_to_generate = min(num_rows, 5)
    local_rows_to_generate = num_rows - ai_rows_to_generate

    # 1. Generate initial rows with AI for context and quality
    col_string = ", ".join(columns)
    prompt = (
        f"Generate {ai_rows_to_generate} rows of realistic CSV data for the domain '{domain}' "
        f"with these columns: {col_string}.\n"
        "IMPORTANT: Your response must be ONLY the data in CSV format. Do NOT include a header row."
    )
    
    response = requests.post(
        f"https://api-inference.huggingface.co/models/{HF_MODEL}",
        headers=headers,
        json={"inputs": prompt, "parameters": {"max_new_tokens": 500}}
    )
    response.raise_for_status()
    result = response.json()
    
    if isinstance(result, list) and "generated_text" in result[0]:
        output = result[0]["generated_text"]
    else:
        output = str(result)
        
    if prompt in output:
        output = output.split(prompt)[-1].strip()

    ai_df = pd.DataFrame(columns=columns)
    if output:
        lines = output.strip().split('\n')
        cleaned_lines = [line for line in lines if line.count(',') >= len(columns) - 1]
        csv_data = "\n".join(cleaned_lines)
        if csv_data:
            # Use a dummy header to handle potential parsing issues
            ai_df = pd.read_csv(StringIO(csv_data), header=None, on_bad_lines='warn', names=columns, index_col=False)
            ai_df = ai_df.head(ai_rows_to_generate)

    # 2. Generate remaining rows locally, using AI samples as a guide
    rows_generated_by_ai = len(ai_df)
    local_rows_to_generate = num_rows - rows_generated_by_ai
    
    local_df = generate_data_from_samples(columns, local_rows_to_generate, ai_df)
        
    # 3. Combine AI-generated and locally-generated data
    final_df = pd.concat([ai_df, local_df], ignore_index=True)
    return final_df

# UI
st.title("🧪 AI-powered Dataset Generator (Hugging Face)")

domain = st.selectbox("📂 Choose a Domain", DOMAINS)
user_input = st.text_area("📝 Enter your own column names (comma-separated)", placeholder="e.g. user_id, age, claim_status")
num_rows = st.slider("🔢 Number of Rows", min_value=10, max_value=1000, value=100)

if st.button("✨ Suggest More Columns with Hugging Face"):
    user_columns = [c.strip() for c in user_input.split(",") if c.strip()]
    try:
        ai_columns = suggest_columns(domain, user_columns)
        all_columns = user_columns + ai_columns
        st.session_state["final_columns"] = all_columns
        st.success(f"AI suggested: {', '.join(ai_columns)}")
    except Exception as e:
        st.error(f"Error from Hugging Face API: {e}")

if "final_columns" in st.session_state:
    st.write("📌 **Choose Columns to Include:**")
    
    selected_columns = st.multiselect(
        "You can remove or keep the suggested columns below.",
        options=st.session_state["final_columns"],
        default=st.session_state["final_columns"]
    )

    if st.button("📁 Generate and Download CSV"):
        if not selected_columns:
            st.warning("Please select at least one column to generate the dataset.")
        else:
            with st.spinner("🤖 AI is generating sample rows... this might take a moment."):
                try:
                    df = generate_hybrid_dataset(selected_columns, num_rows, domain)
                    if not df.empty:
                        st.dataframe(df.head())
                        csv = df.to_csv(index=False).encode("utf-8")
                        st.download_button("📥 Download CSV", csv, file_name=f"{domain.lower()}_dataset.csv", mime="text/csv")
                except Exception as e:
                    st.error(f"An error occurred while generating data: {e}")
