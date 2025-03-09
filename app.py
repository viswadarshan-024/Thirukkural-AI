import streamlit as st
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
import google.generativeai as genai
import os
from dotenv import load_dotenv
import base64
from PIL import Image
import io

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="தமிழ் தகவல் உதவியாளர் | Tamil Information Assistant",
    page_icon="📚",
    layout="wide"
)

# Custom CSS for the app
def apply_custom_css():
    st.markdown("""
    <style>
    .main {
        background-color: #1a1a29;
        color: white;
    }
    .stTextInput > div > div > input {
        background-color: #2d2d3a;
        color: white;
    }
    .stButton > button {
        background-color: #4d61fc;
        color: white;
    }
    .thirukkural-box {
        background-color: #2d2d3a;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 5px solid #4d61fc;
    }
    .kural-text {
        font-size: 1.5em;
        font-weight: bold;
        color: #ffcc00;
    }
    .explanation {
        margin-top: 10px;
    }
    .advice-box {
        background-color: #373750;
        padding: 15px;
        border-radius: 10px;
        margin: 15px 0;
    }
    .tab-container {
        margin-top: 20px;
    }
    .header-container {
        display: flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 20px;
    }
    .header-logo {
        height: 40px;
        margin-right: 10px;
    }
    .header-text {
        color: white;
        font-size: 2em;
    }
    .api-section {
        background-color: #2d2d3a;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Add Tamil logo
def add_logo():
    st.markdown("""
    <div class="header-container">
        <img src="https://img.icons8.com/color/48/000000/tamil-nadu.png" class="header-logo">
        <div class="header-text">தமிழ் தகவல் உதவியாளர்</div>
    </div>
    <p style="text-align: center; margin-bottom: 20px;">தமிழ் வரலாறு, இலக்கியம், பண்பாடு பற்றிய உங்கள் கேள்விகளுக்கு துல்லியமான பதில்களைப் பெறுங்கள்</p>
    """, unsafe_allow_html=True)

# Side panel for API key settings
def sidebar_settings():
    with st.sidebar:
        st.title("⚙️ அமைப்புகள்")
        
        # Gemini API Key
        gemini_api_key = st.text_input("Gemini API Key", type="password")
        
        # Google API Key
        google_api_key = st.text_input("Google API Key", type="password")
        
        # Google CSE ID
        google_cse_id = st.text_input("Google CX (Custom Search Engine ID)", type="password")
        
        if gemini_api_key and google_api_key and google_cse_id:
            st.success("அனைத்து API விசைகளும் சேமிக்கப்பட்டன!")
            
            # Save the API keys to session state
            st.session_state.gemini_api_key = gemini_api_key
            st.session_state.google_api_key = google_api_key
            st.session_state.google_cse_id = google_cse_id
            
            return True
    
    return False

# Load the vector DB and model
@st.cache_resource
def load_vector_db():
    # Load the dataframe
    df = pd.read_pickle("thirukkural_data.pkl")
    
    # Load the FAISS indices
    tamil_index = faiss.read_index("thirukkural_tamil_index.faiss")
    english_index = faiss.read_index("thirukkural_english_index.faiss")
    
    # Load the sentence transformer model
    model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
    
    return df, tamil_index, english_index, model

# Initialize Gemini API
def init_gemini_api(api_key):
    genai.configure(api_key=api_key)
    return genai.GenerativeModel('gemini-1.5-pro')

# Find relevant Thirukkurals
def find_relevant_kurals(query, df, tamil_index, english_index, model, language="both", top_k=5):
    query_embedding = model.encode([query])
    query_embedding = query_embedding / np.linalg.norm(query_embedding)
    
    results = []
    
    if language in ["tamil", "both"]:
        distances_tamil, indices_tamil = tamil_index.search(query_embedding.astype('float32'), top_k)
        for i, idx in enumerate(indices_tamil[0]):
            results.append({"index": int(idx), "distance": float(distances_tamil[0][i]), "language": "tamil"})
    
    if language in ["english", "both"]:
        distances_english, indices_english = english_index.search(query_embedding.astype('float32'), top_k)
        for i, idx in enumerate(indices_english[0]):
            results.append({"index": int(idx), "distance": float(distances_english[0][i]), "language": "english"})
    
    # Sort by distance score (higher is better for cosine similarity)
    results.sort(key=lambda x: x["distance"], reverse=True)
    
    # Return top k results
    return results[:top_k]

# Generate explanation and advice using Gemini API
def generate_gemini_response(model, query, kural_data):
    prompt = f"""
    User Query: {query}
    
    Thirukkural Information:
    Kural ID: {kural_data['ID']}
    Tamil Kural: {kural_data['Kural']}
    Tamil Explanation: {kural_data['Vilakam']}
    Tamil Detailed Explanation: {kural_data['Kalaingar_Urai']}
    
    English Couplet: {kural_data['Couplet']}
    English Explanation: {kural_data['M_Varadharajanar']}
    English Detailed Explanation: {kural_data['Solomon_Pappaiya']}
    
    Chapter: {kural_data['Chapter']}
    Section: {kural_data['Section']}
    
    I need you to:
    1. Analyze how relevant this Thirukkural is to the user's query
    2. Explain why this Thirukkural is appropriate for their situation
    3. Provide personal advice based on the wisdom of this Thirukkural
    
    Return your answer in the following format:
    <relevance_score>Score between 0-10</relevance_score>
    
    <tamil_explanation>
    விளக்கம் (தமிழில்): [Detailed explanation in Tamil about how this Thirukkural relates to the user's query]
    </tamil_explanation>
    
    <english_explanation>
    Explanation (in English): [Detailed explanation in English about how this Thirukkural relates to the user's query]
    </english_explanation>
    
    <tamil_advice>
    ஆலோசனை (தமிழில்): [Personal advice in Tamil based on the Thirukkural]
    </tamil_advice>
    
    <english_advice>
    Advice (in English): [Personal advice in English based on the Thirukkural]
    </english_advice>
    """
    
    response = model.generate_content(prompt)
    return response.text

# Extract information from Gemini response
def parse_gemini_response(response_text):
    result = {}
    
    # Extract relevance score
    relevance_tag = "<relevance_score>"
    relevance_end_tag = "</relevance_score>"
    if relevance_tag in response_text and relevance_end_tag in response_text:
        start = response_text.find(relevance_tag) + len(relevance_tag)
        end = response_text.find(relevance_end_tag)
        try:
            result["relevance_score"] = float(response_text[start:end].strip())
        except:
            result["relevance_score"] = 0
    
    # Extract Tamil explanation
    tamil_explanation_tag = "<tamil_explanation>"
    tamil_explanation_end_tag = "</tamil_explanation>"
    if tamil_explanation_tag in response_text and tamil_explanation_end_tag in response_text:
        start = response_text.find(tamil_explanation_tag) + len(tamil_explanation_tag)
        end = response_text.find(tamil_explanation_end_tag)
        result["tamil_explanation"] = response_text[start:end].strip()
    
    # Extract English explanation
    english_explanation_tag = "<english_explanation>"
    english_explanation_end_tag = "</english_explanation>"
    if english_explanation_tag in response_text and english_explanation_end_tag in response_text:
        start = response_text.find(english_explanation_tag) + len(english_explanation_tag)
        end = response_text.find(english_explanation_end_tag)
        result["english_explanation"] = response_text[start:end].strip()
    
    # Extract Tamil advice
    tamil_advice_tag = "<tamil_advice>"
    tamil_advice_end_tag = "</tamil_advice>"
    if tamil_advice_tag in response_text and tamil_advice_end_tag in response_text:
        start = response_text.find(tamil_advice_tag) + len(tamil_advice_tag)
        end = response_text.find(tamil_advice_end_tag)
        result["tamil_advice"] = response_text[start:end].strip()
    
    # Extract English advice
    english_advice_tag = "<english_advice>"
    english_advice_end_tag = "</english_advice>"
    if english_advice_tag in response_text and english_advice_end_tag in response_text:
        start = response_text.find(english_advice_tag) + len(english_advice_tag)
        end = response_text.find(english_advice_end_tag)
        result["english_advice"] = response_text[start:end].strip()
    
    return result

# Main app function
def main():
    apply_custom_css()
    add_logo()
    
    # Check if API keys are set
    api_keys_set = sidebar_settings()
    
    if not api_keys_set:
        st.warning("முதலில் சைட்பாரில் API விசைகளை உள்ளிடவும் / Please enter API keys in the sidebar first")
        return
    
    # Load vector DB and models
    try:
        df, tamil_index, english_index, model = load_vector_db()
        gemini_model = init_gemini_api(st.session_state.gemini_api_key)
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return
    
    # User query input
    query = st.text_input("உங்கள் கேள்விகளை தமிழில் கேளுங்கள்... / Ask your questions in Tamil...", key="user_query")
    
    if st.button("பதிலைக் காண") or query:
        if not query:
            st.warning("தயவுசெய்து உங்கள் கேள்வியை உள்ளிடவும் / Please enter your query")
            return
        
        with st.spinner("உங்கள் கேள்விக்கான பதிலைத் தேடுகிறேன்... / Searching for relevant Thirukkurals..."):
            # Find relevant kurals
            kural_results = find_relevant_kurals(query, df, tamil_index, english_index, model, top_k=5)
            
            if not kural_results:
                st.error("No relevant Thirukkurals found. Please try a different query.")
                return
            
            best_kural = None
            best_explanation = None
            best_score = -1
            
            # Process each result and find the most relevant one
            for result in kural_results:
                idx = result["index"]
                kural_data = df.iloc[idx].to_dict()
                
                # Generate explanation using Gemini
                gemini_response = generate_gemini_response(gemini_model, query, kural_data)
                parsed_response = parse_gemini_response(gemini_response)
                
                # Check relevance score
                relevance_score = parsed_response.get("relevance_score", 0)
                
                if relevance_score > best_score:
                    best_score = relevance_score
                    best_kural = kural_data
                    best_explanation = parsed_response
            
            if best_kural and best_explanation:
                # Create tabs for Tamil and English
                tab1, tab2 = st.tabs(["தமிழ் / Tamil", "English / ஆங்கிலம்"])
                
                # Tamil Content
                with tab1:
                    # Display the kural in a box
                    st.markdown(f"""
                    <div class="thirukkural-box">
                        <p class="kural-text">{best_kural['Kural']}</p>
                        <p><strong>திருக்குறள் எண்:</strong> {best_kural['ID']}</p>
                        <p><strong>அதிகாரம்:</strong> {best_kural['Adhigaram']} | <strong>பால்:</strong> {best_kural['Paal']} | <strong>இயல்:</strong> {best_kural['Iyal']}</p>
                        <div class="explanation">
                            <p><strong>விளக்கம்:</strong> {best_kural['Vilakam']}</p>
                            <p><strong>கலைஞர் உரை:</strong> {best_kural['Kalaingar_Urai']}</p>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Display AI explanation and advice
                    st.markdown("### குறள் தொடர்புடைய விளக்கம்")
                    st.markdown(best_explanation.get("tamil_explanation", "விளக்கம் இல்லை"))
                    
                    st.markdown("### ஆலோசனை")
                    st.markdown(f"""
                    <div class="advice-box">
                        {best_explanation.get("tamil_advice", "ஆலோசனை இல்லை")}
                    </div>
                    """, unsafe_allow_html=True)
                
                # English Content
                with tab2:
                    # Display the kural in English
                    st.markdown(f"""
                    <div class="thirukkural-box">
                        <p class="kural-text">{best_kural['Couplet']}</p>
                        <p><strong>Kural Number:</strong> {best_kural['ID']}</p>
                        <p><strong>Chapter:</strong> {best_kural['Chapter']} | <strong>Section:</strong> {best_kural['Section']}</p>
                        <div class="explanation">
                            <p><strong>Explanation:</strong> {best_kural['M_Varadharajanar']}</p>
                            <p><strong>Detailed Explanation:</strong> {best_kural['Solomon_Pappaiya']}</p>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Display AI explanation and advice in English
                    st.markdown("### Relevance to Your Query")
                    st.markdown(best_explanation.get("english_explanation", "No explanation available"))
                    
                    st.markdown("### Personal Advice")
                    st.markdown(f"""
                    <div class="advice-box">
                        {best_explanation.get("english_advice", "No advice available")}
                    </div>
                    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()