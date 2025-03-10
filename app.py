import streamlit as st
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
import os
from dotenv import load_dotenv
import requests
import json
import time
import random

# Load environment variables
load_dotenv()

# Load multiple Groq API keys from the environment
API_KEYS = os.getenv("GROQ_API_KEYS").split(",")
if not API_KEYS:
    raise ValueError("No API keys found in the environment variables.")

# Store API keys in session state for use throughout the app
if 'api_keys' not in st.session_state:
    st.session_state.api_keys = API_KEYS

# Function to get a random API key from the pool
def get_random_api_key():
    return random.choice(st.session_state.api_keys)

# Page configuration
st.set_page_config(
    page_title="திருக்குறள் AI",
    page_icon="📖",
    layout="wide"
)

# Custom CSS for the app
def apply_custom_css():
    st.markdown(""" ... (your existing CSS code) ... """, unsafe_allow_html=True)

# Add Tamil logo
def add_logo():
    st.markdown(""" ... (your existing logo code) ... """, unsafe_allow_html=True)

# Modified sidebar - removed API key input
def sidebar_info():
    with st.sidebar:
        st.title("திருக்குறள் AI பற்றி")
        st.markdown("---")
        st.markdown(""" ... (your existing sidebar code) ... """)
        st.markdown("---")
        st.markdown("### Advanced Settings")
        st.session_state.show_process = st.checkbox("Show search process", value=False, 
                                                  help="Shows the intermediate steps in finding the most relevant Thirukkural")
        st.session_state.num_candidates = st.slider("Number of candidates", min_value=3, max_value=10, value=5,
                                                  help="Number of initial candidate kurals to retrieve")
        st.session_state.temperature = st.slider("LLM temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.1,
                                               help="Controls creativity in AI responses (higher = more creative)")

# Load the vector DB and model
@st.cache_resource
def load_vector_db():
    df = pd.read_pickle("thirukkural_data.pkl")
    tamil_index = faiss.read_index("thirukkural_tamil_index.faiss")
    english_index = faiss.read_index("thirukkural_english_index.faiss")
    model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
    return df, tamil_index, english_index, model

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
    results.sort(key=lambda x: x["distance"], reverse=True)
    return results[:top_k]

# Generate evaluation of candidate kurals using Groq API
def evaluate_candidates(api_key, query, candidate_kurals, df):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    url = "https://api.groq.com/openai/v1/chat/completions"
    candidates_text = ""
    for i, candidate in enumerate(candidate_kurals, 1):
        idx = candidate["index"]
        kural_data = df.iloc[idx].to_dict()
        candidates_text += f"""
        Candidate {i}:
        Kural ID: {kural_data['ID']}
        Tamil Kural: {kural_data['Kural']}
        English Couplet: {kural_data['Couplet']}
        Tamil Explanation: {kural_data['Vilakam']}
        English Explanation: {kural_data['M_Varadharajanar']}
        Chapter: {kural_data['Chapter']}
        Section: {kural_data['Section']}
        """
    prompt = f"""
    You are a wisdom expert familiar with Thirukkural, the ancient Tamil literature of ethical principles.
    User Query: {query}
    I have found {len(candidate_kurals)} potential Thirukkural verses that might be relevant to this query.
    {candidates_text}
    Your task is to:
    1. Analyze the user's query to understand the underlying need, concern, or question.
    2. Evaluate each candidate Thirukkural and determine how relevant it is to addressing the user's query.
    3. Select the MOST relevant Thirukkural that best addresses the user's query. If none are particularly relevant, indicate that a different Thirukkural might be better.
    4. Provide a brief analysis of why your selected Kural is most relevant to the user's needs.
    5. If you believe a completely different Thirukkural would be more appropriate (not in the candidates), please explain why, but do not include it in your ranking.
    Return your answer in the following format:
    <query_analysis>
    A brief analysis of what the user is truly seeking advice about.
    </query_analysis>
    <relevance_scores>
    Candidate 1: [Score between 0-10] - Brief reason
    Candidate 2: [Score between 0-10] - Brief reason
    ...
    </relevance_scores>
    <best_candidate>
    The number of the best candidate (e.g., "Candidate 3")
    </best_candidate>
    <rationale>
    A detailed explanation of why this Thirukkural is most appropriate for the user's query.
    </rationale>
    <need_alternative>
    Yes/No - Whether you think a completely different Thirukkural would be better suited
    </need_alternative>
    <alternative_suggestion>
    If you answered Yes above, explain what type of Thirukkural would be more appropriate.
    </alternative_suggestion>
    """
    data = {
        "model": "llama-3.3-70b-versatile",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": st.session_state.temperature,
        "max_tokens": 2048
    }
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        response_data = response.json()
        return response_data["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: {str(e)}")
        return None
    except (KeyError, IndexError) as e:
        st.error(f"Unexpected response format: {str(e)}")
        return None

# Generate explanation and advice using Groq API
def generate_response_for_kural(api_key, query, kural_data, query_analysis=""):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    url = "https://api.groq.com/openai/v1/chat/completions"
    prompt = f"""
    User Query: {query}
    {query_analysis}
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
    I need you to create a detailed, personalized response based on this Thirukkural for the user's query.
    Return your answer in the following format:
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
    data = {
        "model": "llama-3.3-70b-versatile",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": st.session_state.temperature,
        "max_tokens": 2048
    }
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        response_data = response.json()
        return response_data["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: {str(e)}")
        return None
    except (KeyError, IndexError) as e:
        st.error(f"Unexpected response format: {str(e)}")
        return None

# Parse evaluation response to get scores, best candidate, etc.
def parse_evaluation_response(response_text):
    if not response_text:
        return {}
    result = {}
    query_analysis_tag = "<query_analysis>"
    query_analysis_end_tag = "</query_analysis>"
    if query_analysis_tag in response_text and query_analysis_end_tag in response_text:
        start = response_text.find(query_analysis_tag) + len(query_analysis_tag)
        end = response_text.find(query_analysis_end_tag)
        result["query_analysis"] = response_text[start:end].strip()
    relevance_scores_tag = "<relevance_scores>"
    relevance_scores_end_tag = "</relevance_scores>"
    if relevance_scores_tag in response_text and relevance_scores_end_tag in response_text:
        start = response_text.find(relevance_scores_tag) + len(relevance_scores_tag)
        end = response_text.find(relevance_scores_end_tag)
        result["relevance_scores"] = response_text[start:end].strip()
    best_candidate_tag = "<best_candidate>"
    best_candidate_end_tag = "</best_candidate>"
    if best_candidate_tag in response_text and best_candidate_end_tag in response_text:
        start = response_text.find(best_candidate_tag) + len(best_candidate_tag)
        end = response_text.find(best_candidate_end_tag)
        best_candidate_text = response_text[start:end].strip()
        import re
        candidate_match = re.search(r'Candidate\s+(\d+)', best_candidate_text)
        if candidate_match:
            result["best_candidate"] = int(candidate_match.group(1))
        else:
            result["best_candidate"] = best_candidate_text
    rationale_tag = "<rationale>"
    rationale_end_tag = "</rationale>"
    if rationale_tag in response_text and rationale_end_tag in response_text:
        start = response_text.find(rationale_tag) + len(rationale_tag)
        end = response_text.find(rationale_end_tag)
        result["rationale"] = response_text[start:end].strip()
    need_alternative_tag = "<need_alternative>"
    need_alternative_end_tag = "</need_alternative>"
    if need_alternative_tag in response_text and need_alternative_end_tag in response_text:
        start = response_text.find(need_alternative_tag) + len(need_alternative_tag)
        end = response_text.find(need_alternative_end_tag)
        result["need_alternative"] = response_text[start:end].strip().lower() == "yes"
    alternative_suggestion_tag = "<alternative_suggestion>"
    alternative_suggestion_end_tag = "</alternative_suggestion>"
    if alternative_suggestion_tag in response_text and alternative_suggestion_end_tag in response_text:
        start = response_text.find(alternative_suggestion_tag) + len(alternative_suggestion_tag)
        end = response_text.find(alternative_suggestion_end_tag)
        result["alternative_suggestion"] = response_text[start:end].strip()
    return result

# Extract information from Groq response (from the final explanation)
def parse_explanation_response(response_text):
    if not response_text:
        return {}
    result = {}
    tamil_explanation_tag = "<tamil_explanation>"
    tamil_explanation_end_tag = "</tamil_explanation>"
    if tamil_explanation_tag in response_text and tamil_explanation_end_tag in response_text:
        start = response_text.find(tamil_explanation_tag) + len(tamil_explanation_tag)
        end = response_text.find(tamil_explanation_end_tag)
        result["tamil_explanation"] = response_text[start:end].strip()
    english_explanation_tag = "<english_explanation>"
    english_explanation_end_tag = "</english_explanation>"
    if english_explanation_tag in response_text and english_explanation_end_tag in response_text:
        start = response_text.find(english_explanation_tag) + len(english_explanation_tag)
        end = response_text.find(english_explanation_end_tag)
        result["english_explanation"] = response_text[start:end].strip()
    tamil_advice_tag = "<tamil_advice>"
    tamil_advice_end_tag = "</tamil_advice>"
    if tamil_advice_tag in response_text and tamil_advice_end_tag in response_text:
        start = response_text.find(tamil_advice_tag) + len(tamil_advice_tag)
        end = response_text.find(tamil_advice_end_tag)
        result["tamil_advice"] = response_text[start:end].strip()
    english_advice_tag = "<english_advice>"
    english_advice_end_tag = "</english_advice>"
    if english_advice_tag in response_text and english_advice_end_tag in response_text:
        start = response_text.find(english_advice_tag) + len(english_advice_tag)
        end = response_text.find(english_advice_end_tag)
        result["english_advice"] = response_text[start:end].strip()
    return result

# Parse the relevance scores from the text to get scores for visualization
def parse_relevance_scores_text(relevance_scores_text):
    import re
    scores = []
    lines = relevance_scores_text.strip().split('\n')
    for line in lines:
        match = re.search(r'Candidate\s+(\d+):\s+(\d+(?:\.\d+)?)', line)
        if match:
            candidate_num = int(match.group(1))
            score = float(match.group(2))
            reason = line.split('-', 1)[1].strip() if '-' in line else ""
            scores.append({"candidate": candidate_num, "score": score, "reason": reason})
    return scores

# Function for refined search if needed
def perform_refined_search(api_key, query, df, alternative_suggestion):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    url = "https://api.groq.com/openai/v1/chat/completions"
    prompt = f"""
    You are an expert in Thirukkural who can suggest the most appropriate Kural based on a query.
    User Query: {query}
    Based on analysis, we need a different Thirukkural than what was initially suggested.
    Additional guidance: {alternative_suggestion}
    Please recommend one specific Thirukkural that would be most relevant to this query.
    You have access to all 1,330 Thirukkural couplets. Provide the Kural ID number (between 1-1330) that you believe would be most appropriate.
    Return only the Kural ID number, nothing else. For example: 125
    """
    data = {
        "model": "llama-3.3-70b-versatile",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "max_tokens": 50
    }
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        response_data = response.json()
        kural_id_text = response_data["choices"][0]["message"]["content"].strip()
        import re
        kural_id_match = re.search(r'(\d+)', kural_id_text)
        if kural_id_match:
            kural_id = int(kural_id_match.group(1))
            if 1 <= kural_id <= 1330:
                kural_data = df[df['ID'] == kural_id].iloc[0].to_dict() if not df[df['ID'] == kural_id].empty else None
                return kural_data
        return None
    except Exception as e:
        st.error(f"Error in refined search: {str(e)}")
        return None

def add_footer():
    st.markdown(""" ... (your existing footer code) ... """, unsafe_allow_html=True)

# Function to display search results with visualization
def display_search_results(candidates, df, evaluation_results):
    if not candidates or not evaluation_results or "relevance_scores" not in evaluation_results:
        return
    st.markdown("### Search Process")
    if "query_analysis" in evaluation_results:
        st.markdown(f"""
        <div class="query-analysis">
            <strong>Query Analysis:</strong> {evaluation_results["query_analysis"]}
        </div>
        """, unsafe_allow_html=True)
    parsed_scores = parse_relevance_scores_text(evaluation_results["relevance_scores"])
    best_candidate = evaluation_results.get("best_candidate", None)
    for i, candidate in enumerate(candidates, 1):
        idx = candidate["index"]
        kural_data = df.iloc[idx].to_dict()
        score_info = next((s for s in parsed_scores if s["candidate"] == i), None)
        score = score_info["score"] if score_info else 0
        reason = score_info["reason"] if score_info else ""
        score_percentage = (score / 10) * 100
        is_selected = best_candidate == i
        class_name = "search-result-item selected" if is_selected else "search-result-item"
        st.markdown(f"""
        <div class="{class_name}">
                   <strong>Thirukkural #{kural_data['ID']} - {kural_data['Chapter']}</strong>
        <p><em>{kural_data['Couplet']}</em></p>
        <p>{kural_data['Kural']}</p>
        <div class="relevance-indicator">
            <div class="relevance-bar" style="width: {score_percentage}%;"></div>
        </div>
        <p><strong>Relevance Score:</strong> {score}/10</p>
        <p><strong>Reason:</strong> {reason}</p>
    </div>
    """, unsafe_allow_html=True)

    # Show rationale for the best selection
    if "rationale" in evaluation_results:
        st.markdown("### Selection Rationale")
        st.markdown(evaluation_results["rationale"])

    # Show if an alternative is needed
    if evaluation_results.get("need_alternative", False):
        st.markdown("### Alternative Suggestion Needed")
        st.markdown(evaluation_results.get("alternative_suggestion", ""))

# Function to display the final Thirukkural explanation
def display_thirukkural_explanation(kural_data, explanation_data, tab_option="bilingual"):
    st.markdown(f"""
    <div class="thirukkural-box">
        <h3>திருக்குறள் #{kural_data['ID']} - {kural_data['Chapter']}</h3>
        <div class="kural-text">{kural_data['Kural']}</div>
        <div><em>{kural_data['Couplet']}</em></div>
    </div>
    """, unsafe_allow_html=True)

    # Display explanations based on the selected language tab
    if tab_option == "tamil":
        st.markdown(f"""
        <div class="explanation">
            <h4>விளக்கம்:</h4>
            <p>{explanation_data.get('tamil_explanation', '')}</p>
            
            <h4>ஆலோசனை:</h4>
            <p>{explanation_data.get('tamil_advice', '')}</p>
        </div>
        """, unsafe_allow_html=True)
    elif tab_option == "english":
        st.markdown(f"""
        <div class="explanation">
            <h4>Explanation:</h4>
            <p>{explanation_data.get('english_explanation', '')}</p>
            
            <h4>Advice:</h4>
            <p>{explanation_data.get('english_advice', '')}</p>
        </div>
        """, unsafe_allow_html=True)
    else:  # bilingual
        st.markdown(f"""
        <div class="explanation">
            <h4>விளக்கம் (Explanation):</h4>
            <p>{explanation_data.get('tamil_explanation', '')}</p>
            <p>{explanation_data.get('english_explanation', '')}</p>
            
            <h4>ஆலோசனை (Advice):</h4>
            <p>{explanation_data.get('tamil_advice', '')}</p>
            <p>{explanation_data.get('english_advice', '')}</p>
        </div>
        """, unsafe_allow_html=True)

# Main app function
def main():
    # Apply custom CSS
    apply_custom_css()
    
    # Add logo and header
    add_logo()
    
    # Add sidebar information
    sidebar_info()
    
    # Load vector DB and model
    df, tamil_index, english_index, model = load_vector_db()
    
    # Main content
    st.markdown("#### Ask for wisdom and guidance based on Thirukkural")
    query = st.text_input("Enter your question or life situation:",
                         placeholder="E.g., How do I choose good friends?")
    
    if st.button("Get Wisdom"):
        if not query:
            st.warning("Please enter a question or describe your situation.")
            return
        
        with st.spinner("Finding wisdom from Thirukkural..."):
            # Step 1: Find candidate kurals using vector search
            candidates = find_relevant_kurals(
                query, df, tamil_index, english_index, model, 
                language="both", top_k=st.session_state.num_candidates
            )
            
            if not candidates:
                st.error("No relevant Thirukkural found. Please try a different query.")
                return
            
            # Step 2: Evaluate candidates with LLM
            evaluation_response = evaluate_candidates(
                get_random_api_key(), query, candidates, df
            )
            
            if not evaluation_response:
                st.error("Failed to evaluate candidates. Please try again.")
                return
            
            # Parse the evaluation response
            evaluation_results = parse_evaluation_response(evaluation_response)
            
            # Show search process if enabled
            if st.session_state.show_process:
                display_search_results(candidates, df, evaluation_results)
            
            # Step 3: Get the most relevant kural or perform refined search if needed
            if evaluation_results.get("need_alternative", False):
                st.info("Finding a better matching Thirukkural based on your query...")
                
                # Perform refined search
                refined_kural_data = perform_refined_search(
                    get_random_api_key(), 
                    query, 
                    df,
                    evaluation_results.get("alternative_suggestion", "")
                )
                
                if refined_kural_data:
                    selected_kural_data = refined_kural_data
                    st.success("Found a better matching Thirukkural!")
                else:
                    # Fallback to the best candidate from initial search
                    best_idx = candidates[evaluation_results.get("best_candidate", 0) - 1]["index"]
                    selected_kural_data = df.iloc[best_idx].to_dict()
            else:
                # Use the best candidate from evaluation
                best_candidate_index = evaluation_results.get("best_candidate", 1) - 1
                best_idx = candidates[best_candidate_index]["index"]
                selected_kural_data = df.iloc[best_idx].to_dict()
            
            # Step 4: Generate personalized explanation and advice
            explanation_response = generate_response_for_kural(
                get_random_api_key(),
                query,
                selected_kural_data,
                evaluation_results.get("query_analysis", "")
            )
            
            if not explanation_response:
                st.error("Failed to generate explanation. Please try again.")
                return
            
            # Parse the explanation response
            explanation_data = parse_explanation_response(explanation_response)
            
            # Step 5: Display the result in tabs for different languages
            st.markdown("## Wisdom from Thirukkural")
            
            tab1, tab2, tab3 = st.tabs(["Bilingual", "தமிழ் (Tamil)", "English"])
            
            with tab1:
                display_thirukkural_explanation(selected_kural_data, explanation_data, "bilingual")
            
            with tab2:
                display_thirukkural_explanation(selected_kural_data, explanation_data, "tamil")
            
            with tab3:
                display_thirukkural_explanation(selected_kural_data, explanation_data, "english")
    
    # Add related kurals suggestion feature
    if 'last_query' in st.session_state and st.session_state.last_query:
        st.markdown("---")
        if st.button("Explore Related Wisdom"):
            with st.spinner("Finding related wisdom..."):
                related_candidates = find_relevant_kurals(
                    f"More about {st.session_state.last_query}", 
                    df, tamil_index, english_index, model,
                    language="both", top_k=3
                )
                
                st.markdown("### Related Thirukkural Verses")
                
                for candidate in related_candidates:
                    idx = candidate["index"]
                    kural_data = df.iloc[idx].to_dict()
                    
                    st.markdown(f"""
                    <div class="search-result-item">
                        <strong>Thirukkural #{kural_data['ID']} - {kural_data['Chapter']}</strong>
                        <p><em>{kural_data['Couplet']}</em></p>
                        <p>{kural_data['Kural']}</p>
                    </div>
                    """, unsafe_allow_html=True)
    
    # Store current query for future reference
    if query:
        st.session_state.last_query = query
    
    # Add footer
    add_footer()

# Run the app
if __name__ == "__main__":
    main()
