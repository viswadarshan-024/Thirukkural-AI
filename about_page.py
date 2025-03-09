import streamlit as st

def about_page():
    st.title("📖 About Thirukkural AI")
    
    st.markdown("""
    ## Overview
    
    Thirukkural AI is an interactive application that brings the ancient Tamil wisdom literature to life using modern AI technology. The app helps users find relevant Thirukkural couplets that address their personal questions and life situations, providing both the original verses and AI-generated explanations and advice.
    
    ## Features
    
    - **Bilingual Support**: Ask questions and receive answers in both Tamil and English
    - **Semantic Search**: Uses advanced vector embeddings to find the most relevant Thirukkural for your query
    - **Personalized Explanations**: AI-generated explanations that connect the ancient wisdom to your specific situation
    - **Practical Advice**: Receive personalized guidance based on the Thirukkural's principles
    - **Rich Context**: Access original text, traditional explanations, and modern interpretations
    
    ## How It Works
    
    1. **Vector Database Creation**: The application uses a multilingual sentence transformer to create vector embeddings of all 1,330 Thirukkural verses along with their explanations.
    
    2. **Query Processing**: When you ask a question, it's converted into the same vector space and compared against all verses to find the most semantically similar matches.
    
    3. **Relevance Ranking**: The system processes multiple potential matches through the LLM to determine which verse is most relevant to your specific query.
    
    4. **Response Generation**: The LLM generates personalized explanations and advice based on the selected Thirukkural, explaining how the ancient wisdom applies to your situation.
    
    ## Technology Stack
    
    - **Frontend**: Streamlit
    - **Vector Database**: FAISS (Facebook AI Similarity Search)
    - **Embeddings**: Sentence Transformers (paraphrase-multilingual-mpnet-base-v2)
    - **LLM Integration**: Groq API (using Llama-3.3-70b-versatile model)
    - **Data Processing**: Pandas, NumPy
    
    ## Use Cases
    
    - Personal guidance and advice
    - Exploring Tamil cultural wisdom
    - Learning about ethical principles
    - Finding inspiration and motivation
    - Educational tool for understanding Thirukkural
    
    ## Example Questions
    
    You can ask questions like:
    - How to choose good friends?
    - How to stay resilient during difficult times?
    - How to control anger?
    - What makes a happy family life?
    - How to achieve success in work?
    - How to be a good leader?
    
    ## Dataset
    
    This project uses the Thirukkural dataset available on Hugging Face:
    [Selvakumarduraipandian/Thirukural](https://huggingface.co/datasets/Selvakumarduraipandian/Thirukural)
    
    ## Author
    
    - Viswadarshan R R
    
    ## Special Acknowledgement
    
    - Selvakumar Duraipandian for Thirukkural Dataset
    
    ## License
    
    This project is licensed under the MIT License.
    """)
    
    # Add GitHub button
    st.markdown("""
    <div style="text-align: center; margin-top: 30px;">
        <a href="https://github.com/viswadarshan-024/Thirukkural-AI" target="_blank" style="text-decoration: none;">
            <button style="background-color: #4d61fc; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; font-weight: bold;">
                View on GitHub
            </button>
        </a>
    </div>
    """, unsafe_allow_html=True)
    
    # You may also want to add more detailed sections specific to your application
    with st.expander("What is Thirukkural?"):
        st.markdown("""
        Thirukkural is a classic Tamil language text consisting of 1,330 short couplets of seven words each. 
        Written by Thiruvalluvar, it is one of the most important works in Tamil literature and provides 
        guidance on ethics, political and economic matters, and love.
        
        The text is divided into three major sections:
        - **Aram (Virtue/Dharma)**: Chapters 1-38, dealing with moral values
        - **Porul (Wealth/Artha)**: Chapters 39-108, discussing politics and society
        - **Inbam (Love/Kama)**: Chapters 109-133, exploring aspects of human love and relationships
        
        For over 2,000 years, the Thirukkural has offered timeless wisdom that remains relevant today, 
        addressing fundamental aspects of human life and society.
        """)
    
    with st.expander("About the Technology"):
        st.markdown("""
        ### Vector Search Technology
        
        The app uses modern semantic search technology to understand your query conceptually rather than just matching keywords:
        
        1. **Embeddings**: Each Thirukkural couplet and explanation is converted into a high-dimensional vector space using a multilingual transformer model.
        
        2. **Similarity Matching**: Your question is converted into the same vector space, and the system finds Thirukkural verses closest to your query in meaning.
        
        3. **FAISS**: We use Facebook AI Similarity Search, a library for efficient similarity search and clustering of dense vectors.
        
        ### Large Language Model
        
        We use the Llama-3.3-70b-versatile model via the Groq API to:
        
        1. Evaluate multiple potential Thirukkural matches to find the most relevant one
        2. Generate personalized explanations of how the wisdom applies to your specific situation
        3. Provide practical advice based on the principles in the Thirukkural
        
        This combines the timeless wisdom of Thirukkural with the contextual understanding capabilities of modern AI.
        """)
