# Thirukkural AI
### **Experience the Profound Wisdom of Thirukkural, Reimagined**

## 📖 Overview

Thirukkural AI is an interactive application that brings the ancient Tamil wisdom literature to life using modern AI technology. The app helps users find relevant Thirukkural couplets that address their personal questions and life situations, providing both the original verses and AI-generated explanations and advice.

Try out: [Kural-AI](https://kural-ai.streamlit.app/)

## ✨ Features

- **Bilingual Support**: Ask questions and receive answers in both Tamil and English
- **Semantic Search**: Uses advanced vector embeddings to find the most relevant Thirukkural for your query
- **Personalized Explanations**: AI-generated explanations that connect the ancient wisdom to your specific situation
- **Practical Advice**: Receive personalized guidance based on the Thirukkural's principles
- **Rich Context**: Access original text, traditional explanations, and modern interpretations

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **Vector Database**: FAISS (Facebook AI Similarity Search)
- **Embeddings**: Sentence Transformers (paraphrase-multilingual-mpnet-base-v2)
- **LLM Integration**: Groq API (using Llama-3.3-70b-versatile model)
- **Data Processing**: Pandas, NumPy

## 📋 Requirements

- Python 3.8+
- Groq API key
- Internet connection for API calls

## 🚀 Installation & Setup

1. **Clone the repository**

```bash
git clone https://github.com/your-username/thirukkural-ai.git
cd thirukkural-ai
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Prepare the vector database** (optional - pre-built database files are included)

```bash
python vector_db.py
```

4. **Run the application**

```bash
streamlit run app.py
```

5. **Enter your Groq API key in the sidebar**

## 🔍 How It Works

1. **Vector Database Creation**: The application uses a multilingual sentence transformer to create vector embeddings of all 1,330 Thirukkural verses along with their explanations.

2. **Query Processing**: When you ask a question, it's converted into the same vector space and compared against all verses to find the most semantically similar matches.

3. **Relevance Ranking**: The system processes multiple potential matches through the LLM to determine which verse is most relevant to your specific query.

4. **Response Generation**: The LLM generates personalized explanations and advice based on the selected Thirukkural, explaining how the ancient wisdom applies to your situation.

## 🌐 Use Cases

- Personal guidance and advice
- Exploring Tamil cultural wisdom
- Learning about ethical principles
- Finding inspiration and motivation
- Educational tool for understanding Thirukkural

## 🧠 Dataset

This project uses the Thirukkural dataset available on Hugging Face:
[Selvakumarduraipandian/Thirukural](https://huggingface.co/datasets/Selvakumarduraipandian/Thirukural)

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

- Viswadarshan R R

## 🙏 Special Acknowledgement

- Selvakumar Duraipandian for Thirukkural Dataset
