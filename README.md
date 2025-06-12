# 📚 Mini RAG Chatbot

A simple Retrieval-Augmented Generation (RAG) chatbot that answers questions about MarkAIting Pro company policies using semantic search and OpenAI's GPT models.

## 🚀 Quick Start (< 90 seconds)

### Option 1: Local Setup

```bash
# 1. Clone and setup
git clone git@github.com:dizhar/mini-rag-chatbot.git
cd mini-rag-chatbot
pip install -r requirements.txt

# 2. Add your OpenAI API key
cp .env.template .env
Edit .env and add: OPENAI_API_KEY=your_key_here

# 3. Add the 3 PDF files to data/ folder:
- data/security_policy.pdf
- data/expense_policy.pdf
- data/remote_work_policy.pdf

# 4. Run the app
streamlit run src/app.py
```

### Option 2: Docker (Recommended)

```bash
# 1. Clone repo and add PDFs to data/ folder
git clone git@github.com:dizhar/mini-rag-chatbot.git
cd mini-rag-chatbot
# Add your 3 PDFs to data/ folder

# 2. Build and run
docker build -t mini-rag-chatbot .
docker run -p 8501:8501 -e OPENAI_API_KEY=your_key_here mini-rag-chatbot
```

🌐 **Access the app at:** http://localhost:8501

## 📖 How It Works

1. **Document Processing**: Extracts text from the 3 policy PDFs and chunks them into searchable segments
2. **Semantic Search**: Uses OpenAI embeddings to find relevant document sections for your question
3. **Answer Generation**: Sends relevant context to GPT-3.5-turbo to generate accurate answers
4. **Source Attribution**: Shows which documents and pages were used to answer your question

## 💡 Example Questions

- "What's the meal allowance for international travel?"
- "Do I need MFA for remote work?"
- "What happens if I violate the security policy?"
- "How much can I expense for lodging?"
- "What are the core hours for remote work?"

## 🛠️ Technical Stack

- **Frontend**: Streamlit
- **LLM**: OpenAI GPT-3.5-turbo + text-embedding-ada-002
- **PDF Processing**: PyPDF2
- **Vector Search**: Cosine similarity with scikit-learn
- **Storage**: In-memory (no database required)

## 📁 Project Structure

```
mini-rag-chatbot/
├── .gitignore              # Git ignore file
├── .env.template          # API key template
├── requirements.txt       # Dependencies
├── Dockerfile            # Container setup
├── rag_chatbot_architecture.svg # architecture design
├── README.md             # Setup instructions
├── DECISIONS.md          # Architecture decisions
├── venv/                 # Virtual environment (ignored)
├── data/                 # PDF files
└── src/
    ├── app.py           # Streamlit UI
    ├── rag.py           # RAG logic
    └── utils.py         # PDF processing
```

## 🔧 Configuration

The app expects these environment variables:

- `OPENAI_API_KEY`: Your OpenAI API key (required)

## 🚨 Troubleshooting

**"Missing PDF files" error:**

- Ensure the 3 PDFs are in `data/` with exact names: `security_policy.pdf`, `expense_policy.pdf`, `remote_work_policy.pdf`

**"Error initializing chatbot" error:**

- Check your OpenAI API key is set correctly in `.env` file
- Verify you have sufficient OpenAI API credits

**Docker issues:**

- Make sure Docker is running
- Try: `docker system prune` to clean up if needed

## 📝 Development

Time spent: ~3 hours (within the test timeframe)

## 🌐 Live Demo

**Demo URL**: mini-rag-chatbot-axseel5nkomhn9nsdp2gje..streamlit.app

## 📋 Requirements Met

✅ Ingests and chunks 3 PDF policies  
✅ Generates embeddings for semantic search  
✅ Exposes chat interface for questions  
✅ Returns answers with source snippets  
✅ Uses in-memory storage (no external DB)  
✅ One-command Docker setup  
✅ Deployed and accessible online
