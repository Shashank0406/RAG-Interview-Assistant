# 🤖 Gen AI Interview Assistant

Hey there! 👋

So I built this thing for my Gen AI interview prep, and it turned out pretty cool. It's basically a complete RAG system that helps you chat with your documents using fancy AI models. Think of it as a smart study buddy that actually understands what you're asking about.

Built with Python, Streamlit, LangChain, and ChromaDB because those are the tools everyone talks about in interviews anyway.

## 🎯 What This Thing Does

- **Smart Document Chat**: Upload docs and ask questions - it finds relevant answers
- **Pretty Web Interface**: Looks nice, works smoothly, no ugly CLI stuff
- **Multiple AI Models**: Supports GPT and Claude (because why pick just one?)
- **Handles Files**: Text and PDF files, chunks them up smartly
- **Remembers Stuff**: Keeps track of your conversation history
- **Actually Works**: Not just a demo - real vector search and everything

## 🚀 Getting Started (It's Actually Pretty Easy)

### 1. Set Up Your Environment

First things first - let's get your computer ready:

```bash
# Create a fresh conda environment (trust me, this saves headaches)
conda create -n genai-interview python=3.11 -y
conda activate genai-interview

# Grab all the dependencies
pip install -r requirements.txt
```

### 2. Fire It Up!

```bash
# Run the app
python run_app.py
```

Boom! Your browser should open up at `http://localhost:8501` with a shiny new interface.

### 3. API Keys (Makes It Way Smarter)

Want the full AI experience? Add your API keys. Without them, it still works but just does document retrieval.

```bash
# For Linux/Mac folks
export OPENAI_API_KEY="your-openai-key-here"
export ANTHROPIC_API_KEY="your-anthropic-key-here"

# Windows PowerShell users
$env:OPENAI_API_KEY="your-openai-key-here"
$env:ANTHROPIC_API_KEY="your-anthropic-key-here"
```

Don't have keys yet? That's fine - the retrieval part still works great for testing!

## 📁 What's In The Box

```
genai-interview-project/
├── src/
│   ├── rag_system.py          # The brains - handles all the RAG magic
│   ├── app.py                 # The pretty face - Streamlit interface
│   └── document_processor.py  # File wrangling utilities
├── data/
│   ├── sample_docs/           # Some ML/AI docs to play with
│   └── chroma_db/            # Where all the vector embeddings live
├── config/
│   └── settings.py           # All the configuration knobs
├── tests/
│   └── test_rag_system.py    # Makes sure stuff doesn't break
├── notebooks/                # Jupyter playground for experiments
├── requirements.txt          # Python packages you need
├── run_app.py               # Quick launcher script
└── README.md                # This thing you're reading
```

## 🔧 How The Heck It Works

### The RAG Brain (`src/rag_system.py`)
- **DocumentProcessor**: Grabs your files, breaks them into bite-sized chunks
- **VectorStore**: Talks to ChromaDB, stores and searches embeddings
- **LLMManager**: Handles different AI models (GPT, Claude, etc.)
- **RAGSystem**: The main conductor that makes everything dance

### The Shiny Interface (`src/app.py`)
- Chat with your documents like they're your best friend
- Add API keys on the fly
- Remembers what you talked about
- Shows you stats because why not

## 🎮 Playing Around With The Code

### Basic Usage (Super Simple)
```python
from src.rag_system import create_rag_system

# Get the system ready
rag = create_rag_system()

# Feed it some documents
rag.ingest_documents(["./data/sample_docs/"])

# Ask it stuff!
response = rag.query("What is RAG?")
print(response["answer"])  # Should give you a smart answer
```

### Advanced Mode (For When You Want To Tweak Stuff)
```python
from src.rag_system import RAGSystem, RAGConfig

# Customize everything
config = RAGConfig(
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    chunk_size=1000,  # How big you want your text chunks
    openai_api_key="your-secret-key"
)

# Make it yours
rag_system = RAGSystem(config)
```

## 📚 What's Already In There

I threw in some sample documents so you can test stuff right away. Covers:
- ML basics (you know, the important stuff)
- Deep learning (neural nets and all that jazz)
- Generative AI (the fun part!)
- RAG systems (what this whole thing is about)
- How to write good prompts (pro tip: be specific)
- Evaluating AI models (because you gotta measure stuff)
- Interview prep topics (cheat sheet for landing that job)

## 🧪 Testing (Make Sure It Works)

Run the tests to see if everything's working:
```bash
python tests/test_rag_system.py
```

It'll show you some queries and what the system finds. No API keys needed for this part!

## 🔍 Interview Topics This Covers

This project touches on all the hot topics you'll get asked about:

- **RAG Design**: How retrieval systems work, vector databases, good prompts
- **AI Integration**: Working with different LLM providers, API wrangling
- **Vector Stuff**: Embeddings, similarity search, making it fast
- **File Handling**: Breaking up documents, keeping metadata intact
- **System Design**: Building stuff that scales, deployment thoughts
- **Measuring Success**: Metrics, testing, knowing if your AI is any good

## 🚢 Getting It Online

### Docker Deployment

**Quick Start:**
```bash
# Build and run with Docker Compose
docker-compose up --build

# Or build and run manually
docker build -t genai-interview .
docker run -p 8501:8501 -v $(pwd)/data:/app/data genai-interview
```

**With API Keys:**
```bash
# Set environment variables
OPENAI_API_KEY=your_key ANTHROPIC_API_KEY=your_key docker-compose up

# Or create .env file
echo "OPENAI_API_KEY=your_key" > .env
echo "ANTHROPIC_API_KEY=your_key" >> .env
docker-compose up
```

**Production Setup:**
```bash
# Run with nginx reverse proxy
docker-compose --profile nginx up --build

# Scale the application
docker-compose up --scale genai-assistant=3
```

### Cloud Options
- **Streamlit Cloud**: Just push to GitHub and boom - it deploys automatically
- **Heroku**: Container-friendly deployment with `git push heroku main`
- **AWS/GCP**: For when you need it to handle millions of users

## 🤝 Want To Help Make It Better?

Cool! Here's how:
1. Fork this repo
2. Make your changes on a new branch
3. Add some tests so we know it works
4. Send me a pull request

## 📄 Legal Stuff

MIT License - do whatever you want with this, just don't blame me if it breaks. Use it for learning, interviews, whatever!

## 🎯 Interview Prep Tips

Look, I built this to help with Gen AI interviews, and it covers the important stuff:

- **Full-Stack ML**: Data in, answers out, pretty interface
- **Production Ready**: Error handling, logging, all that good stuff
- **Best Practices**: Clean code, tests, documentation
- **Real Skills**: APIs, databases, web dev, AI models

**Questions You'll Probably Get:**
- "Design a RAG system for me"
- "What's the diff between embeddings and fine-tuning?"
- "How do you chunk long documents?"
- "When to use GPT vs Claude?"
- "How do you know if your RAG is working?"

---

**Go crush those interviews! 🚀**

*P.S. Built this while prepping for Gen AI roles - hope it helps you too!*