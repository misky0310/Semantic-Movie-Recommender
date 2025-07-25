# 🎬 Semantic Movie Recommender

A content-based movie recommendation system that leverages **semantic embeddings**, **zero-shot genre classification**, and **LLM-based reranking and explanation** to produce friendly, personalized movie recommendations.

---

## 🌟 Features

### Core Capabilities
- **Semantic Understanding**: Uses advanced embeddings to understand movie plots and user preferences
- **Zero-Shot Genre Classification**: Automatically predicts relevant genres from natural language queries
- **LLM-Powered Explanations**: Provides personalized, conversational explanations for each recommendation
- **Interactive UI**: Beautiful Gradio interface with movie posters and detailed recommendations
- **Vector Search**: Fast similarity search using Chroma vector database

### Key Technologies
- **Ollama** (nomic-embed-text) for semantic embeddings
- **Chroma DB** for vector storage and similarity search
- **GROQ API** for LLM-based reranking and explanations
- **Gradio** for interactive web interface
- **Zero-shot classification** for genre prediction

---

## 🗂️ Project Structure

```text
Semantic-Movie-Recommender/
├── notebooks/
│   ├── data-exploration.ipynb        # Data cleaning and exploration
│   ├── vector-search.ipynb          # Vector generation and testing
│   └── cleaned_movies_dataset.csv   # Final processed dataset
├── utils/
│   ├── predict_genres.py            # Zero-shot genre classification
│   └── rerank_and_explain.py        # LLM reranking & explanation logic
├── main.py                          # Main pipeline: builds vector store & API
├── dashboard.py                     # Gradio UI for interactive recommendations
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

### Directory Details

- **`notebooks/`**: Jupyter notebooks for data preprocessing and experimentation
  - Contains the final cleaned CSV dataset ready for embedding
- **`utils/`**: Backend helper functions and core recommendation logic
- **`main.py`**: Orchestrates the entire pipeline from data loading to vector store creation
- **`dashboard.py`**: User-facing Gradio application for interactive movie discovery

---

## 🚀 Setup & Installation

### Prerequisites
- Python 3.8 or higher
- [Ollama](https://ollama.ai/) installed and running locally
- GROQ API key (get one from [GROQ Console](https://console.groq.com/))

### Step-by-Step Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/misky0310/Semantic-Movie-Recommender.git
   cd Semantic-Movie-Recommender
   ```

2. **Create and activate virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   ```
   Edit `.env` and add your GROQ API key:
   ```ini
   GROQ_API_KEY=your_groq_api_key_here
   ```

5. **Install and start Ollama**
   ```bash
   # Install Ollama (if not already installed)
   curl -fsSL https://ollama.ai/install.sh | sh
   
   # Pull the embedding model
   ollama pull nomic-embed-text
   ```

---

## 🧠 How It Works

### System Architecture

The recommendation system follows a sophisticated multi-stage pipeline:

1. **Data Preprocessing**: Movies dataset is cleaned and genre-encoded
2. **Embedding Generation**: Each movie overview is embedded using Ollama's nomic-embed-text model
3. **Vector Storage**: Embeddings are stored in Chroma DB for fast similarity search
4. **Query Processing**: For each user query:
   - **Genre Prediction**: Zero-shot classification predicts relevant genres
   - **Similarity Search**: Vector search finds semantically similar movies
   - **LLM Reranking**: Advanced reranking and explanation generation
   - **UI Display**: Results shown with posters, titles, and personalized explanations

    
## 🧾 Usage

### 1. Initialize the System (Run Once)

Generate the Chroma DB and embeddings:

```bash
python main.py
```

This process will:
- Load and process the movie dataset
- Generate semantic embeddings for all movie overviews
- Create and populate the Chroma vector database
- Set up the recommendation pipeline

### 2. Launch Interactive Interface

Start the Gradio web application:

```bash
python dashboard.py
```

Open the displayed URL (typically `http://127.0.0.1:7860`) in your browser to start exploring movie recommendations!

### 3. Using the Web Interface

1. **Enter your movie preference** in natural language
   - Example: "A sci-fi thriller about artificial intelligence"
   - Example: "Romantic comedy set in Paris"
   - Example: "Dark psychological horror with plot twists"

2. **Adjust settings** (optional):
   - Number of recommendations (1-10)
   - Similarity threshold
   - Genre preferences

3. **Get personalized recommendations** with:
   - Movie posters
   - Detailed explanations
   - Similarity scores
   - Genre classifications

---

## 🧪 Examples & Use Cases

### Example Queries

| Query | Expected Genres | Sample Recommendations |
|-------|----------------|----------------------|
| "A horror movie where the hero loses his mind" | Horror, Thriller, Psychological | *The Shining*, *Black Swan*, *Shutter Island* |
| "Lighthearted adventure with talking animals" | Animation, Family, Adventure | *Zootopia*, *Finding Nemo*, *The Lion King* |
| "Complex time travel story with multiple timelines" | Sci-Fi, Thriller, Mystery | *Inception*, *Predestination*, *Primer* |
| "Emotional drama about family relationships" | Drama, Family | *Manchester by the Sea*, *The Pursuit of Happyness* |

### Workflow Demonstration

For the query: **"A horror movie where the hero loses his mind"**

1. **Genre Prediction**: `predict_genres()` → `["Horror", "Thriller", "Psychological"]`
2. **Vector Search**: `similarity_search()` → Returns top-N semantically similar movies
3. **LLM Reranking**: `rerank_and_explain_with_llm()` → Reorders and explains each recommendation
4. **UI Display**: Shows movie cards with posters, titles, and personalized 1-2 sentence explanations

---

## 🛠️ Technical Details

### Dependencies

#### Core Libraries
- **`chromadb`**: Vector database for semantic search
- **`ollama`**: Local LLM embeddings and inference
- **`groq`**: Cloud-based LLM API for advanced reasoning
- **`gradio`**: Web interface framework
- **`pandas`**: Data manipulation and analysis
- **`numpy`**: Numerical computing

#### AI/ML Libraries
- **`transformers`**: Hugging Face transformers for zero-shot classification
- **`torch`**: PyTorch for deep learning operations
- **`sentence-transformers`**: Semantic similarity models

### Model Information

- **Embedding Model**: `nomic-embed-text` (via Ollama)
  - Dimension: 768
  - Optimized for semantic similarity tasks
  - Runs locally for privacy and speed

- **LLM for Reranking**: GROQ API
  - Fast inference for real-time recommendations
  - Advanced reasoning capabilities for explanations

### Performance Considerations

- **Vector Search**: Sub-second query times for datasets up to 100K movies
- **Embedding Generation**: One-time setup process (5-10 minutes for full dataset)
- **Memory Usage**: ~2GB RAM for typical movie datasets
- **Scalability**: Horizontal scaling possible with Chroma's distributed mode

---

## 🔧 Configuration Options

### Environment Variables

Create a `.env` file with the following options:

```ini
# Required
GROQ_API_KEY=your_groq_api_key_here

# Optional Configuration
CHROMA_DB_PATH=./chroma_db           # Vector database location
MAX_RECOMMENDATIONS=10              # Maximum recommendations to return
SIMILARITY_THRESHOLD=0.3            # Minimum similarity score
EMBEDDING_MODEL=nomic-embed-text    # Ollama embedding model
RERANK_MODEL=llama3-8b-8192        # GROQ model for reranking
```

### Customization

The system is designed to be highly customizable:

- **Dataset**: Replace `cleaned_movies_dataset.csv` with your own movie data
- **Embedding Model**: Change the Ollama model in the configuration
- **UI Theme**: Modify the Gradio interface in `dashboard.py`
- **Recommendation Logic**: Adjust algorithms in `utils/` modules

---

## 🙏 Acknowledgments

- **Ollama** team for providing excellent local LLM infrastructure
- **GROQ** for fast LLM inference capabilities  
- **Chroma** for the vector database solution
- **Gradio** for the intuitive web interface framework
- **Hugging Face** for transformer models and tools
- Movie data sources and the open-source community



*Built with ❤️ by [misky0310](https://github.com/misky0310)*