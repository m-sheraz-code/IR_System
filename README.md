# Simple Information Retrieval System

A beginner-friendly, single-file IR system that implements hybrid search using TF-IDF and BM25.

## Features

✅ **Single File Implementation** - Everything in `main.py`  
✅ **Hybrid Search** - Combines TF-IDF (scikit-learn) and BM25 (rank_bm25)  
✅ **Simple Preprocessing** - Lowercase normalization  
✅ **Menu-Driven Interface** - Easy-to-use command-line menu  
✅ **Clean Output** - Formatted results with titles, snippets, and scores  

## Requirements

```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install scikit-learn rank-bm25 numpy
```

## Dataset Format

The system expects `Articles.csv` with the following columns:
- `Heading` - Article title
- `Article` - Article content
- `Date` - Publication date (optional)
- `NewsType` - Category (optional)

## How It Works

1. **Load Articles**: Reads CSV and combines title + content into documents
2. **Preprocess**: Converts text to lowercase
3. **Build Models**: 
   - TF-IDF vectorizer with 5000 max features
   - BM25Okapi model
4. **Hybrid Search**: 
   - Computes TF-IDF cosine similarity
   - Computes BM25 scores
   - Combines: `score = 0.5 × TF-IDF + 0.5 × BM25`
5. **Return Top-K**: Returns top 5 ranked documents by default

## Usage

Run the system:
```bash
python main.py
```

### Menu Options

The system provides an interactive menu with the following options:

- **Search**: Enter any search query to find relevant articles
- **Help**: Type `help` to see search tips
- **Exit**: Type `exit`, `quit`, or `q` to close the system

### Example Session

```
🔍 Enter search query: technology innovation
[Results displayed...]

🔍 Enter search query: help
SEARCH TIPS:
  • Use keywords related to your topic
  • Combine multiple terms for better results
  • The system searches both article titles and content
  • Results are ranked by relevance score (0-1)

🔍 Enter search query: sports cricket
[Results displayed...]

🔍 Enter search query: exit
```

## Output Format

For each query, the system displays:
- **Rank** - Position in results (1-5)
- **Score** - Combined relevance score (0-1)
- **Document Index** - Position in original dataset
- **Title** - Article heading
- **Snippet** - First 200 characters of content

## Example Output

```
================================================================================
Search Results for: 'technology innovation'
================================================================================

Rank 1 | Score: 0.8542 | Document Index: 1234
Title: New AI Technology Transforms Healthcare
Snippet: New AI Technology Transforms Healthcare Artificial intelligence is revolutionizing medical diagnostics with innovative machine learning algorithms that can detect diseases earlier than...
--------------------------------------------------------------------------------
```

## Code Structure

```python
load_articles()      # Load and parse CSV file
preprocess_text()    # Simple lowercase normalization
build_tfidf_model()  # Create TF-IDF vectorizer
build_bm25_model()   # Create BM25 model
hybrid_search()      # Combine scores and rank results
display_results()    # Format and print output
main()              # Interactive search loop
```

## Customization

You can easily modify:
- **Top-K results**: Change `top_k=5` in `hybrid_search()` call
- **Hybrid weights**: Adjust `0.5 * tfidf + 0.5 * bm25` in `hybrid_search()`
- **TF-IDF features**: Change `max_features=5000` in `build_tfidf_model()`
- **Snippet length**: Modify `content[:200]` in `hybrid_search()`

## Notes

- The system handles encoding issues automatically (UTF-8 with fallback)
- Windows console encoding is configured for special characters
- Empty rows in CSV are skipped automatically
- Stop words are removed by TF-IDF vectorizer

## License

Free to use and modify for educational purposes.
