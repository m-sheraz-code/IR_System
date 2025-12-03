import csv
import sys
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
import numpy as np

# Configure console encoding for Windows
if sys.platform == 'win32':
    try:
        # Set UTF-8 encoding for console output
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except:
        # Fallback for older Python versions
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'replace')
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'replace')


def load_articles(filename):
    """
    Load articles from CSV file and combine title and content
    
    Args:
        filename: Path to the CSV file
        
    Returns:
        documents: List of combined document strings
        titles: List of article titles
    """
    documents = []
    titles = []
    
    print(f"Loading articles from {filename}...")
    
    try:
        # Use utf-8 with errors='replace' to handle any encoding issues
        with open(filename, 'r', encoding='utf-8', errors='replace') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                # Combine heading and article content
                heading = row.get('Heading', '').strip()
                content = row.get('Article', '').strip()
                
                # Skip empty rows
                if not heading and not content:
                    continue
                
                # Create combined document
                combined = f"{heading} {content}"
                
                documents.append(combined)
                titles.append(heading if heading else "Untitled")
        
        print(f"✓ Loaded {len(documents)} articles successfully!\n")
        return documents, titles
        
    except FileNotFoundError:
        print(f"Error: {filename} not found!")
        return [], []
    except Exception as e:
        print(f"Error loading articles: {e}")
        import traceback
        traceback.print_exc()
        return [], []


def preprocess_text(text):
    """
    Simple preprocessing: lowercase only
    
    Args:
        text: Input text string
        
    Returns:
        Preprocessed text (lowercased)
    """
    return text.lower()


def build_tfidf_model(documents):
    """
    Build TF-IDF vectorizer model
    
    Args:
        documents: List of document strings
        
    Returns:
        vectorizer: Fitted TfidfVectorizer
        tfidf_matrix: Document-term matrix
    """
    print("Building TF-IDF model...")
    
    # Create and fit TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        lowercase=True,
        max_features=5000,  # Limit vocabulary size
        stop_words='english'  # Remove common English stop words
    )
    
    tfidf_matrix = vectorizer.fit_transform(documents)
    
    print(f"✓ TF-IDF model built with {tfidf_matrix.shape[1]} features\n")
    
    return vectorizer, tfidf_matrix


def build_bm25_model(documents):
    """
    Build BM25 model
    
    Args:
        documents: List of document strings
        
    Returns:
        bm25: Fitted BM25Okapi model
    """
    print("Building BM25 model...")
    
    # Tokenize documents (simple split by whitespace)
    tokenized_docs = [doc.lower().split() for doc in documents]
    
    # Create BM25 model
    bm25 = BM25Okapi(tokenized_docs)
    
    print(f"✓ BM25 model built with {len(tokenized_docs)} documents\n")
    
    return bm25


def hybrid_search(query, vectorizer, tfidf_matrix, bm25, documents, titles, top_k=5):
    """
    Perform hybrid search combining TF-IDF and BM25
    
    Args:
        query: Search query string
        vectorizer: Fitted TfidfVectorizer
        tfidf_matrix: Document-term matrix
        bm25: Fitted BM25 model
        documents: List of document strings
        titles: List of document titles
        top_k: Number of top results to return
        
    Returns:
        List of tuples (doc_index, title, snippet, combined_score)
    """
    # Preprocess query
    processed_query = preprocess_text(query)
    
    # 1. Compute TF-IDF similarity
    query_vector = vectorizer.transform([processed_query])
    tfidf_scores = cosine_similarity(query_vector, tfidf_matrix).flatten()
    
    # 2. Compute BM25 scores
    tokenized_query = processed_query.split()
    bm25_scores = bm25.get_scores(tokenized_query)
    
    # Normalize BM25 scores to [0, 1] range
    if bm25_scores.max() > 0:
        bm25_scores_normalized = bm25_scores / bm25_scores.max()
    else:
        bm25_scores_normalized = bm25_scores
    
    # 3. Combine scores (50% TF-IDF + 50% BM25)
    combined_scores = 0.5 * tfidf_scores + 0.5 * bm25_scores_normalized
    
    # 4. Get top-k results
    top_indices = np.argsort(combined_scores)[::-1][:top_k]
    
    # 5. Prepare results
    results = []
    for idx in top_indices:
        title = titles[idx]
        content = documents[idx]
        
        # Extract snippet (first 150-200 characters of content)
        snippet = content[:200].strip()
        if len(content) > 200:
            snippet += "..."
        
        score = combined_scores[idx]
        
        results.append((idx, title, snippet, score))
    
    return results


def display_results(results, query):
    """
    Display search results in a formatted way
    
    Args:
        results: List of search results
        query: Original search query
    """
    print("\n" + "="*80)
    print(f"Search Results for: '{query}'")
    print("="*80 + "\n")
    
    if not results:
        print("No results found.\n")
        return
    
    for rank, (doc_idx, title, snippet, score) in enumerate(results, 1):
        print(f"Rank {rank} | Score: {score:.4f} | Document Index: {doc_idx}")
        print(f"Title: {title}")
        print(f"Snippet: {snippet}")
        print("-" * 80 + "\n")


def main():
    """
    Main function to run the IR system with menu-driven interface
    """
    # Display welcome banner
    print("\n" + "="*80)
    print(" " * 20 + "INFORMATION RETRIEVAL SYSTEM")
    print(" " * 25 + "Hybrid Search Engine")
    print(" " * 22 + "(TF-IDF + BM25 Algorithm)")
    print("="*80 + "\n")
    
    # Load articles
    documents, titles = load_articles('Articles.csv')
    
    if not documents:
        print("❌ Failed to load articles. Exiting...")
        return
    
    # Preprocess all documents
    print("Preprocessing documents...")
    preprocessed_docs = [preprocess_text(doc) for doc in documents]
    print(f"✓ Preprocessed {len(preprocessed_docs)} documents\n")
    
    # Build TF-IDF model
    vectorizer, tfidf_matrix = build_tfidf_model(preprocessed_docs)
    
    # Build BM25 model
    bm25 = build_bm25_model(preprocessed_docs)
    
    # Display menu
    print("="*80)
    print("✓ System Ready!")
    print("="*80)
    print("\n📋 MENU:")
    print("  • Enter your search query to find relevant articles")
    print("  • Type 'exit' or 'quit' to close the system")
    print("  • Type 'help' for search tips")
    print("="*80 + "\n")
    
    # Search loop
    while True:
        try:
            # Get user query
            query = input("🔍 Enter search query: ").strip()
            
            # Check for exit commands
            if query.lower() in ['exit', 'quit', 'q']:
                print("\n" + "="*80)
                print("Thank you for using the Information Retrieval System!")
                print("="*80 + "\n")
                break
            
            # Show help
            if query.lower() == 'help':
                print("\n" + "-"*80)
                print("SEARCH TIPS:")
                print("  • Use keywords related to your topic (e.g., 'technology innovation')")
                print("  • Combine multiple terms for better results (e.g., 'sports cricket')")
                print("  • The system searches both article titles and content")
                print("  • Results are ranked by relevance score (0-1)")
                print("-"*80 + "\n")
                continue
            
            # Skip empty queries
            if not query:
                print("⚠️  Please enter a valid query.\n")
                continue
            
            # Perform hybrid search
            results = hybrid_search(
                query=query,
                vectorizer=vectorizer,
                tfidf_matrix=tfidf_matrix,
                bm25=bm25,
                documents=documents,
                titles=titles,
                top_k=5
            )
            
            # Display results
            display_results(results, query)
            
        except KeyboardInterrupt:
            print("\n\n" + "="*80)
            print("System interrupted. Goodbye!")
            print("="*80 + "\n")
            break
        except Exception as e:
            print(f"\n❌ An error occurred: {e}")
            print("Please try again.\n")


if __name__ == "__main__":
    main()
