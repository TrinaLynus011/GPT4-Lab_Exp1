import pandas as pd
import numpy as np
import ollama
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline

# -------------------- Configuration --------------------
DATA_DIR = "data"
FAQ_FILE = f"{DATA_DIR}/ecommerce_faq.csv"
PRODUCT_FILE = f"{DATA_DIR}/products.csv"
REVIEW_FILE = f"{DATA_DIR}/reviews.csv"

OLLAMA_MODEL = "qwen2.5:0.5b"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
FAQ_THRESHOLD = 0.35
TOP_K = 3

# -------------------- Load datasets --------------------
faq_df = pd.read_csv(FAQ_FILE).fillna("")
products_df = pd.read_csv(PRODUCT_FILE).fillna("")
reviews_df = pd.read_csv(REVIEW_FILE).fillna("")

required_faq = {"question", "answer"}
required_products = {"title", "description", "price", "category"}
required_reviews = {"product", "review"}

if not required_faq.issubset(faq_df.columns):
    raise ValueError(f"FAQ CSV must contain columns: {sorted(required_faq)}")
if not required_products.issubset(products_df.columns):
    raise ValueError(f"Products CSV must contain columns: {sorted(required_products)}")
if not required_reviews.issubset(reviews_df.columns):
    raise ValueError(f"Reviews CSV must contain columns: {sorted(required_reviews)}")

if faq_df.empty:
    raise ValueError("The FAQ dataset is empty. Add at least one FAQ row.")

# -------------------- Create FAQ embeddings --------------------
print("Loading sentence embedding model...")
embedding_model = SentenceTransformer(EMBEDDING_MODEL)
faq_embeddings = embedding_model.encode(
    faq_df["question"].astype(str).tolist(),
    convert_to_numpy=True
)

# -------------------- Create product TF-IDF vectors --------------------
product_text = (
    products_df["title"].astype(str) + " " +
    products_df["description"].astype(str) + " " +
    products_df["category"].astype(str)
).tolist()

tfidf_vectorizer = TfidfVectorizer(stop_words="english")
product_vectors = tfidf_vectorizer.fit_transform(product_text) if product_text else None

# -------------------- Sentiment model (loaded only when needed) --------------------
sentiment_model = None

def analyze_sentiment(text):
    """Classify a review as negative, neutral, or positive."""
    global sentiment_model
    if sentiment_model is None:
        print("Loading sentiment analysis model (first use may take a few minutes)...")
        sentiment_model = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest"
        )
    result = sentiment_model(text[:512])[0]
    return result["label"].upper(), float(result["score"])

# -------------------- Retrieval functions --------------------
def find_best_faq(user_question):
    query_embedding = embedding_model.encode([user_question], convert_to_numpy=True)
    scores = cosine_similarity(query_embedding, faq_embeddings)[0]
    best_index = int(np.argmax(scores))
    return (
        faq_df.iloc[best_index]["question"],
        faq_df.iloc[best_index]["answer"],
        float(scores[best_index])
    )

def recommend_products(user_query, top_k=TOP_K):
    if products_df.empty:
        return products_df
    query_vector = tfidf_vectorizer.transform([user_query])
    scores = cosine_similarity(query_vector, product_vectors)[0]
    best_indices = np.argsort(scores)[::-1][:top_k]
    result = products_df.iloc[best_indices].copy()
    result["similarity"] = scores[best_indices]
    return result[result["similarity"] > 0]

def find_related_reviews(product_name=None):
    if reviews_df.empty:
        return reviews_df
    if product_name:
        matches = reviews_df[
            reviews_df["product"].astype(str).str.contains(product_name, case=False, na=False)
        ]
        return matches.head(3)
    return reviews_df.head(3)

# -------------------- Chat response --------------------
SYSTEM_PROMPT = (
    "You are a helpful e-commerce assistant. Be concise, friendly, and professional. "
    "Use only the supplied store information when answering store-specific questions. "
    "Do not invent prices, policies, stock status, delivery dates, or product features. "
    "If the supplied information does not answer the question, say so clearly."
)

def ask_qwen(user_question, context):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Customer question: {user_question}\n\n"
                f"Relevant store information:\n{context}\n\n"
                "Answer the customer using the information above."
            )
        }
    ]
    response = ollama.chat(model=OLLAMA_MODEL, messages=messages)
    return response["message"]["content"]

def handle_query(user_question):
    query_lower = user_question.lower()

    # Sentiment analysis request
    sentiment_words = ["sentiment", "feeling", "review tone", "positive or negative",
                       "is this review", "analyze this review"]
    if any(word in query_lower for word in sentiment_words):
        label, score = analyze_sentiment(user_question)
        return f"Sentiment: {label} (confidence: {score:.2f})"

    # First try semantic FAQ matching
    matched_question, matched_answer, faq_score = find_best_faq(user_question)
    if faq_score >= FAQ_THRESHOLD:
        context = f"Matched FAQ: {matched_question}\nOfficial FAQ answer: {matched_answer}"
        return ask_qwen(user_question, context)

    # Otherwise retrieve relevant products
    recommendations = recommend_products(user_question)
    if not recommendations.empty:
        product_context = "\n".join(
            f"- {row.title}: {row.description} | Category: {row.category} | Price: {row.price}"
            for row in recommendations.itertuples(index=False)
        )
        return ask_qwen(user_question, f"Relevant products:\n{product_context}")

    # If no useful FAQ/product match, provide a safe fallback
    return (
        "I couldn't find a sufficiently relevant FAQ or product match. "
        "Could you rephrase your question or ask about a specific product?"
    )

# -------------------- Main chat loop --------------------
def main():
    print("\n======================================")
    print("       E-COMMERCE CHATBOT")
    print("======================================")
    print("Ask about store FAQs or products.")
    print("For sentiment analysis, start with: 'Analyze this review: ...'")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        try:
            user_input = input("You: ").strip()
            if user_input.lower() in {"exit", "quit"}:
                print("Bot: Thank you for visiting. Have a great day!")
                break
            if not user_input:
                continue
            print("Bot:", handle_query(user_input), "\n")
        except KeyboardInterrupt:
            print("\nBot: Chat ended.")
            break
        except Exception as error:
            print(f"Bot: Something went wrong: {error}")
            print("Check that Ollama is running and the required model is installed.\n")

if __name__ == "__main__":
    main()
