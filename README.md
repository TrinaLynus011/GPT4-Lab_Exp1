# E-commerce Chatbot Experiment 1

A local, terminal-based e-commerce chatbot based on the supplied experiment sheet. It uses:

- **SentenceTransformer (`all-MiniLM-L6-v2`)** to embed FAQ questions.
- **Cosine similarity** to retrieve the closest FAQ.
- **Ollama + Qwen (`qwen2.5:0.5b`)** to generate a concise response from retrieved store context.
- **TF-IDF + cosine similarity** to recommend matching products.
- **Transformers sentiment pipeline** to classify review text.

## 1. Requirements

- Windows 10/11
- Python 3.10 or 3.11 recommended
- Ollama installed
- Internet access on first run to download the embedding and sentiment models

## 2. Open the project in VS Code

Extract the ZIP, open the `ecommerce_chatbot_experiment` folder in VS Code, then open **Terminal → New Terminal**.

## 3. Create and activate a virtual environment (PowerShell)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run this once in that terminal:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## 4. Install Python libraries

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 5. Set up Ollama and Qwen

Install Ollama from https://ollama.com/download, then in a terminal run:

```powershell
ollama pull qwen2.5:0.5b
ollama run qwen2.5:0.5b
```

Wait for the model prompt, type `/bye`, and leave Ollama available in the background. If the Ollama app is running, the Python library can connect to it locally.

## 6. Run the chatbot

From the project folder:

```powershell
python app.py
```

The first run downloads `all-MiniLM-L6-v2`. The sentiment model downloads the first time you request sentiment analysis.

## 7. Try these questions

```text
How can I track my order?
I need a comfortable mouse for office work
Suggest headphones
Analyze this review: The product is excellent and works perfectly.
exit
```

## 8. CSV formats

Keep the column names unchanged:

- `ecommerce_faq.csv`: `question`, `answer`
- `products.csv`: `title`, `description`, `price`, `category`
- `reviews.csv`: `product`, `review`

Replace the sample rows with your own experiment data. Do not delete the header row.

## Notes

- The FAQ similarity threshold is set to `0.35` in `app.py`. Adjust it after testing your own FAQ dataset.
- The sample product prices and FAQs are demonstration data, not real store policies or listings.
- The supplied experiment sheet's program focuses on FAQ retrieval and Qwen responses. This project extends its stated aim to include TF-IDF product recommendations and sentiment analysis.
- The chatbot is a local prototype, not a production e-commerce service.
