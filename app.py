from flask import Flask, render_template, request, send_file, make_response
import os
import re
import PyPDF2
import docx
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import pickle
import csv
import io
from collections import Counter

app = Flask(__name__)
DOCUMENT_FOLDER = "documents"
CACHE_FILE = "preprocessed_cache.pkl"

# ====================================================
# 1. Utility: Baca file PDF, DOCX, TXT
# ====================================================
def read_text(path):
    ext = path.split('.')[-1].lower()

    # TXT
    if ext == "txt":
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    # DOCX
    elif ext == "docx":
        doc = docx.Document(path)
        return "\n".join([p.text for p in doc.paragraphs])

    # PDF
    elif ext == "pdf":
        text = ""
        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text()
        return text

    return ""


# ====================================================
# 2. Preprocessing (case folding, tokenizing, filtering)
# ====================================================
factory = StemmerFactory()
stemmer = factory.create_stemmer()

stopwords = set([
    "yang","dan","di","ke","dari","untuk","pada",
    "dengan","atau","itu","ini","karena","adalah"
])

def preprocessing(text):
    # case folding
    text = text.lower()

    # tokenizing
    tokens = re.findall(r'\b\w+\b', text)

    # filtering stopwords
    filtered = [t for t in tokens if t.isalpha() and t not in stopwords]

    # stemming
    stemmed = [stemmer.stem(t) for t in filtered]

    return {
        "casefold": text,
        "tokens": tokens,
        "filtered": filtered,
        "stemmed": stemmed
    }

# ====================================================
#  SUMMARY (Extractive Summarization using TF-IDF)
# ====================================================
def summarize_text(text, num_sentences=3):
    import re
    from sklearn.feature_extraction.text import TfidfVectorizer
    import numpy as np

    # Pisahkan menjadi kalimat
    sentences = re.split(r'(?<=[.!?]) +', text)

    # Jika dokumen terlalu pendek
    if len(sentences) <= num_sentences:
        return text

    # TF-IDF untuk kalimat
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(sentences)

    # Hitung skor tiap kalimat (jumlah bobot TF-IDF)
    scores = tfidf_matrix.sum(axis=1).A1

    # Ambil kalimat dengan skor tertinggi
    ranked_index = np.argsort(scores)[::-1]
    selected = [sentences[i] for i in ranked_index[:num_sentences]]

    # Gabungkan ringkasan
    summary = " ".join(selected)
    return summary

# ====================================================
# 3. Cache Management - Preprocessing semua dokumen
# ====================================================
def load_or_create_cache():
    """Load cache atau buat baru jika belum ada"""
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "rb") as f:
                cache = pickle.load(f)
                print("✅ Cache loaded")
                return cache
        except:
            print("⚠️ Cache corrupt, rebuilding...")
    
    # Build cache baru
    print("🔄 Building cache...")
    cache = {}
    files = [f for f in os.listdir(DOCUMENT_FOLDER) if os.path.isfile(os.path.join(DOCUMENT_FOLDER, f))]
    
    for filename in files:
        path = os.path.join(DOCUMENT_FOLDER, filename)
        raw_text = read_text(path)
        processed = preprocessing(raw_text)
        
        # Simpan text yang sudah di-stem untuk TF-IDF
        cache[filename] = {
            "raw": raw_text,
            "processed": processed,
            "stemmed_text": " ".join(processed["stemmed"])
        }
    
    # Simpan cache
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(cache, f)
    
    print(f"✅ Cache created with {len(cache)} documents")
    return cache

# Load cache saat startup
document_cache = load_or_create_cache()

# ====================================================
# Build Inverted Index
# ====================================================
def build_inverted_index(cache):
    inverted = {}
    for filename, data in cache.items():
        for term in set(data["processed"]["stemmed"]):
            if term not in inverted:
                inverted[term] = []
            inverted[term].append(filename)
    return inverted

inverted_index = build_inverted_index(document_cache)

@app.route("/inverted-index")
def view_inverted_index():
    return render_template("inverted_index.html", index=inverted_index)

# ====================================================
# Feature Selection (Top-K TF-IDF Global)
# ====================================================
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer

def compute_feature_selection(cache, k=20):
    docs = [cache[f]["stemmed_text"] for f in cache]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(docs)

    feature_array = vectorizer.get_feature_names_out()
    tfidf_scores = tfidf_matrix.sum(axis=0).A1

    ranked = sorted(zip(feature_array, tfidf_scores), 
                    key=lambda x: x[1], reverse=True)

    return ranked[:k]

@app.route("/features")
def features():
    selected = compute_feature_selection(document_cache, 20)
    return render_template("features.html", features=selected)



# ====================================================
# 4. Homepage
# ====================================================
@app.route("/")
def index():
    files = list(document_cache.keys())
    return render_template("index.html", files=files)


# ====================================================
# 5. Baca Dokumen & Preprocessing (dari cache)
# ====================================================
@app.route("/process/<filename>")
def process(filename):
    if filename not in document_cache:
        return "File tidak ditemukan", 404
    
    cached_data = document_cache[filename]
    
    return render_template("result.html",
                           filename=filename,
                           raw=cached_data["raw"],
                           result=cached_data["processed"])

# ====================================================
#  SUMMARY ROUTE
# ====================================================
@app.route("/summary/<filename>")
def summary(filename):
    if filename not in document_cache:
        return "File tidak ditemukan", 404

    raw_text = document_cache[filename]["raw"]
    summary_text = summarize_text(raw_text, num_sentences=3)

    return render_template(
        "summary.html", 
        filename=filename, 
        summary=summary_text,
        raw=raw_text
    )


# ====================================================
# 6. Hitung Kemiripan Query (OPTIMIZED + FILTERED)
# ====================================================
@app.route("/similarity", methods=["GET", "POST"])
def similarity():
    if request.method == "POST":
        query = request.form["query"]
        
        # Preprocess query
        query_processed = preprocessing(query)
        query_stemmed = " ".join(query_processed["stemmed"])
        
        # Ambil semua dokumen dari cache
        filenames = list(document_cache.keys())
        documents = [document_cache[f]["stemmed_text"] for f in filenames]
        
        # TF-IDF Vectorization
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform([query_stemmed] + documents)
        
        # Cosine Similarity
        sims = cosine_similarity(vectors[0:1], vectors[1:]).flatten()
        
        # Gabungkan dengan nama file ewe ayam
        results = list(zip(filenames, sims))
        
        # ✨ FILTER: Hanya tampilkan dokumen dengan similarity > 0
        results = [(f, s) for f, s in results if s > 0]
        
        # Sort berdasarkan similarity tertinggi
        results.sort(key=lambda x: x[1], reverse=True)
        
        return render_template("similarity.html",
                               query=query,
                               results=results,
                               total_found=len(results))

    return render_template("similarity.html",
                           query=None,
                           results=None,
                           total_found=0)


# ====================================================
# 7. Export ke CSV
# ====================================================
@app.route("/export-csv/<query>")
def export_csv(query):
    # Preprocess query
    query_processed = preprocessing(query)
    query_stemmed = " ".join(query_processed["stemmed"])
    
    # Ambil semua dokumen dari cache
    filenames = list(document_cache.keys())
    documents = [document_cache[f]["stemmed_text"] for f in filenames]
    
    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([query_stemmed] + documents)
    
    # Cosine Similarity
    sims = cosine_similarity(vectors[0:1], vectors[1:]).flatten()
    
    # Gabungkan dengan nama file
    results = list(zip(filenames, sims))
    
    # Filter dan sort
    results = [(f, s) for f, s in results if s > 0]
    results.sort(key=lambda x: x[1], reverse=True)
    
    # Buat CSV
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['No', 'Dokumen', 'Similarity Score', 'Match Percentage'])
    
    for idx, (filename, score) in enumerate(results, 1):
        writer.writerow([idx, filename, f"{score:.4f}", f"{score * 100:.2f}%"])
    
    # Return as download
    output.seek(0)
    response = make_response(output.getvalue())
    response.headers["Content-Disposition"] = f"attachment; filename=similarity_{query}.csv"
    response.headers["Content-Type"] = "text/csv"
    
    return response


# ====================================================
# 8. Statistik Dokumen
# ====================================================
@app.route("/statistics")
def statistics():
    stats = {
        "total_documents": len(document_cache),
        "total_words": 0,
        "avg_words": 0,
        "most_common_words": {},
        "documents": []
    }
    
    all_words = []
    
    # Hitung statistik per dokumen
    for filename, data in document_cache.items():
        words = data["processed"]["stemmed"]
        word_count = len(words)
        
        stats["total_words"] += word_count
        all_words.extend(words)
        
        stats["documents"].append({
            "name": filename,
            "word_count": word_count
        })
    
    # Rata-rata kata per dokumen
    if stats["total_documents"] > 0:
        stats["avg_words"] = stats["total_words"] // stats["total_documents"]
    
    # 10 kata paling sering muncul
    word_counts = Counter(all_words)
    stats["most_common_words"] = dict(word_counts.most_common(10))
    
    # Sort dokumen berdasarkan jumlah kata
    stats["documents"].sort(key=lambda x: x["word_count"], reverse=True)
    
    return render_template("statistics.html", stats=stats)


# ====================================================
# 9. Rebuild Cache (jika ada dokumen baru)
# ====================================================
@app.route("/rebuild-cache")
def rebuild_cache():
    global document_cache
    if os.path.exists(CACHE_FILE):
        os.remove(CACHE_FILE)
    document_cache = load_or_create_cache()
    return """
    <html>
    <head>
        <meta http-equiv="refresh" content="2;url=/" />
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet" />
    </head>
    <body class="bg-light">
        <div class="container mt-5">
            <div class="alert alert-success text-center">
                <h4>✅ Cache berhasil di-rebuild!</h4>
                <p>Anda akan diarahkan ke homepage dalam 2 detik...</p>
                <a href="/" class="btn btn-primary mt-3">Kembali Sekarang</a>
            </div>
        </div>
    </body>
    </html>
    """


# ====================================================
# 10. Main
# ====================================================
if __name__ == "__main__":
    app.run(debug=True)