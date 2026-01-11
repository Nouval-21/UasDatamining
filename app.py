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
import numpy as np

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
# 3. IMPROVED SUMMARY (Extractive Summarization)
# ====================================================
def summarize_text(text, num_sentences=3):
    """
    Extractive summarization menggunakan TF-IDF untuk ranking kalimat
    
    Args:
        text: Teks dokumen yang akan diringkas
        num_sentences: Jumlah kalimat dalam ringkasan (default: 3)
    
    Returns:
        Dictionary dengan informasi ringkasan
    """
    # Pisahkan menjadi kalimat
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]  # Filter kalimat pendek
    
    # Jika dokumen terlalu pendek
    if len(sentences) <= num_sentences:
        return {
            "summary": text,
            "original_sentences": len(sentences),
            "summary_sentences": len(sentences),
            "compression_ratio": 100.0,
            "ranked_sentences": []
        }
    
    # Preprocessing untuk setiap kalimat
    processed_sentences = []
    for sent in sentences:
        tokens = preprocessing(sent)
        processed_sentences.append(" ".join(tokens["stemmed"]))
    
    # TF-IDF untuk kalimat
    vectorizer = TfidfVectorizer()
    try:
        tfidf_matrix = vectorizer.fit_transform(processed_sentences)
        
        # Hitung skor tiap kalimat (rata-rata bobot TF-IDF)
        scores = np.asarray(tfidf_matrix.mean(axis=1)).flatten()
        
        # Ranking kalimat berdasarkan skor
        ranked_indices = np.argsort(scores)[::-1]
        
        # Ambil top-N kalimat dengan skor tertinggi
        selected_indices = sorted(ranked_indices[:num_sentences])
        
        # Buat ringkasan (urutan sesuai dokumen asli)
        summary_sentences = [sentences[i] for i in selected_indices]
        summary = " ".join(summary_sentences)
        
        # Informasi tambahan untuk ditampilkan
        ranked_info = [
            {
                "rank": idx + 1,
                "sentence": sentences[i],
                "score": float(scores[i]),
                "selected": i in selected_indices
            }
            for idx, i in enumerate(ranked_indices[:10])  # Top 10 untuk referensi
        ]
        
        compression_ratio = (len(summary_sentences) / len(sentences)) * 100
        
        return {
            "summary": summary,
            "original_sentences": len(sentences),
            "summary_sentences": len(summary_sentences),
            "compression_ratio": compression_ratio,
            "ranked_sentences": ranked_info
        }
        
    except Exception as e:
        print(f"Error dalam summarization: {e}")
        # Fallback: ambil N kalimat pertama
        fallback_summary = " ".join(sentences[:num_sentences])
        return {
            "summary": fallback_summary,
            "original_sentences": len(sentences),
            "summary_sentences": num_sentences,
            "compression_ratio": (num_sentences / len(sentences)) * 100,
            "ranked_sentences": [],
            "error": str(e)
        }

# ====================================================
# 4. Cache Management - Preprocessing semua dokumen
# ====================================================
def load_or_create_cache():
    """Load cache atau buat baru jika belum ada"""
    
    # Pastikan folder documents ada
    if not os.path.exists(DOCUMENT_FOLDER):
        os.makedirs(DOCUMENT_FOLDER)
        print(f"📁 Folder '{DOCUMENT_FOLDER}' dibuat")
    
    # Ambil daftar file yang ada di folder documents
    files = []
    for f in os.listdir(DOCUMENT_FOLDER):
        file_path = os.path.join(DOCUMENT_FOLDER, f)
        # Hanya ambil file (bukan folder) dengan ekstensi yang valid
        if os.path.isfile(file_path) and f.split('.')[-1].lower() in ['pdf', 'docx', 'txt']:
            files.append(f)
    
    print(f"📄 Ditemukan {len(files)} dokumen di folder '{DOCUMENT_FOLDER}'")
    
    # Cek apakah cache sudah ada
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "rb") as f:
                cache = pickle.load(f)
            
            # Validasi: apakah semua file di cache masih ada?
            cache_files = set(cache.keys())
            current_files = set(files)
            
            # Jika file di folder sama dengan di cache, gunakan cache
            if cache_files == current_files:
                print("✅ Cache loaded (semua dokumen sudah ter-cache)")
                return cache
            else:
                print("⚠️ Ada perubahan dokumen, rebuilding cache...")
        except Exception as e:
            print(f"⚠️ Cache corrupt ({e}), rebuilding...")
    
    # Build cache baru
    print("🔄 Building cache dari dokumen yang ada...")
    cache = {}
    
    for filename in files:
        try:
            path = os.path.join(DOCUMENT_FOLDER, filename)
            print(f"   Processing: {filename}")
            
            raw_text = read_text(path)
            
            if not raw_text.strip():
                print(f"   ⚠️ Warning: {filename} kosong, dilewati")
                continue
            
            processed = preprocessing(raw_text)
            
            # Generate summary untuk setiap dokumen
            summary_info = summarize_text(raw_text, num_sentences=3)
            
            # Simpan text yang sudah di-stem untuk TF-IDF
            cache[filename] = {
                "raw": raw_text,
                "processed": processed,
                "stemmed_text": " ".join(processed["stemmed"]),
                "summary": summary_info  # Simpan summary di cache
            }
            
        except Exception as e:
            print(f"   ❌ Error processing {filename}: {e}")
            continue
    
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
# 5. Homepage
# ====================================================
@app.route("/")
def index():
    files = list(document_cache.keys())
    return render_template("index.html", files=files)


# ====================================================
# 6. Baca Dokumen & Preprocessing (dari cache)
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
# 7. IMPROVED SUMMARY ROUTE
# ====================================================
@app.route("/summary/<filename>")
def summary(filename):
    """
    Menampilkan ringkasan dokumen dengan informasi detail
    """
    if filename not in document_cache:
        return "File tidak ditemukan", 404

    cached_data = document_cache[filename]
    raw_text = cached_data["raw"]
    
    # Cek apakah summary sudah ada di cache
    if "summary" in cached_data:
        summary_info = cached_data["summary"]
    else:
        # Generate summary jika belum ada
        summary_info = summarize_text(raw_text, num_sentences=3)
        document_cache[filename]["summary"] = summary_info

    return render_template(
        "summary.html", 
        filename=filename,
        summary_info=summary_info,
        raw=raw_text
    )


# ====================================================
# 8. BULK SUMMARY - Ringkas semua dokumen
# ====================================================
@app.route("/all-summaries")
def all_summaries():
    """
    Menampilkan ringkasan dari semua dokumen
    """
    summaries = []
    
    for filename in document_cache.keys():
        cached_data = document_cache[filename]
        
        # Ambil summary dari cache atau generate baru
        if "summary" in cached_data:
            summary_info = cached_data["summary"]
        else:
            summary_info = summarize_text(cached_data["raw"], num_sentences=3)
            document_cache[filename]["summary"] = summary_info
        
        summaries.append({
            "filename": filename,
            "summary": summary_info["summary"],
            "stats": {
                "original": summary_info["original_sentences"],
                "compressed": summary_info["summary_sentences"],
                "ratio": f"{summary_info['compression_ratio']:.1f}%"
            }
        })
    
    return render_template("all_summaries.html", summaries=summaries)


# ====================================================
# 9. Hitung Kemiripan Query (DENGAN RINGKASAN DOKUMEN)
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
        
        # Gabungkan dengan nama file (format tuple untuk template)
        results = list(zip(filenames, sims))
        
        # ✨ FILTER: Hanya tampilkan dokumen dengan similarity > 0
        results = [(f, s) for f, s in results if s > 0]
        
        # Sort berdasarkan similarity tertinggi
        results.sort(key=lambda x: x[1], reverse=True)
        
        # ✨ TAMBAHAN: Buat dictionary untuk summary setiap dokumen
        summaries = {}
        for filename in filenames:
            doc_data = document_cache[filename]
            if "summary" in doc_data:
                summaries[filename] = doc_data["summary"]["summary"]
            else:
                # Generate summary on-the-fly jika belum ada
                summaries[filename] = doc_data["raw"][:200] + "..."
        
        return render_template("similarity.html",
                               query=query,
                               results=results,
                               summaries=summaries,
                               total_found=len(results))

    return render_template("similarity.html",
                           query=None,
                           results=None,
                           summaries={},
                           total_found=0)


# ====================================================
# 10. Export ke CSV
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
# 11. Statistik Dokumen
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
# 12. Download Dokumen Asli
# ====================================================
@app.route("/download/<filename>")
def download_document(filename):
    """
    Mengunduh dokumen asli
    """
    if filename not in document_cache:
        return "File tidak ditemukan", 404
    
    file_path = os.path.join(DOCUMENT_FOLDER, filename)
    
    if not os.path.exists(file_path):
        return "File tidak ditemukan di sistem", 404
    
    return send_file(
        file_path,
        as_attachment=True,
        download_name=filename
    )


# ====================================================
# 13. Main
# ====================================================
if __name__ == "__main__":
    app.run(debug=True)