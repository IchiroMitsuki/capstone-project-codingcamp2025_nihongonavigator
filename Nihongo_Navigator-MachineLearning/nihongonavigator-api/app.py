import re
import os
import string
import nltk
import pickle
import numpy as np
nltk.data.path.append(r"nltk_data")
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('punkt_tab')
nltk.download('stopwords')
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from flask import Flask, request, jsonify
from flask_cors import CORS

from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences



# ---- Konfigurasi awal ----
app = Flask(__name__)
MAX_LEN = 200  # Sama seperti saat training
CORS(app)  # Ini yang bikin semua domain bisa akses API kamu

@app.route('/')
def home():
    return "Hello from Flask! API is working"

# Load model
model = load_model('best_model_lstm.h5')

# Load tokenizer
with open('tokenizer_lstm.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Load label encoder
with open('label_encoder.pkl', 'rb') as handle:
    label_encoder = pickle.load(handle)


def cleaningText(text):
    text = re.sub(r'@[A-Za-z0-9]+', '', text) # menghapus mention
    text = re.sub(r'#[A-Za-z0-9]+', '', text) # menghapus hashtag
    text = re.sub(r'RT[\s]', '', text) # menghapus RT
    text = re.sub(r"http\S+", '', text) # menghapus link
    text = re.sub(r'[0-9]+', '', text) # menghapus angka
    text = re.sub(r'[^\w\s]', '', text) # menghapus karakter selain huruf dan angka
    text = re.sub(r'[^A-Za-z\s]', '', text)


    text = text.replace('\n', ' ') # mengganti baris baru dengan spasi
    text = text.translate(str.maketrans('', '', string.punctuation)) # menghapus semua tanda baca
    text = text.strip(' ') # menghapus karakter spasi dari kiri dan kanan teks
    return text

def casefoldingText(text): # Mengubah semua karakter dalam teks menjadi huruf kecil
    text = text.lower()
    return text

def tokenizingText(text): # Memecah atau membagi string, teks menjadi daftar token
    text = word_tokenize(text)
    return text

def filteringText(text): # Menghapus stopwords dalam teks
    listStopwords = set(stopwords.words('indonesian'))
    listStopwords1 = set(stopwords.words('english'))
    listStopwords.update(listStopwords1)
    listStopwords.update(['iya','yaa','gak','nya','na','sih','ku',"di","ga","ya","gaa","loh","kah","woi","woii","woy", "wkwk", "uuu"])
    important_words = {'baik', 'buruk', 'mantap', 'bagus', 'jelek', 'mudah', 'cepat', 'nyaman'} #Bisa ditambah
    # Hapus kata penting dari daftar stopwords
    listStopwords = listStopwords - important_words

    filtered = []
    for txt in text:
        if txt not in listStopwords:
            filtered.append(txt)
    text = filtered
    return text

# Membuat objek stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

def stemmingText(text): # Mengurangi kata ke bentuk dasarnya yang menghilangkan imbuhan awalan dan akhiran atau ke akar kata
        # Memecah teks menjadi daftar kata
    words = text.split()

    # Menerapkan stemming pada setiap kata dalam daftar
    stemmed_words = [stemmer.stem(word) for word in words]

    # Menggabungkan kata-kata yang telah distem
    stemmed_text = ' '.join(stemmed_words)

    return stemmed_text

def toSentence(list_words): # Mengubah daftar kata menjadi kalimat
    sentence = ' '.join(word for word in list_words)
    return sentence

def load_slang_dictionary():
    slangwords = {}

    def read_file(path, delimiter):
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if delimiter not in line or not line:
                    continue  # lewati baris yang kosong atau tidak valid
                parts = line.split(delimiter, 1)
                if len(parts) != 2:
                    continue  # lewati jika tetap gagal dibagi jadi 2
                key, value = parts
                if key and value:
                    slangwords[key.lower()] = value.lower()

    # Gabungkan dari semua file
    read_file(r'slangwords/formalizationDict.txt', '\t')
    read_file(r'slangwords/kbba.txt', '\t')
    read_file(r'slangwords/slangword.txt', ':')

    return slangwords

# Panggil fungsi
slangwords = load_slang_dictionary()

def fix_slangwords(text, slang_dict):
    words = text.split()  # Memecah teks jadi list kata
    fixed_words = [slang_dict.get(word.lower(), word) for word in words]  # Ganti jika ada di kamus
    return ' '.join(fixed_words)  # Gabung kembali jadi kalimat

def split_heuristic(text):
    # Ganti koma dan kata penghubung umum dengan titik
    text = re.sub(r'\s*(,|namun|tapi|tetapi|meskipun|walaupun|serta)\s*', '.', text)
    
    # Pisah berdasarkan titik atau baris baru
    sentences = re.split(r'\.|\n', text)

    # Bersihkan hasil
    sentences = [s.strip() for s in sentences if len(s.strip()) > 3]

    return sentences

def normalize_tokens(tokens, norm_dict):
    return [norm_dict.get(word, word) for word in tokens]

custom_normalization = {
    "mantaaap": "mantap",
    "bangett": "banget",
    "kerenn": "keren",
    "prem": "premium",
    "bagusss": "bagus",
    "membantuuu": "membantu",
    "bangettt": "banget",
    "bingunggg": "bingung",
    "sangattt": "sangat",
    "waaaah": "wah",
    "polll": "pol",           
    "tulisannnya": "tulisannya",
    "maziiini": "mazii ini",
    "salahhadehhhh": "salah hadeh",
    "sangaatttt": "sangat",
    "lagiii": "lagi",
    "bagusssss": "bagus",
    "bagusssbisa": "bagus bisa",
    "mantaaapberjamjam": "mantap berjam-jam",
    "waaaaah": "wah",
    "yaaa": "ya",
    "mmmmungkin": "mungkin",
    "bagussss": "bagus",
    "mantappppuuu": "mantap",
    "majuuu": "maju",
    "bagussssss": "bagus",
    "akuuu": "aku",
    "hehheee": "hehe",
    "kerennnn": "keren",
    "makasiii": "makasih",
    "gimanasi": "bagaimana sih",
    "arigato": "terima kasih",
    "arigatou": "terima kasih",
    "radical": "radikal",
    "knji": "kanji",
    "knj": "kanji",
}


# ---- API Route ----
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get('text', '')

    if not text:
        return jsonify({'error': 'Teks tidak boleh kosong'}), 400

    # Preprocessing mirip saat training
    text_clean = stemmingText(toSentence(filteringText(normalize_tokens(tokenizingText(fix_slangwords(toSentence(split_heuristic(casefoldingText(cleaningText(text)))), slangwords)), custom_normalization))))
    seq = tokenizer.texts_to_sequences([text_clean])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post', truncating='post')

    # Prediksi
    pred = model.predict(padded)
    label_idx = np.argmax(pred)
    label = label_encoder.inverse_transform([label_idx])[0]

    return jsonify({
        'text': text,
        'prediction': label,
        'confidence': float(np.max(pred))
    })

# ---- Jalankan ----
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Ambil dari env, default 5000
    app.run(host='0.0.0.0', port=port, debug=False)
