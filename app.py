import os
from flask import Flask, request, render_template_string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

app = Flask(__name__)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head><title>Resume Scanner</title></head>
<body>
    <h2>Resume & Job Description Scanner</h2>
    <form method="POST" enctype="multipart/form-data">
        <label>Resume Text File:</label><br><input type="file" name="resume"><br><br>
        <label>Job Description Text File:</label><br><input type="file" name="job"><br><br>
        <input type="submit" value="Scan">
    </form>

    {% if score %}
        <h3>Match Score: {{ score }}%</h3>
        <h4>Missing Keywords from Resume:</h4>
        <ul>
        {% for word in missing %}
            <li>{{ word }}</li>
        {% endfor %}
        </ul>
    {% endif %}
</body>
</html>
"""

def preprocess(text):
    words = text.lower().split()
    return ' '.join([word for word in words if word not in stop_words])

@app.route('/', methods=['GET', 'POST'])
def scan():
    if request.method == 'POST':
        resume_file = request.files['resume']
        job_file = request.files['job']

        resume = resume_file.read().decode('utf-8')
        job_desc = job_file.read().decode('utf-8')

        resume_clean = preprocess(resume)
        job_clean = preprocess(job_desc)

        tfidf = TfidfVectorizer()
        vectors = tfidf.fit_transform([resume_clean, job_clean])
        score = cosine_similarity(vectors[0:1], vectors[1:2])[0][0] * 100

        resume_words = set(resume_clean.split())
        job_words = set(job_clean.split())
        missing_keywords = list(job_words - resume_words)

        return render_template_string(HTML_TEMPLATE, score=round(score, 2), missing=missing_keywords)

    return render_template_string(HTML_TEMPLATE)

if __name__ == '__main__':
    app.run(debug=True)
