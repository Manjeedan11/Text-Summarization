from flask import Flask, request, render_template
from transformers import pipeline
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Initialize the summarizer pipeline with an explicit model
summarizer = pipeline('summarization', model='sshleifer/distilbart-cnn-12-6')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    article = request.form['article']
    summary = summarizer(article, max_length=500, min_length=10, do_sample=False)
    summary_text = summary[0]['summary_text']
    return render_template('index.html', original_text=article, summary_text=summary_text)

if __name__ == '__main__':
    app.run(debug=True)
