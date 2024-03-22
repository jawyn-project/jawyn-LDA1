from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
from LDA import LDA,read_json,extract_features,select_top_topics_for_all_words,get_relevant_documents_for_topics,display_relevant_word

app = Flask(__name__)
#CORS(app, origins="http://localhost:5173")

n_samples = 200# amostras
n_features = 20 # caracteristicas
n_components = 6 # componentes para os modelos de topicos
n_top_words = 6 # palavras principais
init = "nndsvda"

data_samples = read_json(n_samples)

wikis = pd.read_json('data.json',lines=True)

tf, tf_vectorizer = extract_features(data_samples, n_features)

topics , lda = LDA(tf,tf_vectorizer,n_components)

top_topics_for_all_words = select_top_topics_for_all_words(lda, tf_vectorizer)

topic_assignments = lda.transform(tf)

documents = get_relevant_documents_for_topics(wikis,top_topics_for_all_words,topic_assignments)


@app.route('/')
def index():
    return 'Hello, World!'

@app.route('/search')
def search():
    parametro_q = request.args.get('q')
    document_set = display_relevant_word(documents, parametro_q,wikis)
    lista_de_dicionarios = [dict(fs) for fs in document_set]
    # Responder com os dados convertidos para JSON
    return jsonify(lista_de_dicionarios)
   



if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')

