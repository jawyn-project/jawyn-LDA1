
import requests
from bs4 import BeautifulSoup
import re
import time
import json
import pandas as pd
from urllib.parse import quote
import re

def take(link, result, key):
    url = link
    try:
        r = requests.get(url)
    except requests.exceptions.RequestException as e:
        print(f"Couldn't access {url}. Error: {e}")
        return [], [], []

    html_2 = r.text.encode("utf8")

    soup = BeautifulSoup(html_2, 'html.parser')
    paragraphs = soup.find_all('p')
    all_text = ' '.join(p.get_text() for p in paragraphs)

    headline = soup.title.string

    result.append({"key": key, "link": link, "headline": headline, "content": all_text})  # Modificação aqui

    wiki_links = []
    other_links = []

    for link in soup.find_all('a', href=True):
        full_link = link['href']
        if full_link.startswith('/'):
            full_link = 'https://en.wikipedia.org' + full_link
            wiki_links.append(full_link)
        elif full_link.startswith('http'):
            other_links.append(full_link)

    title_links = []
    other_wiki_links = []

    for link in wiki_links:
        if re.match(r'https://en\.wikipedia\.org/wiki/[^/:(]*$', link) and not link.endswith('.png'):
            if link not in title_links:
                title_links.append(link)
        else:
            other_wiki_links.append(link)

    return title_links, other_wiki_links, other_links


def traverse(title_links, result, link_acess, key):
    print()

    for i in range(5):
        if title_links:
            link = title_links.pop(0)

            if link not in link_acess:
                print("calling link:" + link)
                time.sleep(1)
                new_title_links, other_wiki_links, other_links = take(link, result, key)
                title_links.extend(new_title_links)
                link_acess.append(link)

# First Keys Example: The initial topics for search and crawler process

'''keys = [
    "Dog", "Cat", "Cow", "Tiger", "Elephant", "Horse", "Lion", "Giraffe", "Monkey", "Penguin",
    "Bear", "Wolf", "Dolphin", "Snake", "Democracy", "Monarchy", "Dictatorship", "Elections",
    "Political parties", "Republic", "Anarchy", "Oligarchy", "Fascism", "Communism", "Capitalism",
    "Socialism", "Constitution", "Middle Ages", "Modern Age", "Industrial Revolution", "World Wars",
    "Ancient Egypt", "Renaissance", "Cold War", "Colonialism", "American Revolution", "French Revolution",
    "Civil Rights Movement", "Space Race", "Brazil", "United States", "China", "India", "Russia", "France",
    "Germany", "Japan", "Canada", "Australia", "United Kingdom", "Italy", "Mexico", "South Africa", "Physics",
    "Chemistry", "Biology", "Astronomy", "Mathematics", "Genetics", "Geology", "Ecology", "Neuroscience",
    "Robotics", "Particle Physics", "Quantum Mechanics", "Cosmology", "Microbiology", "Artificial Intelligence",
    "Blockchain", "Internet of Things", "Virtual Reality", "Augmented Reality", "Cybersecurity", "Machine Learning",
    "Cryptocurrency", "Biotechnology", "Space Exploration", "Nanotechnology", "Renewable Energy", "Football",
    "Basketball", "Tennis", "Swimming", "Athletics", "Golf", "Cricket", "Baseball", "Soccer", "Rugby", "Boxing",
    "Martial Arts", "Cycling", "Surfing", "Pizza", "Sushi", "Burger", "Pasta", "Chocolate", "Salad", "Steak",
    "Ice Cream", "Curry", "Tacos", "Sushi", "Dim Sum", "Paella", "Baguette", "Painting", "Sculpture", "Photography",
    "Architecture", "Drawing", "Music", "Literature", "Film", "Dance", "Theater", "Poetry", "Graffiti", "Calligraphy",
    "Pottery"
]'''

# Second Keys Example: The initial topics for search and crawler process

keys = [
    "Dog", "Cat", "Cow", "Tiger", "Elephant", "Horse",
    "Cryptocurrency", "Biotechnology", "Space Exploration", "Nanotechnology", "Renewable Energy", "Football"
]


result = []
link_acess = []

for key in keys:
    link = 'https://en.wikipedia.org/wiki/' + key
    title_links, other_wiki_links, other_links = take(link, result, key)
    traverse(title_links, result, link_acess, key)

df = pd.DataFrame(result)

file_name = "data.json"

try:
    df.to_json(file_name, orient='records', lines=True, force_ascii=False)
    print(f"JSON saved successfully at: {file_name}")
except Exception as e:
    print(f"Error saving JSON: {e}")


# read the .json file

# try:
#     with open(file_name, "r", encoding='utf-8') as file:
#         json_content = [json.loads(line) for line in file]
#     print(json_content)
# except Exception as e:
#     print(f"Error reading JSON: {e}")

"""# LDA implementation with SKlearn"""

from time import time
import matplotlib.pyplot as plt
import json
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

n_samples = 2000 # amostras
n_features = 1000 # caracteristicas
n_components = 6 # componentes para os modelos de topicos
n_top_words = 6 # palavras principais
init = "nndsvda"

# função usada para visualiar as palavras principais pra cada tópico
def plot_top_words(model, feature_names, n_top_words, title):
    fig, axes = plt.subplots(2, 5, figsize=(30, 15), sharex=True)
    axes = axes.flatten()
    for topic_idx, topic in enumerate(model.components_):
        top_features_ind = topic.argsort()[-n_top_words:]
        top_features = feature_names[top_features_ind]
        weights = topic[top_features_ind]

        ax = axes[topic_idx]
        ax.barh(top_features, weights, height=0.7)
        ax.set_title(f"Topic {topic_idx +1}", fontdict={"fontsize": 30})
        ax.tick_params(axis="both", which="major", labelsize=20)
        for i in "top right left".split():
            ax.spines[i].set_visible(False)
        fig.suptitle(title, fontsize=40)

    plt.subplots_adjust(top=0.90, bottom=0.05, wspace=0.90, hspace=0.3)
    plt.show()


data_samples = []

with open('data.json', 'r', encoding='utf-8') as f:
    for line in f:
        # carrega cada linha como um objeto JSON separado
        data_sample = json.loads(line)

        # Supondo que 'data_sample' é um dicionário e você quer extrair o texto
        # Ajuste isso conforme necessário para corresponder à estrutura do seu arquivo

        content = data_sample['content']
        data_samples.append(content)

data_samples = data_samples[:n_samples]


# utiliza-se o método TF para extrair características do texto para também serem
# utilizadas mais adiante em um modelo LDA

def extract_features(data_samples, n_features):
    print("Extracting tf features for LDA...")

    tf_vectorizer = CountVectorizer(
        max_df=0.95, min_df=2, max_features=n_features, stop_words="english"
    )
    t0 = time()
    tf = tf_vectorizer.fit_transform(data_samples)
    # o resultado é uma matriz onde cada linha é um documento ----> ISSO
    # e cada coluna uma palavra

    return tf, tf_vectorizer

tf, tf_vectorizer = extract_features(data_samples, n_features)

def LDA(tf, tf_vectorizer,n_components):

    lda = LatentDirichletAllocation(
        n_components=n_components,
        max_iter=5,
        learning_method="online",
        learning_offset=50.0,
        random_state=0,
    )

    t0 = time()
    lda.fit(tf)

    tf_feature_names = tf_vectorizer.get_feature_names_out()
    #plot_top_words(lda, tf_feature_names, n_top_words, "Topics in LDA model")

    return lda.components_, lda

topics , lda = LDA(tf,tf_vectorizer,n_components)


def word_probabilities(lda, tf_vectorizer):
    # obter o nome das palavras
    feature_names = tf_vectorizer.get_feature_names_out()
    
    topics_list = []

    # para cada tópico, armazena as palavras e suas probabilidades
    for topic_idx, topic in enumerate(lda.components_):

        topic_dict = {}

        for word_idx, word_prob in enumerate(topic):
            topic_dict[feature_names[word_idx]] = word_prob
        
        # ordena o dicionário pelas probabilidades das palavras
        sorted_topic_dict = dict(sorted(topic_dict.items(), key=lambda item: item[1], reverse=True))
        
        # adiciona o dicionário ordenado à lista de tópicos
        topics_list.append(sorted_topic_dict)
    
    return topics_list
            

def select_top_topics_for_all_words(lda, tf_vectorizer, num_topics=3):

    topics_list = word_probabilities(lda, tf_vectorizer)
    
    top_topics_for_all_words = {}
    
    for topic_idx, topic_dict in enumerate(topics_list):
        for word, prob in topic_dict.items():
            if word not in top_topics_for_all_words:
                top_topics_for_all_words[word] = []
            # Adicionar o índice do tópico e a probabilidade à lista
            top_topics_for_all_words[word].append((topic_idx, prob))
    
    # ordena os tópicos por probabilidade e selecionar os 3 com maiores probabilidades para cada palavra
    for word, topics in top_topics_for_all_words.items():

        adjusted_topics = [(t+1, p) for t, p in topics]

        # ordena os tópicos por probabilidade em ordem decrescente
        sorted_topics = sorted(adjusted_topics, key=lambda x: x[1], reverse=True)
        
        # seleciona os 3 tópicos com maiores probabilidades
        top_topics_for_all_words[word] = sorted_topics[:num_topics]
    
    return top_topics_for_all_words

top_topics_for_all_words = select_top_topics_for_all_words(lda, tf_vectorizer)

'''
for word, topics in top_topics_for_all_words.items():
    print(f"Palavra: {word},  Tópicos: {topics}")'''

def get_documents_for_topic(topic_idx, topic_assignments):
    # Supondo que topic_assignments seja uma matriz onde cada linha é um documento e cada coluna é um tópico
    # e cada célula contém a probabilidade de que o documento pertença ao tópico
    
    # Encontrar os índices dos documentos com as maiores probabilidades para o tópico especificado
    document_probs = topic_assignments[:, topic_idx]
    top_document_indices = document_probs.argsort()[-3:][::-1] # Seleciona os 3 documentos com maiores probabilidades
    
    return top_document_indices

def get_documents_for_top_topics(lda, tf, tf_vectorizer, top_topics_for_all_words):
    # Mapear palavras para seus documentos
    print("ENTROU AQUI")
    word_to_docs = {}
    for doc_idx, word_freqs in enumerate(tf):
        for word_idx, freq in enumerate(word_freqs):
            if freq > 0:
                word = tf_vectorizer.get_feature_names_out()[word_idx]
                if word not in word_to_docs:
                    word_to_docs[word] = []
                word_to_docs[word].append(doc_idx)
    
    # para cada palavra, verificar as probabilidades nos tópicos específicos e retornar os documentos correspondentes
    docs_for_top_topics = {}
    for word, topics in top_topics_for_all_words.items():
        docs_for_word = []
        for topic_idx, _ in topics:
            # Aqui, você precisa verificar as probabilidades da palavra no tópico e, em seguida, adicionar os documentos correspondentes
            # Isso pode ser feito usando a matriz de tópicos do modelo LDA e a matriz de termos frequentes (tf)
            # Como isso é complexo e depende da estrutura exata dos seus dados, vou fornecer um exemplo genérico
            # Supondo que você tenha uma maneira de mapear índices de tópicos para documentos
            docs_for_topic = get_documents_for_topic(topic_idx, tf)
            docs_for_word.extend(docs_for_topic)
        docs_for_top_topics[word] = docs_for_word
    
    return docs_for_top_topics

docs_for_top_topics = get_documents_for_top_topics(lda, tf, tf_vectorizer, top_topics_for_all_words)

# imprime os documentos para cada palavra
for word, docs in docs_for_top_topics.items():
    print(f"Palavra: {word}, Documentos: {docs}")





'''for topic_idx, topic_dict in enumerate(topics_list):
    print(f"\nTopic {topic_idx + 1}:")
    for word, prob in topic_dict.items():
        print(f"Palavra : {word} / Probabilidade: {prob}")
'''
'''
def find_top_documents_for_words(lda, tf, tf_vectorizer, top_topics=[2, 5, 8]):
     # Extrair tópicos dos documentos
    topic_assignments = lda.transform(tf)
    
    # Verificar se os índices em top_topics estão dentro dos limites
    n_topics = lda.n_components
    top_topics = [t for t in top_topics if t < n_topics]
    
    # Mapear palavras para seus documentos
    word_to_docs = {}
    for doc_idx, word_freqs in enumerate(tf):
        for word_idx, freq in enumerate(word_freqs):
            if freq.getnnz() > 0:
                word = tf_vectorizer.get_feature_names_out()[word_idx]
                if word not in word_to_docs:
                    word_to_docs[word] = []
                word_to_docs[word].append(doc_idx)
    
    # Para cada palavra, verificar as probabilidades nos tópicos específicos e retornar os documentos correspondentes
    top_docs_for_words = {}
    for word, docs in word_to_docs.items():
        top_docs = []
        for doc_idx in docs:
            top_topic_probs = [topic_assignments[doc_idx, t] for t in top_topics]
            if max(top_topic_probs) > 0: # Verifica se a palavra tem maior probabilidade em pelo menos um dos tópicos específicos
                top_docs.append(doc_idx)
        if top_docs:
            top_docs_for_words[word] = top_docs
    
    return top_docs_for_words

# Chamar a função e armazenar o resultado em uma variável
top_docs_for_words = find_top_documents_for_words(lda, tf, tf_vectorizer)

# Imprimir os documentos para cada palavra
for word, docs in top_docs_for_words.items():
    print(f"Palavra: {word}, Documentos: {docs}")
'''