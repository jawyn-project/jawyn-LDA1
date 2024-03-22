from time import time
import matplotlib.pyplot as plt
import json
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas as pd


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



def read_json(n_samples):
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

    return data_samples



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
            

def select_top_topics_for_all_words(lda, tf_vectorizer, num_topics=2):

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



def get_relevant_documents_for_topics(wikis, top_topics_for_all_words, topic_assignments):
    
    # Dicionário para armazenar os documentos mais relevantes para cada tópico de cada palavra
    relevant_documents = {}
    
    for word, topics in top_topics_for_all_words.items():
        relevant_documents[word] = {}
        for topic_idx, _ in topics:
            # Ajustar o índice do tópico para corresponder à matriz 'topic_assignments'
            adjusted_topic_idx = topic_idx - 1
            # Encontrar os documentos mais relevantes para o tópico
            top_document_indices = topic_assignments[:, adjusted_topic_idx].argsort()[-3:][::-1]
            # Mapear os índices dos documentos para os documentos reais
            relevant_docs = [wikis.iloc[i]['content'] for i in top_document_indices]
            relevant_documents[word][topic_idx] = top_document_indices
    
    return relevant_documents


def display_relevant_word(documents, example,wikis):
    document_set = set()

    for word, topics in documents.items():
        if word == example:
            print(f"Documentos relevantes para a palavra '{word}':")
            for topic_idx, documents in topics.items():
                    print(f" Tópico {topic_idx}:")
                    for doc_idx, content in enumerate(documents, start=1):
                        print(f"    Documento {doc_idx} : {content}")
                        print(wikis["headline"][content])
                        document = {
                        'link': wikis['link'][content],
                        'headline': wikis['headline'][content],
                        'content': wikis['content'][content][:50]
                        }
                        document_set.add(frozenset(document.items()))
                        
                    print() # Adiciona uma linha em branco entre os tópicos'''

    return document_set



