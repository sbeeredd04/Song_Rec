import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
import warnings
from sklearn.exceptions import ConvergenceWarning

#ignore warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Load your dataset
df1 = pd.read_csv('chatbot_dataset/dialogs.txt', sep='\t', names=['question', 'answer'])

# Load your additional dataset
df2 = pd.read_csv('chatbot_dataset/chatbot dataset.txt', sep='\t', names=['question', 'answer'])

df3 = pd.read_csv('chatbot_dataset/Conversation.csv')

# Combine the datasets
df = pd.concat([df1, df2, df3])

# Combine the question and answer columns into one column
df['text'] = df['question'] + ' ' + df['answer']

# Initialize the vectorizer
vectorizer = TfidfVectorizer(tokenizer=nltk.word_tokenize, stop_words=stopwords.words('english'))

# Fit and transform the text
tfidf = vectorizer.fit_transform(df['text'])

def chatbot(user_input):
    # Vectorize the user's input
    user_input_tfidf = vectorizer.transform([user_input])

    # Compute the cosine similarity between the user's input and the dataset
    cosine_similarities = cosine_similarity(user_input_tfidf, tfidf).flatten()

    # Get the index of the most similar text
    best_index = cosine_similarities.argmax()

    # Return the most similar response
    return df['answer'].iloc[best_index]

# Get user input
user_input = input('You: ')

while True:
    # Get the chatbot response
    response = chatbot(user_input)

    # Print the response
    print('Bot: ' + response)

    # Get new input from the user
    user_input = input('You: ')

    # If the user's input is 'bye' exit the loop
    if user_input == 'bye':
        break