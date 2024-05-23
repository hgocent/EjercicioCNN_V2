import pandas as pd
import numpy as np
import re
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

# Carga de dataset (Reseñas de IMDB catalogadas como positivas o negativas)
df = pd.read_csv('imdb_reviews10kV2.csv')

# Cargar el modelo entrenado
model = load_model('trained-imdb-reviews.h5')
    
# Eliminar registros inválidos
df.dropna(inplace=True)

# Preprocesamiento de texto
def clean_text(text):
    text = re.sub(r'<[^>]+>', '', text)  # Eliminar etiquetas HTML
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Eliminar caracteres no alfabéticos
    text = text.lower()  # Convertir a minúsculas
    return text

df['review'] = df['review'].apply(clean_text)

# Balanceo del dataset
positive_reviews = df[df['sentiment'] == 'positive']
negative_reviews = df[df['sentiment'] == 'negative']

# Asegurando que ambos conjuntos tengan el mismo tamaño
min_count = min(len(positive_reviews), len(negative_reviews))
balanced_df = pd.concat([positive_reviews.sample(min_count, random_state=42),negative_reviews.sample(min_count, random_state=42)])

# Separa train y test sets
X_train, X_test, y_train, y_test = train_test_split(balanced_df['review'], balanced_df['sentiment'], test_size=0.2, random_state=42)

# Mapeo de etiquetas
label_map = {'positive': 1, 'negative': 0}
y_train = y_train.map(label_map)
y_test = y_test.map(label_map)

# Ajustar el tokenizador con los textos de entrenamiento originales
tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

max_length = 300 #(el modelo fué entrenado con este valor no se puede cambiar)

# Limpiar el texto de entrada
def clean_text(text):
    text = text.lower()  # Convertir a minúsculas
    return text

# Definir una función para predecir el sentimiento de una reseña
def predict_sentiment(review):
    review_cleaned = clean_text(review)
    review_seq = tokenizer.texts_to_sequences([review_cleaned])
    review_padded = pad_sequences(review_seq, maxlen=max_length, padding='post', truncating='post')
    prediction = model.predict(review_padded)
    #print(f"Prediction: {prediction}")
    if prediction > 0.5:
        return "Positiva"
    elif prediction > 0.25:
        return "Indefinida"
    else:
        return "Negativa"
    
# Loop para ingresar reseñas por consola y predecir su sentimiento
while True:
    print("\nIngrese una reseña (en idioma inglés) para predecir su sentimiento (o 'salir' para terminar):")
    new_review = input()
    if new_review.lower() == 'salir':
        break
      
    sentiment = predict_sentiment(new_review)
    print(f"La reseña es: {sentiment}\n")

print("Programa terminado.")
