import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
import re

# Carga de dataset (Reseñas de IMDB catalogadas como positivas o negativas)
df = pd.read_csv('imdb_reviews10kV2.csv')


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
balanced_df = pd.concat([positive_reviews.sample(min_count, random_state=42), negative_reviews.sample(min_count, random_state=42)])

# Separa train y test sets
X_train, X_test, y_train, y_test = train_test_split(balanced_df['review'], balanced_df['sentiment'], test_size=0.2, random_state=42)

# Mapeo de etiquetas
label_map = {'positive': 1, 'negative': 0}
y_train = y_train.map(label_map)
y_test = y_test.map(label_map)

# Tokenización y padding
tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

max_length = 300
X_train_padded = pad_sequences(X_train_seq, maxlen=max_length, padding='post', truncating='post')
X_test_padded = pad_sequences(X_test_seq, maxlen=max_length, padding='post', truncating='post')

# Modelo de red neuronal
model = Sequential()
model.add(Embedding(input_dim=5000, output_dim=64, input_length=max_length))
model.add(Bidirectional(LSTM(64, return_sequences=True)))
model.add(Dropout(0.5))
model.add(Bidirectional(LSTM(64)))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Entrenamiento del modelo
history = model.fit(X_train_padded, y_train, epochs=12, batch_size=64, validation_split=0.2)

# Guardar el modelo
model.save('trained-imdb-reviews.h5')

# Evaluación del modelo
y_pred = (model.predict(X_test_padded) > 0.5).astype("int32")
accuracy = accuracy_score(y_test, y_pred)
print(f"\nPrecisión del modelo: {accuracy:.4f}\n")
print("Clasificación Reporte:\n", classification_report(y_test, y_pred))
#print("Matriz de Confusión:\n", confusion_matrix(y_test, y_pred))
