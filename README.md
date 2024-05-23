# Aclaraciones

1) el script imdb-reviews-model sirve para crear un modelo y entrenarlo, genera el archivo trained-imdb-reviews para luego poder usarlo sin tener que re-entrenar el modelo cada vez que uno quiera hacer una predicción. (Necesita tener en la misma unicación el archivo: imdb_reviews10kV2.csv para poder iniciar el entrenamiento a partir de él)

2) Para realizar nuevas predicciones es decir análisis de sentimiento de nuevas reseñas ingresadas, debemos correr el script predict_nueva-review (el cual necesita en la misma ubicación, los archivos: trained-imdb-reviews.h5 e imdb_reviews10kV2.csv Tengo entendido que podria no necesitar el csv pero según leí es recomendable que se vuelva a leer las palabras que utilizó en el entrenamiento para hacer mejor las predicciones.)

 Al correr el script para analizar una nueva review, solicitará por consola ingresar una descripción de tipo reseña que se te ocurra, debe ser escrita en idioma inglés porque el modelo se entrenó con reseñas en inglés. Por ejemplo una reseña podría ser: "I loved this movie, the plot and acting were excellent" la cual debería responder como una reseña "Positiva".
El programa seguirá pidiendo ingresar reseñas para predecir hasta que se ingrese la palabra "salir".
