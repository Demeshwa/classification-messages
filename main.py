import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

# Étape 1 : Lire le fichier CSV contenant les messages et leurs labels
df = pd.read_csv("dataset.csv")

# Étape 2 : Extraire les messages et les étiquettes
messages = df["message"].tolist()
labels = df["label"].tolist()

# Étape 3 : Transformer les messages en chiffres
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(messages)

# Étape 4 : Entraîner le modèle IA
model = LogisticRegression()
model.fit(X, labels)

print("✅ Le modèle est entraîné avec succès !")

# Étape 5 : Interface en ligne de commande
while True:
    texte = input("\n💬 Écris un message (ou 'exit' pour quitter) : ")
    if texte.lower() == 'exit':
        break
    X_test = vectorizer.transform([texte])
    prediction = model.predict(X_test)
    print("🧠 Sentiment prédit :", prediction[0])
