from transformers import pipeline
from sklearn.metrics import accuracy_score
import nltk

print("==================================================")
print("  ÉTAPE 1 : Configuration et Environnement")
print("==================================================")
nltk.download('punkt', quiet=True)
print("✓ Ressources NLTK vérifiées et chargées.\n")


print("==================================================")
print("  ÉTAPE 2 : Classification d'Opinion (Sentiment Analysis)")
print("==================================================")
sentiment_pipeline = pipeline(
    task="sentiment-analysis",
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english"
)

texts_sent = [
    "I really liked the movie!!",
    "Great job ruining my day.",
    "This product exceeded my expectations.",
    "Wow, just what I needed... another problem.",
    "Absolutely fantastic experience!"
]
true_labels_sent = ["POSITIVE", "NEGATIVE", "POSITIVE", "NEGATIVE", "POSITIVE"]

results_sent = sentiment_pipeline(texts_sent)
preds_sent = [res['label'] for res in results_sent]

print("Résultats des prédictions :")
for text, res in zip(texts_sent, results_sent):
    print(f"  [{res['label']:<8}] ({res['score']:.4f}) -> '{text}'")

acc_sent = accuracy_score(true_labels_sent, preds_sent)
print(f"\n>> Accuracy Sentiment Analysis : {acc_sent * 100:.1f}%\n")


print("==================================================")
print("  ÉTAPE 3 : Classification Zero-Shot")
print("==================================================")
zero_shot_classifier = pipeline(
    task="zero-shot-classification",
    model="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
)

text_zs = "The national football team won the cup yesterday."
labels_zs = ["sports", "technology", "health"]

res_zs = zero_shot_classifier(text_zs, labels_zs)
print(f"Texte analysé : '{text_zs}'")
print("Scores par catégorie :")
for label, score in zip(res_zs['labels'], res_zs['scores']):
    print(f"  - {label:<12} : {score:.4f}")
print()

print("==================================================")
print("  ÉTAPE 4 : Inférence Question–Réponse (QNLI)")
print("==================================================")
qnli_pipeline = pipeline(
    task="text-classification",
    model="cross-encoder/qnli-electra-base"
)
qnli_map = {"LABEL_0": "Réponse présente", "LABEL_1": "Pas de réponse"}

# 1. Définir les données d'abord
qnli_dataset = [
    {"passage": "Python was created by Guido van Rossum.", "question": "Who created Python?", "expected": "LABEL_0"},
    {"passage": "The Eiffel Tower is located in Paris.", "question": "How tall is it?", "expected": "LABEL_1"},
    {"passage": "Water freezes at 0 degrees Celsius.", "question": "At what temperature does water freeze?", "expected": "LABEL_0"}
]

preds_qnli, true_qnli = [], []

# 2. Exécuter la boucle d'inférence
for item in qnli_dataset:
    out = qnli_pipeline({"text": item["question"], "text_pair": item["passage"]})
    preds_qnli.append(out["label"])
    true_qnli.append(item["expected"])
    print(f"Question : '{item['question']}'")
    print(f"Passage  : '{item['passage']}'")
    print(f"-> Résultat : {qnli_map[out['label']]} (Score : {out['score']:.4f})\n")

# 3. Calculer l'accuracy À LA FIN de l'étape
acc_qnli = accuracy_score(true_qnli, preds_qnli)
print(f">> Accuracy QNLI : {acc_qnli * 100:.1f}%\n")

print("==================================================")
print("  ÉTAPE 5 : Détection de Paraphrases (QQP)")
print("==================================================")
qqp_pipeline = pipeline(
    task="text-classification",
    model="textattack/bert-base-uncased-QQP"
)
qqp_map = {"LABEL_0": "Non-paraphrase", "LABEL_1": "Paraphrase"}

qqp_dataset = [
    {"q1": "How do I reset my password?", "q2": "What are the steps to change my password?", "expected": "LABEL_1"},
    {"q1": "How do I cook pasta?", "q2": "Where can I buy fresh pasta?", "expected": "LABEL_0"}
]

preds_qqp, true_qqp = [], []
for item in qqp_dataset:
    out = qqp_pipeline({"text": item["q1"], "text_pair": item["q2"]})
    preds_qqp.append(out["label"])
    true_qqp.append(item["expected"])
    print(f"Q1 : '{item['q1']}'")
    print(f"Q2 : '{item['q2']}'")
    print(f"-> Résultat : {qqp_map[out['label']]} (Score : {out['score']:.4f})\n")

acc_qqp = accuracy_score(true_qqp, preds_qqp)
print(f">> Accuracy QQP : {acc_qqp * 100:.1f}%\n")


print("==================================================")
print("  ÉTAPE 6 : Acceptabilité Grammaticale (CoLA)")
print("==================================================")
cola_pipeline = pipeline(
    task="text-classification",
    model="textattack/distilbert-base-uncased-CoLA"
)
cola_map = {"LABEL_0": "Inacceptable", "LABEL_1": "Acceptable"}

cola_dataset = [
    {"text": "The cat sat on the mat.", "expected": "LABEL_1"},
    {"text": "The cat on sat mat the.", "expected": "LABEL_0"},
    {"text": "Colorless green ideas sleep furiously.", "expected": "LABEL_1"}
]

preds_cola, true_cola = [], []
for item in cola_dataset:
    out = cola_pipeline(item["text"])[0]
    preds_cola.append(out["label"])
    true_cola.append(item["expected"])
    print(f"Phrase : '{item['text']}'")
    print(f"-> Résultat : {cola_map[out['label']]} (Score : {out['score']:.4f})\n")

acc_cola = accuracy_score(true_cola, preds_cola)
print(f">> Accuracy CoLA : {acc_cola * 100:.1f}%\n")


print("==================================================")
print("  EXECUTION DU LAB TERMINÉE AVEC SUCCÈS !")
print("==================================================")