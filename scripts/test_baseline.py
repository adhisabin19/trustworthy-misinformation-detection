
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

texts = [
    "Scientists confirm results using verified evidence",
    "Researchers publish findings in a scientific journal",
    "Official reports provide verified information",
    "Secret miracle cure hidden by scientists",
    "Shocking conspiracy proves all experts are lying",
    "Fake claims promise impossible medical results",
]

labels = [1, 1, 1, 0, 0, 0]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

model = LogisticRegression(max_iter=1000)
model.fit(X, labels)

prediction = model.predict(
    vectorizer.transform(["Researchers verify scientific evidence"])
)

print("TrustFact baseline model executed successfully!")
print("Predicted class:", prediction[0])
