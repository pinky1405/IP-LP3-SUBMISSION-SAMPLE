import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveClassifier
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

news = pd.read_csv("news.csv")
labels = news.label
x_train, x_test, y_train, y_test = train_test_split(news['text'], labels, test_size=0.2, random_state=7)
vectors = TfidfVectorizer(stop_words='english', max_df=0.7)
t_train = vectors.fit_transform(x_train)
t_test = vectors.transform(x_test)
pred_model = PassiveAggressiveClassifier(max_iter=50)
pred_model.fit(t_train, y_train)
y_pred = pred_model.predict(t_test)
newsscore = accuracy_score(y_test, y_pred)
confusion_mat = confusion_matrix(y_test, y_pred, labels=['FAKE', 'REAL'])
print("Model-Accuracy : " + str(newsscore))
print("Confusion Matrix is :-\n" + str(confusion_mat))
print("True Positive News  :- " + str(confusion_mat[0][0]) + "\nTrue Negative News  :- " + str(confusion_mat[1][1]))
print("False Positive News :- " + str(confusion_mat[1][0]) + "\nFalse Negative News :- " + str(confusion_mat[0][1]))
text = input("Write  news to check is it False or True: ")
print("The News is:",text)
pkl_filename = "testmodel.pkl"
try:
    with open(pkl_filename, 'rb') as file:
        model = pickle.load(file)
        tf1 = pickle.load(open("tfidf.pkl", 'rb'))
    print("Model Runned Successfully.")
except:
    print("Model not found")
    

tf1_new = TfidfVectorizer(analyzer='word', stop_words = "english", vocabulary = tf1.vocabulary_)

X_temp = tf1_new.fit_transform([text])
X_temp.toarray()
predict = model.predict(X_temp)
print(predict)

