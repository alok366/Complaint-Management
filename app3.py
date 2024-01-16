import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import tkinter as tk
from tkinter import Label, Entry, Button

nltk.download('stopwords')
nltk.download('punkt')

tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words=stopwords.words('english'))

df = None
clf = None
X_train_tfidf = None
y_train = None
encoder = LabelEncoder()

def preprocess_text(text):
    if isinstance(text, str):  # Check if text is a string
        text = text.lower()
        text = ' '.join([word for word in word_tokenize(text) if word.isalnum()])
        return text
    else:
        return ''  # Replace NaN or non-string values with an empty string
    
# Load the dataset and perform data preprocessing 

df = pd.read_csv("complaintDB.csv")
df.head()

# columns_to_drop = ["Unnamed: 2", "Unnamed: 3", "Unnamed: 4","Unnamed: 5", "Unnamed: 6", "Unnamed: 7","Unnamed: 8", "Unnamed: 9", "Unnamed: 10","Unnamed: 11", "Unnamed: 12", "Unnamed: 13","Unnamed: 14", "Unnamed: 15", "Unnamed: 16","Unnamed: 17", "Unnamed: 18", "Unnamed: 19", "Unnamed: 18", "Unnamed: 19", "Unnamed: 20", "Unnamed: 21", "Unnamed: 22", "Unnamed: 23"]
columns_to_drop = ["Unnamed: 2", "Unnamed: 3"]
df.drop(columns=columns_to_drop, axis=1, inplace=True)
df.head()

df.head(20)
df.rename(columns={"Enter a Complain (20-40 words)\n\nex: about Bad roads, sewage clogging, electricity pole etc": "complaint"}, inplace=True)

df['complaint'] = df['complaint'].apply(preprocess_text)

df['Department_ID'] = encoder.fit_transform(df['Department Name'])
df.head()

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['complaint'], df['Department_ID'], test_size=0.2, random_state=42)

# TF-IDF Vectorization
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

clf = MultinomialNB()
clf.fit(X_train_tfidf, y_train)

y_pred = clf.predict(X_test_tfidf)

# Decode numerical labels back to department names
y_pred_department = encoder.inverse_transform(y_pred)
y_test_department = encoder.inverse_transform(y_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)    

# Create a Tkinter application window
app = tk.Tk()
app.title("Complaint Department Predictor")

# Create a label for displaying accuracy
accuracy_label = Label(app, text="Accuracy:")
accuracy_label.pack()

# Create a label for displaying the classification report
report_label = Label(app, text="Classification Report:")
report_label.pack()

# Create a label for displaying the predicted department
predicted_label = Label(app, text="Predicted Department:")
predicted_label.pack()

# Create a text entry field for user input
complaint_entry = Entry(app, width=50)
complaint_entry.pack()

# Create a function to predict the department based on the entered complaint
def predict_department():
    new_complaint = complaint_entry.get()
    new_complaint = preprocess_text(new_complaint)
    new_complaint_tfidf = tfidf_vectorizer.transform([new_complaint])
    predicted_department = clf.predict(new_complaint_tfidf)
    predicted_department_name = encoder.inverse_transform(predicted_department)
    predicted_label.config(text="Predicted Department: " + predicted_department_name[0])

# Create a button to trigger the prediction
predict_button = Button(app, text="Predict Department", command=predict_department)
predict_button.pack()

# Display accuracy and classification report
accuracy_label.config(text="Accuracy: " + str(accuracy))
report_label.config(text="Classification Report:\n" + report)

# Start the Tkinter main loop
app.mainloop()
