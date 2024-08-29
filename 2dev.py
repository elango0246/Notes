import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('D:/SIBIYA/DEV LAB/emails.csv')
print(df.head().info)

# Feature Selection
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, roc_curve, auc, confusion_matrix
from sklearn.svm import SVC

# Assuming the column with words is named "text"
df = df.dropna(subset=["text"])

# Separate the features (words) and the target variable (spam)
X = df.drop(["Email No.", "Prediction"], axis=1)  # Exclude Email_no. and spam columns
y = df["Prediction"]

# Perform feature selection using mutual information
selector = SelectKBest(score_func=mutual_info_classif, k=1500)  # Select top 1500 features
X_selected = selector.fit_transform(X, y)

# Get the selected feature names
selected_feature_names = X.columns[selector.get_support()].tolist()

# Create a new dataframe with the selected features
df_selected = df[["Email No.", "Prediction"] + selected_feature_names]
# Print the shape of the new dataframe
print("New dataframe shape:", df_selected.shape)
# Perform train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Print the shape of the train and test sets
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

# Fit the model on the training data
model = MultinomialNB()
model.fit(X_train, y_train)

# Predict probabilities for the test data
probs = model.predict_proba(X_test)
# Predict labels for the test data
predicted_labels = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, predicted_labels)
print("Accuracy:", accuracy)

# Plotting ROC-AUC Curve # Compute the ROC curve and AUC score
fpr, tpr, thresholds = roc_curve(y_test, probs[:, 1])
auc_score = auc(fpr, tpr)
print("AUC Score:", auc_score)

# Plot the ROC curve
plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score)
plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line indicating random chance
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# Plotting Confusion Matrix # Example predicted labels
predicted_labels = np.array(['spam', 'ham', 'spam', 'ham', 'spam'])
# Example true labels
true_labels = np.array(['spam', 'ham', 'ham', 'ham', 'spam'])
# Define the classes and the order of the confusion matrix
classes = ['spam', 'ham']

# Create confusion matrix
cm = confusion_matrix(true_labels, predicted_labels, labels=classes)

# Plot confusion matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

# Using Support Vector Classifier (SVC) from scikit-learn # Example feature vectors (X) and labels (y)
X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
y = np.array(['spam', 'ham', 'spam', 'ham', 'spam'])

# Create an instance of SVC classifier
model = SVC()
model.fit(X, y)

# Predict labels for the same data
predicted_labels = model.predict(X)

# Calculate accuracy
accuracy = accuracy_score(y, predicted_labels)
print("Accuracy:", accuracy)
# Confusion Matrix # Create confusion matrix
cm = confusion_matrix(y, predicted_labels)

# Plot confusion matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

