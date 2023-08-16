
import numpy as np
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pickle

dataset= pd.read_csv('./dataset.csv')
# dataset.tail()

df=dataset
#Removing Symptoms and replacing i  t with symptoms nname
symptom_columns = df.columns[1:]  # Exclude the first column (Disease)
unique_symptoms = df[symptom_columns].values.ravel().tolist()
unique_symptoms = list(set(unique_symptoms))
# unique_symptoms
for symptom in unique_symptoms:
    df[symptom] = df[symptom_columns].apply(lambda row: int(symptom in row.values), axis=1)
df.drop(symptom_columns, axis=1, inplace=True)

#removing leading space
for column in df.columns[1:]:
    new_column_name = str(column).lstrip()  # Convert column name to string before applying lstrip
    df.rename(columns={column: new_column_name}, inplace=True)

#Removing nan colums
if 'nan' in df.columns:
    df = df.drop('nan', axis=1)


df.to_csv("processed_data.csv", index=False)
# consolidated_df = df.groupby('Disease').max().reset_index()
# consolidated_df.to_csv("processed_data.csv", index=False)
final_df=pd.read_csv('./processed_data.csv')
#Spliting data
X=final_df.iloc[:,1:].values
X.shape

y=final_df.iloc[:,0].values
y.shape

#Training
from sklearn.model_selection import train_test_split
X_train,X_test, y_train, y_test= train_test_split(X,y, test_size=0.7,
                                                  random_state=905)

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

classifier = DecisionTreeClassifier(max_depth=50, random_state=42)
classifier.fit(X, y)
y_pred=classifier.predict(X)
accuracy = accuracy_score(y, y_pred)
n_estimators=1000
print(f"n_estimators = {n_estimators}, Accuracy = {accuracy:.2f}")


classifier.fit(X_train,y_train)
classifier.fit(X,y)

#Predicting
y_pred = classifier.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

pickle.dump(classifier, open('model.pkl', 'wb'))