import pandas as pd
import os

def convert_to_df(directory):
    file_list = os.listdir(directory)

    # Iterate over the filenames and read each file
    dfs = []  # List to store all DataFrames

    for filename in file_list:
        if filename.endswith('.csv'):  # Consider only CSV files
            file_path = os.path.join(directory, filename)
            df = pd.read_csv(file_path)
            dfs.append(df)

    return dfs

neg_directory = 'Negative'
pos_directory = 'Positive'
print("here")
pos_dfs = convert_to_df(pos_directory)
neg_dfs = convert_to_df(neg_directory)

neg = pd.concat(neg_dfs)
pos = pd.concat(pos_dfs)
data = pd.concat([neg, pos])
text_data = data['sentence']
labels = data['label']


import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from imblearn.under_sampling import RandomUnderSampler

# Step 1: Load the CSV file and extract text and binary label columns

# Step 2: Preprocess the text data to convert it into a bag of words representation
vectorizer = CountVectorizer()
text_features = vectorizer.fit_transform(text_data)

# Step 3: Perform undersampling to balance the classes
undersampler = RandomUnderSampler(random_state=42)
text_features_resampled, labels_resampled = undersampler.fit_resample(text_features, labels)

# Step 4: Split the resampled dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(text_features_resampled, labels_resampled, test_size=0.2, random_state=42)

# Step 5: Train a Random Forest classifier using the bag of words features and binary labels
model = RandomForestClassifier()
model.fit(X_train, y_train)
print("there")
# Step 6: Evaluate the trained model on the testing set
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
