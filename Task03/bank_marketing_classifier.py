import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Load the dataset
# Using a reliable raw link from a dataset repository
url = "https://raw.githubusercontent.com/selva86/datasets/master/bank-full.csv"
try:
    # Try with semicolon first, then comma as fallback
    df = pd.read_csv(url, sep=';')
    if len(df.columns) == 1: # If only one column, it likely failed to split by ';'
        df = pd.read_csv(url, sep=',')
    print("Dataset loaded successfully!")
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit()

# 1. Exploratory Data Analysis (EDA)
print("\nFirst 5 rows of the dataset:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nTarget Variable Distribution:")
print(df['y'].value_counts())

# 2. Preprocessing
le = LabelEncoder()
for col in df.select_dtypes(include=['object']).columns:
    df[col] = le.fit_transform(df[col])

# Use only 2 key features for the absolute simplest tree
X = df[['duration', 'nr.employed']]
y = df['y']

# 3. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Build a very simple Decision Tree (depth 2)
clf = DecisionTreeClassifier(max_depth=2, random_state=42)
clf.fit(X_train, y_train)

# 5. Evaluation
y_pred = clf.predict(X_test)
print(f"\nModel Accuracy: {accuracy_score(y_test, y_pred):.2%}")

# 6. Visualize the Tree
plt.figure(figsize=(12,8))
plot_tree(clf, feature_names=X.columns, class_names=['No', 'Yes'], filled=True)
plt.title("Simple Decision Tree")
plt.savefig("decision_tree.png")
print("\nDecision Tree visualization saved as 'decision_tree.png'")
