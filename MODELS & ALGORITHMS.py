import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import _tree


# Load the dataset
file_path = r"C:\Users\raghu\Desktop\AIT 614\AIT 614 FINAL PROJECT\AIT 614 PROJECT PROPOSAL\cleaned_diabetes_dataset.csv"
df = pd.read_csv(file_path)

# Features and target variable
X = df.drop('Diabetes_012', axis=1)
y = df['Diabetes_012']

# Normalization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split training and testing set
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Logistic Regression Model
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
y_prob = logreg.predict_proba(X_test)
roc_auc = roc_auc_score(y_test, y_prob, multi_class='ovr')
report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print("Logistic Regression Model Performance Metrics:")
print(f"Accuracy: {accuracy:.2%}")
print(f"ROC AUC Score: {roc_auc:.2%}")
print("\nClassification Report:")
print(report)
print("\nConfusion Matrix:")
conf_matrix_df = pd.DataFrame(conf_matrix,
                              index=['True Non-diabetic', 'True Pre-diabetic', 'True Diabetic'],
                              columns=['Predicted Non-diabetic', 'Predicted Pre-diabetic', 'Predicted Diabetic'])
print(conf_matrix_df)

# Decision Tree Model
dtree = DecisionTreeClassifier(max_depth=4, random_state=42)
dtree.fit(X_train, y_train)

plt.figure(figsize=(20, 10))
tree_plot = plot_tree(dtree, feature_names=X.columns,
                      class_names=['Non-diabetic', 'Pre-diabetic', 'Diabetic'],
                      filled=True, fontsize=9)
plt.title("Decision Tree Visualization")
plt.show()


def get_rules(tree, feature_names):
    rules_list = []

    def recurse(node, path_rules):
        if tree.tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_names[tree.tree_.feature[node]]
            threshold = tree.tree_.threshold[node]
            recurse(tree.tree_.children_left[node], path_rules + [f"{name} <= {threshold:.2f}"])
            recurse(tree.tree_.children_right[node], path_rules + [f"{name} > {threshold:.2f}"])
        else:
            rules_list.append((path_rules, tree.tree_.value[node]))
    recurse(0, [])
    return rules_list


rules_list = get_rules(dtree, X.columns)
diabetic_rules = [rules for rules, values in rules_list if values.argmax() == 2]
non_diabetic_rules = [rules for rules, values in rules_list if values.argmax() == 0]

print("Diabetic health indicators:")
for rule in diabetic_rules:
    print(" , ".join(rule))
print("\nNon-Diabetic health indicators:")
for rule in non_diabetic_rules:
    print(", ".join(rule))

# Random Forest Model
random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
random_forest.fit(X_train, y_train)
y_pred_rf = random_forest.predict(X_test)

accuracy_rf = accuracy_score(y_test, y_pred_rf)
report_rf = classification_report(y_test, y_pred_rf)
conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)

print("Random Forest Model Performance Metrics:")
print(f"Accuracy: {accuracy_rf:.2%}")
print("\nClassification Report:")
print(report_rf)
print("\nConfusion Matrix:")
print(pd.DataFrame(conf_matrix_rf,
                   index=['True Non-diabetic', 'True Pre-diabetic', 'True Diabetic'],
                   columns=['Predicted Non-diabetic', 'Predicted Pre-diabetic', 'Predicted Diabetic']))

# Plot confusion matrix for Random Forest
conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_rf, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predicted Non-diabetic', 'Predicted Pre-diabetic', 'Predicted Diabetic'],
            yticklabels=['True Non-diabetic', 'True Pre-diabetic', 'True Diabetic'])
plt.title('Confusion Matrix - Random Forest Classifier')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
