import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-learn imports
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Graphviz import (optional, for higher quality tree visualization)
import graphviz

# Configure settings
%matplotlib inline
plt.style.use('seaborn-v0_8-darkgrid')
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
import warnings
warnings.filterwarnings('ignore')

print("Libraries imported.")

# Load the dataset
file_path = 'heart.csv' # Ensure filename matches
df = pd.read_csv(file_path)

# Basic Inspection
print("Dataset Information:")
df.info()

print("\nFirst 5 Rows:")
display(df.head())

print("\nTarget Variable Distribution (target: 0=No disease, 1=Disease):")
print(df['target'].value_counts())
print(f"\nPercentage of Disease cases (Class 1): {df['target'].mean() * 100:.2f}%") # Check balance

print("\nSummary Statistics:")
display(df.describe())

print("\nChecking for Missing Values:")
print(df.isnull().sum())
print(f"\nTotal missing values: {df.isnull().sum().sum()}") # Typically none in this dataset

# Separate features (X) and target (y)
X = df.drop('target', axis=1) # All columns except target
y = df['target']             # Target variable

print("\nShape of Features (X):", X.shape)
print("Shape of Target (y):", y.shape)
print("\nFeature names:", X.columns.tolist())

# Split data into training (70%) and testing (30%) sets
# Stratify ensures proportion of target classes is similar in train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print("--- Data Split Shapes ---")
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)

# Initialize the Decision Tree Classifier
# random_state ensures reproducibility
# No depth limit initially to see potential overfitting
dt_basic = DecisionTreeClassifier(random_state=42)

# Train the model
dt_basic.fit(X_train, y_train)

# Predict on the test set
y_pred_basic_dt = dt_basic.predict(X_test)

# Evaluate basic accuracy
accuracy_basic_dt = accuracy_score(y_test, y_pred_basic_dt)

print("--- Basic Decision Tree (Unpruned) ---")
print(f"Test Accuracy: {accuracy_basic_dt:.4f}")
# We expect this might be lower than training accuracy (sign of overfitting)
print(f"Train Accuracy: {accuracy_score(y_train, dt_basic.predict(X_train)):.4f}")

print("\n--- Visualizing the Decision Tree (Limited Depth for Readability) ---")

# --- Method 1: Using sklearn.tree.plot_tree (Good for smaller trees) ---
plt.figure(figsize=(20, 10)) # Adjust size as needed
plot_tree(dt_basic,
          filled=True, # Color nodes by majority class
          rounded=True, # Use rounded boxes
          feature_names=X.columns.tolist(), # Show feature names
          class_names=['No Disease', 'Disease'], # Show class names
          max_depth=3, # Limit depth for readability in the plot
          fontsize=10)
plt.title("Decision Tree Visualization (Depth limited to 3)")
plt.show()

# --- Method 2: Using Graphviz (Better for larger trees, requires Graphviz install) ---
# Export the tree to a .dot file
dot_data = export_graphviz(dt_basic,
                           out_file=None, # Don't save to file, keep in memory
                           feature_names=X.columns.tolist(),
                           class_names=['No Disease', 'Disease'],
                           filled=True,
                           rounded=True,
                           special_characters=True,
                           max_depth=4) # Limit depth here too if desired for the image

# Create graph from .dot data
# If Graphviz is not installed or not in PATH, this will error
try:
    graph = graphviz.Source(dot_data)
    # You can render it to a file (e.g., PNG)
    # graph.render("heart_disease_decision_tree") # Saves heart_disease_decision_tree.gv and .gv.png
    print("Graphviz visualization object created (display depends on environment).")
    # In some environments (like standard Jupyter), displaying the graph object might render it:
    # display(graph)
    # Or save and show image:
    # graph.view() # Opens in default viewer
    # If direct display doesn't work, check the saved PNG file.
except Exception as e:
    print(f"\nGraphviz visualization failed: {e}")
    print("Ensure Graphviz is installed AND in your system's PATH.")
    print("Alternatively, analyze the plot_tree visualization above.")

# Test different max_depth values to see effect on overfitting
max_depths = range(1, 15) # Test depths from 1 to 14
train_accuracies = []
test_accuracies = []

for depth in max_depths:
    # Train a tree with the current max_depth
    dt_depth = DecisionTreeClassifier(max_depth=depth, random_state=42)
    dt_depth.fit(X_train, y_train)

    # Record training accuracy
    train_acc = accuracy_score(y_train, dt_depth.predict(X_train))
    train_accuracies.append(train_acc)

    # Record testing accuracy
    test_acc = accuracy_score(y_test, dt_depth.predict(X_test))
    test_accuracies.append(test_acc)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(max_depths, train_accuracies, 'bo-', label='Training Accuracy')
plt.plot(max_depths, test_accuracies, 'ro-', label='Test Accuracy')
plt.xlabel('Max Depth')
plt.ylabel('Accuracy')
plt.title('Decision Tree Accuracy vs. Max Depth')
plt.legend()
plt.grid(True)
plt.show()

print("\n--- Overfitting Analysis ---")
print("Observe the plot:")
print("- Training accuracy usually increases or stays high as depth increases.")
print("- Test accuracy often increases initially, peaks, and then might decrease or plateau.")
print("- A large gap between training and test accuracy indicates overfitting.")
# Find the depth that gives a good balance (often where test accuracy peaks or starts to plateau)
best_depth_index = np.argmax(test_accuracies)
best_depth = max_depths[best_depth_index]
print(f"\nBest test accuracy ({test_accuracies[best_depth_index]:.4f}) achieved at max_depth = {best_depth}")

# Train a decision tree with the potentially better max_depth found above
# Let's use the best_depth identified, or choose a reasonable value like 3 or 4 if the peak isn't clear
pruned_depth = best_depth # Or set manually, e.g., pruned_depth = 4

print(f"\n--- Training Pruned Decision Tree (max_depth={pruned_depth}) ---")
dt_pruned = DecisionTreeClassifier(max_depth=pruned_depth, random_state=42)
dt_pruned.fit(X_train, y_train)

# Evaluate the pruned tree
y_pred_pruned = dt_pruned.predict(X_test)
accuracy_pruned = accuracy_score(y_test, y_pred_pruned)
report_pruned = classification_report(y_test, y_pred_pruned, target_names=['No Disease', 'Disease'])
cm_pruned = confusion_matrix(y_test, y_pred_pruned)

print(f"Pruned Tree Test Accuracy: {accuracy_pruned:.4f}")
print("\nClassification Report (Pruned Tree):")
print(report_pruned)

print("\nConfusion Matrix (Pruned Tree):")
plt.figure(figsize=(6, 4))
sns.heatmap(cm_pruned, annot=True, fmt='d', cmap='Greens',
            xticklabels=['No Disease', 'Disease'], yticklabels=['No Disease', 'Disease'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title(f'Confusion Matrix (Pruned DT, max_depth={pruned_depth})')
plt.show()

# Initialize the Random Forest Classifier
# n_estimators: number of trees in the forest
# random_state for reproducibility
# n_jobs=-1 uses all available CPU cores for faster training
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, max_depth=pruned_depth)
                                 # Optional: can add max_depth here too, often similar to pruned DT depth is good starting point
                                 # Or leave max_depth=None for default behavior (trees grow deep)

# Train the model
rf_model.fit(X_train, y_train)

# Predict on the test set
y_pred_rf = rf_model.predict(X_test)

# Evaluate the Random Forest
accuracy_rf = accuracy_score(y_test, y_pred_rf)
report_rf = classification_report(y_test, y_pred_rf, target_names=['No Disease', 'Disease'])
cm_rf = confusion_matrix(y_test, y_pred_rf)

print("\n--- Random Forest Evaluation ---")
print(f"Random Forest Test Accuracy: {accuracy_rf:.4f}")
print(f"(Compare to Pruned DT Accuracy: {accuracy_pruned:.4f})")

print("\nClassification Report (Random Forest):")
print(report_rf)

print("\nConfusion Matrix (Random Forest):")
plt.figure(figsize=(6, 4))
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Oranges',
            xticklabels=['No Disease', 'Disease'], yticklabels=['No Disease', 'Disease'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix (Random Forest)')
plt.show()

# Get feature importances from the trained Random Forest model
importances = rf_model.feature_importances_
feature_names = X.columns

# Create a DataFrame for better visualization
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

print("\n--- Feature Importances (from Random Forest) ---")
print(importance_df)

# Plot feature importances
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df.head(10)) # Plot top 10
plt.title('Top 10 Feature Importances from Random Forest')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.grid(axis='x')
plt.show()

print("\nInterpretation: Features with higher scores contribute more, on average,")
print("to the decisions made by the trees in the Random Forest.")

# Cross-validation provides a more robust estimate of model performance
# It trains and tests the model multiple times on different subsets of the data

# Use the entire dataset (X, y) for cross-validation
# cv=5 means 5-fold cross-validation

print("\n--- Cross-Validation Evaluation ---")

# Cross-validate the Pruned Decision Tree
cv_scores_dt = cross_val_score(dt_pruned, X, y, cv=5, scoring='accuracy')
print(f"Pruned Decision Tree CV Accuracy Scores: {cv_scores_dt}")
print(f"Pruned Decision Tree CV Mean Accuracy: {np.mean(cv_scores_dt):.4f}")
print(f"Pruned Decision Tree CV Std Dev Accuracy: {np.std(cv_scores_dt):.4f}")

# Cross-validate the Random Forest
cv_scores_rf = cross_val_score(rf_model, X, y, cv=5, scoring='accuracy')
print(f"\nRandom Forest CV Accuracy Scores: {cv_scores_rf}")
print(f"Random Forest CV Mean Accuracy: {np.mean(cv_scores_rf):.4f}")
print(f"Random Forest CV Std Dev Accuracy: {np.std(cv_scores_rf):.4f}")

print("\nComparison: Higher mean CV accuracy suggests better generalization.")
print("Lower standard deviation suggests more consistent performance across different data subsets.")
