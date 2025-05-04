# Heart Disease Prediction with Decision Trees & Random Forests

This repository contains Heart Disease Prediction, focusing on building and evaluating tree-based classification models (Decision Tree and Random Forest) to predict the presence of heart disease.

## Objective
To implement, visualize, and compare Decision Tree and Random Forest classifiers using the Heart Disease dataset. Key aspects include understanding tree visualization, managing overfitting via pruning, interpreting feature importances, and evaluating models using cross-validation.

## Dataset
- **Source:** Heart Disease Dataset (commonly derived from UCI Heart Disease dataset).
- **File:** `Heart Disease Dataset.csv`
- **Description:** The dataset includes clinical parameters from patients, aiming to predict the presence or absence of heart disease.
- **Target Variable:** `target` (0 = No Heart Disease, 1 = Heart Disease Present).
- **Features:** Include age, sex, chest pain type (cp), resting blood pressure (trestbps), cholesterol (chol), fasting blood sugar (fbs), resting ECG results (restecg), max heart rate achieved (thalach), exercise-induced angina (exang), ST depression (oldpeak), slope of peak exercise ST segment (slope), number of major vessels colored by fluoroscopy (ca), and thalassemia type (thal).

## Files in this Repository
- `Heart_Disease_Trees.ipynb`: Jupyter Notebook containing the Python code for data loading, analysis, model training (Decision Tree, Random Forest), visualization, and evaluation.
- `Heart Disease Dataset.csv`: The dataset used for the analysis.
- `README.md`: This explanatory file.
- `.gitignore` (Optional): Specifies intentionally untracked files.

## Tools and Libraries Used
- Python 3.x
- Pandas: For data loading and manipulation.
- NumPy: For numerical operations.
- Scikit-learn:
    - `train_test_split`, `cross_val_score`
    - `DecisionTreeClassifier`, `RandomForestClassifier`
    - `plot_tree`, `export_graphviz`
    - `accuracy_score`, `classification_report`, `confusion_matrix`
- Matplotlib & Seaborn: For plotting graphs and visualizations.
- Graphviz: Python library and external software (required for `export_graphviz` rendering).
- Jupyter Notebook: For interactive code development.

## Methodology / Steps Taken
1.  **Load Data:** Imported `Heart Disease Dataset.csv` using Pandas.
2.  **Inspect Data:** Checked data types, missing values (none found), summary statistics, and target variable distribution.
3.  **Feature/Target Split:** Separated features (X) from the target variable (y).
4.  **Train/Test Split:** Divided data into 70% training and 30% testing sets, stratified by the target variable. (Note: Feature scaling was not performed as it's generally unnecessary for tree-based models).
5.  **Basic Decision Tree:** Trained an initial `DecisionTreeClassifier` without constraints on depth to observe baseline performance and potential overfitting.
6.  **Visualize Tree:** Plotted the decision tree using `plot_tree` (limiting depth for readability) and optionally exported using `export_graphviz`.
7.  **Overfitting Analysis:** Trained trees with varying `max_depth` and plotted training vs. test accuracy to identify overfitting and find a potentially optimal depth.
8.  **Pruned Decision Tree:** Trained a `DecisionTreeClassifier` with `max_depth` set based on the previous analysis (or a reasonable default) to reduce overfitting. Evaluated its performance.
9.  **Random Forest:** Trained a `RandomForestClassifier` (e.g., with 100 trees) as an ensemble alternative.
10. **Model Comparison:** Compared the test accuracy and classification reports of the pruned Decision Tree and the Random Forest.
11. **Feature Importance:** Extracted and visualized feature importances from the Random Forest model to identify key predictors.
12. **Cross-Validation:** Performed 5-fold cross-validation on both the pruned Decision Tree and Random Forest using the full dataset to get a more robust estimate of their generalization performance.

## How to Run
1.  Clone this repository:
2.  Navigate to the cloned directory:
3.  Install required libraries
4.  Launch Jupyter Notebook:
5.  Open and run the `Heart_Disease_Trees.ipynb` notebook cells sequentially.

## Questions and Answers

### 1. How does a decision tree work?
A decision tree is a flowchart-like structure used for classification or regression.
*   It starts with a **root node** representing the entire dataset.
*   At each internal **node**, it splits the data based on a specific **feature** and a **threshold** (or category for categorical features). The split chosen is the one that best separates the data according to a certain criterion (like Gini impurity or information gain for classification).
*   This splitting process continues recursively, creating **branches**.
*   The process stops when a node reaches a predefined condition (e.g., maximum depth, minimum samples per leaf) or contains data points of only one class. These terminal nodes are called **leaf nodes**, which represent the final predicted outcome (class label or value).
*   To make a prediction for a new data point, it traverses the tree from the root down, following the branches based on its feature values, until it reaches a leaf node.

### 2. What is entropy and information gain?
These are criteria used to decide the best split at each node in a decision tree (especially for classification).
*   **Entropy:** A measure of impurity or randomness in a set of examples.
    *   Entropy is 0 if all samples in a node belong to the same class (pure node).
    *   Entropy is maximal (usually 1 for binary classification) if the samples are equally mixed among classes.
    *   Formula (for binary classification): `Entropy = -p₁ * log₂(p₁) - p₀ * log₂(p₀)`, where p₁ and p₀ are the proportions of class 1 and class 0 samples in the node.
*   **Information Gain:** The reduction in entropy achieved by splitting the data on a particular feature.
    *   The tree algorithm calculates the information gain for all possible splits (features and thresholds) and chooses the split that results in the *highest* information gain (i.e., the greatest reduction in impurity).
    *   Formula: `Information Gain = Entropy(parent) - [Weighted Average] * Entropy(children)`.
*   *(Note: Gini Impurity is another common criterion similar to Entropy used for measuring impurity and selecting splits.)*

### 3. How is random forest better than a single tree?
Random Forest is an **ensemble method** that builds multiple decision trees and aggregates their predictions. It's generally better because:
*   **Reduces Overfitting:** Single decision trees are prone to overfitting the training data. By averaging the predictions of many trees (each trained on slightly different data and considering different features), Random Forests reduce variance and generalize better to unseen data.
*   **Improved Accuracy:** The ensemble approach typically leads to higher accuracy and more robust predictions than a single, potentially overfit, tree.
*   **Handles High Dimensionality:** Works well even with many features.
*   **Provides Feature Importance:** Can estimate the importance of each feature in making predictions.

### 4. What is overfitting and how do you prevent it?
*   **Overfitting:** Occurs when a model learns the training data *too well*, including its noise and random fluctuations. As a result, it performs poorly on new, unseen data because it hasn't learned the underlying general patterns. In decision trees, this often manifests as very deep trees with nodes that split on very few samples.
*   **Prevention Techniques (for Decision Trees):**
    *   **Pruning:** Limiting the growth of the tree.
        *   **Pre-Pruning:** Stop the tree from growing early by setting constraints like `max_depth` (maximum depth), `min_samples_split` (minimum samples required to split a node), `min_samples_leaf` (minimum samples required in a leaf node).
        *   **Post-Pruning (Cost Complexity Pruning):** Grow the full tree first, then remove branches that provide little predictive power, often using a complexity parameter (`ccp_alpha` in scikit-learn).
    *   **Using Ensemble Methods:** Techniques like Random Forests inherently reduce overfitting.
    *   **Cross-Validation:** Helps tune hyperparameters (like `max_depth`) based on performance on validation sets, not just the training set.

### 5. What is bagging?
*   **Bagging** stands for **Bootstrap Aggregating**. It's an ensemble machine learning technique designed to improve the stability and accuracy of models and reduce variance (overfitting).
*   **How it works:**
    1.  **Bootstrap:** Create multiple random subsets of the original training dataset by sampling *with replacement*. Each subset is typically the same size as the original dataset but contains duplicates and omits some original samples.
    2.  **Aggregate:** Train a separate base model (e.g., a decision tree) independently on each bootstrap sample.
    3.  **Combine:** Aggregate the predictions from all the individual models. For classification, this is usually done by majority voting; for regression, by averaging.
*   **Random Forest is a specific implementation of bagging** applied to decision trees, with an added feature randomness step (considering only a random subset of features at each split).

### 6. How do you visualize a decision tree?
*   **Using `sklearn.tree.plot_tree`:** A built-in function in scikit-learn that uses Matplotlib to draw the tree. Good for quick visualization, especially for smaller or pruned trees. Allows customization like coloring nodes (`filled=True`), showing feature/class names, and limiting depth (`max_depth`).
*   **Using Graphviz:** A dedicated graph visualization software.
    1.  Use `sklearn.tree.export_graphviz` to export the trained tree structure into a `.dot` file format.
    2.  Use the Graphviz library (Python wrapper) or the Graphviz command-line tools (`dot`) to convert the `.dot` file into an image format (like PNG, PDF, SVG). This often produces higher-quality and more readable visualizations, especially for larger trees. Requires installing both the Graphviz software and the Python `graphviz` library.

### 7. How do you interpret feature importance?
*   Feature importance scores indicate the relative contribution of each feature in making predictions within the model (commonly used with Random Forests or Gradient Boosting).
*   **In Tree-based Ensembles (like Random Forest):** Importance is typically calculated based on how much each feature contributes to reducing impurity (e.g., Gini impurity or entropy reduction) across all the splits where that feature was used, averaged over all trees in the forest.
*   **Interpretation:** A higher score means the feature was more influential in the model's predictions. It helps identify the most relevant predictors but doesn't necessarily imply causality or the direction of the effect (use coefficients from linear models for direction). Features with very low importance might be candidates for removal.

### 8. What are the pros/cons of random forests?
*   **Pros:**
    *   **High Accuracy:** Generally provides high prediction accuracy.
    *   **Robust to Overfitting:** Less prone to overfitting compared to single decision trees due to bagging and feature randomness.
    *   **Handles Non-Linearity:** Can capture complex non-linear relationships between features and the target.
    *   **Handles High Dimensions & Missing Data:** Works well with many features and can handle missing values to some extent (though imputation is often preferred).
    *   **Provides Feature Importance:** Offers insights into feature relevance.
    *   **Parallelizable:** Training multiple trees can be done in parallel.
*   **Cons:**
    *   **Less Interpretable:** A forest of trees is much harder to visualize and interpret ("black box" nature) compared to a single decision tree or a linear model.
    *   **Computationally Expensive:** Can require more time and memory to train compared to single trees or linear models, especially with many trees or large datasets.
    *   **Can Still Overfit:** While more robust, they can still overfit on very noisy datasets, requiring tuning (e.g., `max_depth`, `n_estimators`).
    *   **May Not Perform Well on Sparse Data:** Can be less effective than linear models on very high-dimensional, sparse data (like text data).
