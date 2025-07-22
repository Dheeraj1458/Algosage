
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import re
from typing import Dict, List, Tuple
import json

# Configure Streamlit page
st.set_page_config(
    page_title="🤖 ML Learning Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

class AIMLAnalyzer:
    def __init__(self):
        self.ml_patterns = {
            'classification': {
                'keywords': ['classify', 'category', 'class', 'predict category', 'identify type', 
                           'spam detection', 'image recognition', 'sentiment analysis', 'diagnosis'],
                'algorithms': {
                    'primary': ['Random Forest', 'Support Vector Machine', 'Logistic Regression'],
                    'alternatives': ['Decision Tree', 'Naive Bayes', 'K-Nearest Neighbors', 'Neural Networks']
                }
            },
            'regression': {
                'keywords': ['predict', 'estimate', 'forecast', 'price', 'value', 'continuous', 
                           'amount', 'sales prediction', 'revenue', 'cost', 'temperature'],
                'algorithms': {
                    'primary': ['Linear Regression', 'Random Forest Regressor', 'Support Vector Regression'],
                    'alternatives': ['Decision Tree Regressor', 'Ridge Regression', 'Lasso Regression']
                }
            },
            'clustering': {
                'keywords': ['group', 'segment', 'cluster', 'similar', 'pattern', 
                           'customer segmentation', 'market segmentation'],
                'algorithms': {
                    'primary': ['K-Means', 'DBSCAN', 'Hierarchical Clustering'],
                    'alternatives': ['Gaussian Mixture Models', 'Mean Shift', 'Spectral Clustering']
                }
            }
        }
        
        self.algorithm_explanations = {
            'Random Forest': {
                'why': 'Combines multiple decision trees to reduce overfitting and improve accuracy. Handles missing values well and provides feature importance.',
                'pros': ['Robust to overfitting', 'Handles missing values', 'Feature importance', 'Works with mixed data types'],
                'cons': ['Can be slow on large datasets', 'Less interpretable than single trees'],
                'best_for': 'Beginners, tabular data, when you need feature importance'
            },
            'Support Vector Machine': {
                'why': 'Finds optimal decision boundary by maximizing margin between classes. Effective in high-dimensional spaces.',
                'pros': ['Effective in high dimensions', 'Memory efficient', 'Versatile with different kernels'],
                'cons': ['Slow on large datasets', 'Sensitive to feature scaling', 'No probabilistic output'],
                'best_for': 'High-dimensional data, text classification, when you have limited data'
            },
            'Logistic Regression': {
                'why': 'Simple yet powerful linear model for classification. Easy to interpret and implement.',
                'pros': ['Fast and simple', 'No tuning of hyperparameters', 'Provides probabilities', 'Less prone to overfitting'],
                'cons': ['Assumes linear relationship', 'Sensitive to outliers'],
                'best_for': 'Binary classification, when you need interpretability, baseline model'
            },
            'Linear Regression': {
                'why': 'Predicts continuous values using linear relationships. Simple and interpretable.',
                'pros': ['Simple and fast', 'Highly interpretable', 'No hyperparameters', 'Good baseline'],
                'cons': ['Assumes linear relationship', 'Sensitive to outliers'],
                'best_for': 'Simple linear relationships, interpretability is key, baseline model'
            },
            'K-Means': {
                'why': 'Groups data into k clusters based on similarity. Simple and efficient unsupervised learning.',
                'pros': ['Simple and fast', 'Works well with spherical clusters', 'Scales well'],
                'cons': ['Need to specify k', 'Sensitive to initialization', 'Assumes spherical clusters'],
                'best_for': 'Customer segmentation, market research, data exploration'
            }
        }

    def analyze_with_ai(self, problem_statement: str) -> Dict:
        """AI-powered analysis of the problem statement"""
        problem_lower = problem_statement.lower()
        
        # Advanced pattern matching
        detected_type = 'classification'  # default
        confidence_scores = {}
        
        for ml_type, config in self.ml_patterns.items():
            score = 0
            matched_keywords = []
            for keyword in config['keywords']:
                if keyword in problem_lower:
                    score += 1
                    matched_keywords.append(keyword)
            
            # Weight by keyword relevance
            if matched_keywords:
                confidence_scores[ml_type] = score / len(config['keywords'])
        
        if confidence_scores:
            detected_type = max(confidence_scores, key=confidence_scores.get)
        
        # Determine ML category
        ml_category = 'supervised' if detected_type in ['classification', 'regression'] else 'unsupervised'
        
        return {
            'problem_type': detected_type,
            'ml_category': ml_category,
            'confidence_scores': confidence_scores,
            'matched_keywords': [kw for kw in self.ml_patterns[detected_type]['keywords'] if kw in problem_lower]
        }

    def get_algorithm_recommendation(self, problem_type: str) -> Dict:
        """Get detailed algorithm recommendations"""
        if problem_type not in self.ml_patterns:
            problem_type = 'classification'
        
        algorithms = self.ml_patterns[problem_type]['algorithms']
        recommended_algo = algorithms['primary'][0]
        
        return {
            'recommended': recommended_algo,
            'alternatives': algorithms['alternatives'][:3],
            'explanation': self.algorithm_explanations.get(recommended_algo, {}),
            'all_primary': algorithms['primary']
        }

def generate_sample_code(algorithm: str, problem_type: str) -> str:
    """Generate comprehensive sample code"""
    if problem_type == 'classification':
        if 'Random Forest' in algorithm:
            return '''
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset
# df = pd.read_csv('your_dataset.csv')
# X = df.drop('target_column', axis=1)  # Features
# y = df['target_column']  # Target

# Example with synthetic data
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)

# Handle categorical variables if needed
# le = LabelEncoder()
# y = le.fit_transform(y)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Feature scaling (optional for Random Forest but good practice)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train the Random Forest model
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

# Train the model
rf_model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = rf_model.predict(X_test_scaled)
y_pred_proba = rf_model.predict_proba(X_test_scaled)

# Calculate performance metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Cross-validation score
cv_scores = cross_val_score(rf_model, X_train_scaled, y_train, cv=5)

print("=== MODEL PERFORMANCE ===")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Cross-validation Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# Detailed classification report
print("\\n=== CLASSIFICATION REPORT ===")
print(classification_report(y_test, y_pred))

# Feature importance
feature_importance = pd.DataFrame({
    'feature': [f'feature_{i}' for i in range(X.shape[1])],
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\\n=== TOP 10 IMPORTANT FEATURES ===")
print(feature_importance.head(10))

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', ax=axes[0,0])
axes[0,0].set_title('Confusion Matrix')

# Feature Importance
top_features = feature_importance.head(10)
axes[0,1].barh(top_features['feature'], top_features['importance'])
axes[0,1].set_title('Top 10 Feature Importance')

# ROC Curve (for binary classification)
if len(np.unique(y)) == 2:
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
    roc_auc = auc(fpr, tpr)
    axes[1,0].plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
    axes[1,0].plot([0, 1], [0, 1], 'k--')
    axes[1,0].set_xlabel('False Positive Rate')
    axes[1,0].set_ylabel('True Positive Rate')
    axes[1,0].set_title('ROC Curve')
    axes[1,0].legend()

# Cross-validation scores
axes[1,1].boxplot(cv_scores)
axes[1,1].set_title('Cross-validation Scores')
axes[1,1].set_ylabel('Accuracy')

plt.tight_layout()
plt.show()
'''
    elif problem_type == 'regression':
        return '''
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset
# df = pd.read_csv('your_dataset.csv')
# X = df.drop('target_column', axis=1)  # Features
# y = df['target_column']  # Target

# Example with synthetic data
from sklearn.datasets import make_regression
X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create models
linear_model = LinearRegression()
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train models
linear_model.fit(X_train_scaled, y_train)
rf_model.fit(X_train_scaled, y_train)

# Make predictions
y_pred_linear = linear_model.predict(X_test_scaled)
y_pred_rf = rf_model.predict(X_test_scaled)

# Calculate metrics for both models
def calculate_metrics(y_true, y_pred, model_name):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print(f"=== {model_name} PERFORMANCE ===")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R² Score: {r2:.4f}")
    print(f"MSE: {mse:.4f}")
    return rmse, mae, r2

# Calculate metrics
rmse_lr, mae_lr, r2_lr = calculate_metrics(y_test, y_pred_linear, "LINEAR REGRESSION")
rmse_rf, mae_rf, r2_rf = calculate_metrics(y_test, y_pred_rf, "RANDOM FOREST")

# Cross-validation
cv_scores_lr = cross_val_score(linear_model, X_train_scaled, y_train, cv=5, scoring='r2')
cv_scores_rf = cross_val_score(rf_model, X_train_scaled, y_train, cv=5, scoring='r2')

print(f"\\nLinear Regression CV R²: {cv_scores_lr.mean():.4f} (+/- {cv_scores_lr.std() * 2:.4f})")
print(f"Random Forest CV R²: {cv_scores_rf.mean():.4f} (+/- {cv_scores_rf.std() * 2:.4f})")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Actual vs Predicted for Linear Regression
axes[0,0].scatter(y_test, y_pred_linear, alpha=0.7)
axes[0,0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[0,0].set_xlabel('Actual Values')
axes[0,0].set_ylabel('Predicted Values')
axes[0,0].set_title(f'Linear Regression (R² = {r2_lr:.3f})')

# Actual vs Predicted for Random Forest
axes[0,1].scatter(y_test, y_pred_rf, alpha=0.7)
axes[0,1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[0,1].set_xlabel('Actual Values')
axes[0,1].set_ylabel('Predicted Values')
axes[0,1].set_title(f'Random Forest (R² = {r2_rf:.3f})')

# Residual plots
residuals_lr = y_test - y_pred_linear
residuals_rf = y_test - y_pred_rf

axes[1,0].scatter(y_pred_linear, residuals_lr, alpha=0.7)
axes[1,0].axhline(y=0, color='r', linestyle='--')
axes[1,0].set_xlabel('Predicted Values')
axes[1,0].set_ylabel('Residuals')
axes[1,0].set_title('Linear Regression Residuals')

axes[1,1].scatter(y_pred_rf, residuals_rf, alpha=0.7)
axes[1,1].axhline(y=0, color='r', linestyle='--')
axes[1,1].set_xlabel('Predicted Values')
axes[1,1].set_ylabel('Residuals')
axes[1,1].set_title('Random Forest Residuals')

plt.tight_layout()
plt.show()

# Feature importance for Random Forest
if hasattr(rf_model, 'feature_importances_'):
    feature_importance = pd.DataFrame({
        'feature': [f'feature_{i}' for i in range(X.shape[1])],
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\\n=== FEATURE IMPORTANCE (Random Forest) ===")
    print(feature_importance)
'''
    else:  # clustering
        return '''
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset
# df = pd.read_csv('your_dataset.csv')
# X = df.select_dtypes(include=[np.number])  # Only numeric features

# Example with synthetic data
from sklearn.datasets import make_blobs
X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=42)

# Feature scaling (important for clustering)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Determine optimal number of clusters using Elbow Method
inertias = []
silhouette_scores = []
k_range = range(2, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))

# Find optimal k
optimal_k = k_range[np.argmax(silhouette_scores)]
print(f"Optimal number of clusters: {optimal_k}")

# Apply K-Means with optimal k
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
cluster_labels = kmeans.fit_predict(X_scaled)

# Apply DBSCAN for comparison
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan_labels = dbscan.fit_predict(X_scaled)

# Calculate metrics
kmeans_silhouette = silhouette_score(X_scaled, cluster_labels)
dbscan_silhouette = silhouette_score(X_scaled, dbscan_labels) if len(set(dbscan_labels)) > 1 else -1

print(f"\\n=== CLUSTERING PERFORMANCE ===")
print(f"K-Means Silhouette Score: {kmeans_silhouette:.4f}")
print(f"DBSCAN Silhouette Score: {dbscan_silhouette:.4f}")
print(f"K-Means Inertia: {kmeans.inertia_:.4f}")
print(f"Number of clusters found by DBSCAN: {len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)}")

# If we have true labels, calculate adjusted rand score
if 'y_true' in locals():
    ari_kmeans = adjusted_rand_score(y_true, cluster_labels)
    ari_dbscan = adjusted_rand_score(y_true, dbscan_labels)
    print(f"K-Means Adjusted Rand Index: {ari_kmeans:.4f}")
    print(f"DBSCAN Adjusted Rand Index: {ari_dbscan:.4f}")

# Visualization
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Elbow curve
axes[0,0].plot(k_range, inertias, 'bo-')
axes[0,0].set_xlabel('Number of Clusters (k)')
axes[0,0].set_ylabel('Inertia')
axes[0,0].set_title('Elbow Method for Optimal k')
axes[0,0].axvline(x=optimal_k, color='r', linestyle='--', label=f'Optimal k={optimal_k}')
axes[0,0].legend()

# Silhouette scores
axes[0,1].plot(k_range, silhouette_scores, 'ro-')
axes[0,1].set_xlabel('Number of Clusters (k)')
axes[0,1].set_ylabel('Silhouette Score')
axes[0,1].set_title('Silhouette Score vs Number of Clusters')
axes[0,1].axvline(x=optimal_k, color='r', linestyle='--', label=f'Optimal k={optimal_k}')
axes[0,1].legend()

# Original data (if 2D) or PCA projection
if X.shape[1] == 2:
    axes[0,2].scatter(X[:, 0], X[:, 1], c='black', alpha=0.6)
    axes[0,2].set_title('Original Data')
else:
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    axes[0,2].scatter(X_pca[:, 0], X_pca[:, 1], c='black', alpha=0.6)
    axes[0,2].set_title(f'Original Data (PCA - {pca.explained_variance_ratio_.sum():.2f} variance explained)')

# K-Means results
if X.shape[1] == 2:
    axes[1,0].scatter(X[:, 0], X[:, 1], c=cluster_labels, cmap='viridis', alpha=0.7)
    axes[1,0].scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
                     c='red', marker='x', s=200, linewidths=3, label='Centroids')
else:
    axes[1,0].scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='viridis', alpha=0.7)
    # Transform centroids to PCA space
    centroids_pca = pca.transform(scaler.inverse_transform(kmeans.cluster_centers_))
    axes[1,0].scatter(centroids_pca[:, 0], centroids_pca[:, 1], 
                     c='red', marker='x', s=200, linewidths=3, label='Centroids')

axes[1,0].set_title(f'K-Means Clustering (k={optimal_k})')
axes[1,0].legend()

# DBSCAN results
if X.shape[1] == 2:
    axes[1,1].scatter(X[:, 0], X[:, 1], c=dbscan_labels, cmap='viridis', alpha=0.7)
else:
    axes[1,1].scatter(X_pca[:, 0], X_pca[:, 1], c=dbscan_labels, cmap='viridis', alpha=0.7)
axes[1,1].set_title('DBSCAN Clustering')

# Cluster statistics
cluster_stats = pd.DataFrame({
    'Cluster': range(optimal_k),
    'Size': [np.sum(cluster_labels == i) for i in range(optimal_k)],
    'Percentage': [np.mean(cluster_labels == i) * 100 for i in range(optimal_k)]
})

axes[1,2].bar(cluster_stats['Cluster'], cluster_stats['Size'])
axes[1,2].set_xlabel('Cluster')
axes[1,2].set_ylabel('Number of Points')
axes[1,2].set_title('Cluster Sizes')

plt.tight_layout()
plt.show()

print("\\n=== CLUSTER STATISTICS ===")
print(cluster_stats)

# Detailed cluster analysis
print("\\n=== CLUSTER CENTERS (K-Means) ===")
centers_df = pd.DataFrame(kmeans.cluster_centers_, 
                         columns=[f'Feature_{i}' for i in range(X.shape[1])])
print(centers_df)
'''

def main():
    # Initialize the AI analyzer
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = AIMLAnalyzer()
    
    # App header
    st.title("Algosage")
    st.markdown("**Your intelligent companion for machine learning problem solving**")
    
    # Sidebar for navigation
    st.sidebar.title("🎯 Navigation")
    mode = st.sidebar.selectbox(
        "Choose Mode:",
        ["Problem Analysis", "Algorithm Comparison", "Code Generator", "Learning Resources"]
    )
    
    if mode == "Problem Analysis":
        st.header("📝 Describe Your ML Problem")
        
        # Problem input
        problem_statement = st.text_area(
            "Enter your machine learning problem statement:",
            height=150,
            placeholder="e.g., I want to predict house prices based on features like location, size, and age..."
        )
        
        if st.button("🔍 Analyze Problem", type="primary"):
            if problem_statement.strip():
                with st.spinner("🧠 AI is analyzing your problem..."):
                    # Analyze with AI
                    analysis = st.session_state.analyzer.analyze_with_ai(problem_statement)
                    recommendation = st.session_state.analyzer.get_algorithm_recommendation(analysis['problem_type'])
                
                # Display results in columns
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.subheader("🎯 Problem Analysis")
                    st.success(f"**Problem Type:** {analysis['problem_type'].title()}")
                    st.info(f"**ML Category:** {analysis['ml_category'].title()} Learning")
                    
                    if analysis['confidence_scores']:
                        st.write("**Confidence Scores:**")
                        for ml_type, score in analysis['confidence_scores'].items():
                            st.metric(ml_type.title(), f"{score:.2%}")
                    
                    if analysis['matched_keywords']:
                        st.write("**Detected Keywords:**")
                        st.write(", ".join(analysis['matched_keywords']))
                
                with col2:
                    st.subheader("🚀 Algorithm Recommendations")
                    
                    # Recommended algorithm
                    st.success(f"**🌟 Recommended:** {recommendation['recommended']}")
                    
                    if recommendation['explanation']:
                        exp = recommendation['explanation']
                        st.write(f"**Why this algorithm:** {exp.get('why', '')}")
                        
                        # Pros and Cons
                        with st.expander("📊 Detailed Analysis"):
                            if exp.get('pros'):
                                st.write("**✅ Pros:**")
                                for pro in exp['pros']:
                                    st.write(f"• {pro}")
                            
                            if exp.get('cons'):
                                st.write("**❌ Cons:**")
                                for con in exp['cons']:
                                    st.write(f"• {con}")
                            
                            if exp.get('best_for'):
                                st.write(f"**🎯 Best for:** {exp['best_for']}")
                    
                    # Alternative algorithms
                    st.write("**🔄 Alternatives:**")
                    for i, algo in enumerate(recommendation['alternatives'], 1):
                        st.write(f"{i}. {algo}")
                
                # Performance metrics section
                st.subheader("📊 Performance Metrics to Track")
                
                metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                
                if analysis['problem_type'] == 'classification':
                    with metrics_col1:
                        st.metric("Accuracy", "Overall correctness")
                    with metrics_col2:
                        st.metric("Precision", "Positive prediction accuracy")
                    with metrics_col3:
                        st.metric("F1 Score", "Balanced precision & recall")
                elif analysis['problem_type'] == 'regression':
                    with metrics_col1:
                        st.metric("RMSE", "Root Mean Square Error")
                    with metrics_col2:
                        st.metric("MAE", "Mean Absolute Error")
                    with metrics_col3:
                        st.metric("R² Score", "Coefficient of determination")
                else:
                    with metrics_col1:
                        st.metric("Silhouette Score", "Cluster quality")
                    with metrics_col2:
                        st.metric("Inertia", "Within-cluster variation")
                    with metrics_col3:
                        st.metric("ARI", "Adjusted Rand Index")
                
                # Sample code section
                st.subheader("💻 Sample Code")
                code = generate_sample_code(recommendation['recommended'], analysis['problem_type'])
                st.code(code, language='python')
                
                # Download code button
                st.download_button(
                    label="📥 Download Code",
                    data=code,
                    file_name=f"{analysis['problem_type']}_{recommendation['recommended'].replace(' ', '_').lower()}_example.py",
                    mime="text/python"
                )
                
                # Store analysis in session state for other modes
                st.session_state.current_analysis = analysis
                st.session_state.current_recommendation = recommendation
            
            else:
                st.error("Please enter a problem statement!")
    
    elif mode == "Algorithm Comparison":
        st.header("⚖️ Algorithm Comparison")
        
        if hasattr(st.session_state, 'current_analysis'):
            analysis = st.session_state.current_analysis
            recommendation = st.session_state.current_recommendation
            
            st.write(f"Comparing algorithms for **{analysis['problem_type']}** problems:")
            
            # Create comparison table
            all_algorithms = recommendation['all_primary'] + recommendation['alternatives']
            
            comparison_data = []
            for algo in all_algorithms[:5]:  # Limit to 5 for display
                exp = st.session_state.analyzer.algorithm_explanations.get(algo, {})
                comparison_data.append({
                    'Algorithm': algo,
                    'Best For': exp.get('best_for', 'General purpose'),
                    'Complexity': 'Medium' if 'Forest' in algo else 'Low' if 'Linear' in algo or 'Logistic' in algo else 'High',
                    'Interpretability': 'High' if any(x in algo for x in ['Linear', 'Logistic', 'Decision Tree']) else 'Medium'
                })
            
            df = pd.DataFrame(comparison_data)
            st.dataframe(df, use_container_width=True)
            
            # Detailed comparison
            selected_algos = st.multiselect(
                "Select algorithms to compare in detail:",
                all_algorithms,
                default=all_algorithms[:2]
            )
            
            if selected_algos:
                for algo in selected_algos:
                    exp = st.session_state.analyzer.algorithm_explanations.get(algo, {})
                    with st.expander(f"📋 {algo} Details"):
                        col1, col2 = st.columns(2)
                        with col1:
                            if exp.get('pros'):
                                st.write("**✅ Advantages:**")
                                for pro in exp['pros']:
                                    st.write(f"• {pro}")
                        with col2:
                            if exp.get('cons'):
                                st.write("**❌ Disadvantages:**")
                                for con in exp['cons']:
                                    st.write(f"• {con}")
        else:
            st.info("Please analyze a problem first in the 'Problem Analysis' tab!")
    
    elif mode == "Code Generator":
        st.header("🛠️ Code Generator")
        
        algorithm_type = st.selectbox(
            "Select algorithm type:",
            ["Classification", "Regression", "Clustering"]
        )
        
        algorithm_name = st.selectbox(
            "Select specific algorithm:",
            {
                "Classification": ["Random Forest", "Logistic Regression", "Support Vector Machine"],
                "Regression": ["Linear Regression", "Random Forest Regressor", "Support Vector Regression"],
                "Clustering": ["K-Means", "DBSCAN", "Hierarchical Clustering"]
            }[algorithm_type]
        )
        
        if st.button("🔧 Generate Code"):
            code = generate_sample_code(algorithm_name, algorithm_type.lower())
            st.code(code, language='python')
            
            st.download_button(
                label="📥 Download Code",
                data=code,
                file_name=f"{algorithm_type.lower()}_{algorithm_name.replace(' ', '_').lower()}_example.py",
                mime="text/python"
            )
    
    else:  # Learning Resources
        st.header("📚 Learning Resources")
        
        st.subheader("🎯 Quick Tips for Beginners")
        tips = [
            "Start with simple algorithms like Linear/Logistic Regression before moving to complex ones",
            "Always split your data into training and testing sets (80/20 or 70/30 split)",
            "Clean your data: handle missing values, outliers, and encode categorical variables",
            "Use cross-validation to get a better estimate of model performance",
            "Feature engineering often matters more than the choice of algorithm",
            "Understand your problem first: is it supervised or unsupervised?",
            "Don't forget to scale your features for distance-based algorithms",
            "Start with a baseline model and iteratively improve",
            "Visualize your data to understand patterns and relationships",
            "Document your experiments and results for future reference"
        ]
        
        for i, tip in enumerate(tips, 1):
            st.write(f"**{i}.** {tip}")
        
        st.subheader("🔗 Useful Libraries")
        libraries = {
            "scikit-learn": "General-purpose ML library with most common algorithms",
            "pandas": "Data manipulation and analysis",
            "numpy": "Numerical computing and array operations",
            "matplotlib/seaborn": "Data visualization",
            "tensorflow/pytorch": "Deep learning frameworks",
            "xgboost": "Gradient boosting algorithm",
            "plotly": "Interactive visualizations"
        }
        
        for lib, desc in libraries.items():
            st.write(f"**{lib}:** {desc}")
        
        st.subheader("📖 Recommended Learning Path")
        path = [
            "Understand the basics of statistics and probability",
            "Learn Python programming fundamentals",
            "Master pandas and numpy for data manipulation",
            "Start with simple supervised learning (linear/logistic regression)",
            "Learn about model evaluation and cross-validation",
            "Explore ensemble methods (Random Forest, XGBoost)",
            "Understand unsupervised learning (clustering, PCA)",
            "Study deep learning fundamentals",
            "Work on real-world projects and datasets",
            "Learn about MLOps and model deployment"
        ]
        
        for i, step in enumerate(path, 1):
            st.write(f"**Step {i}:** {step}")

if __name__ == "__main__":
    main()
