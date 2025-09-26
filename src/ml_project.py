import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.metrics import roc_curve, auc
from scipy.spatial.distance import cosine, euclidean
import os
from collections import defaultdict
import warnings
from sklearn.preprocessing import LabelEncoder
from scipy.interpolate import interp1d
warnings.filterwarnings('ignore')

def load_data(filepath):
    """
    Loads data from pickle file
    """
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data

def flatten_data(data):
    """
    Transforms hierarchical structure into embedding arrays and labels
    Returns:
    - embeddings: array (n_samples, 320)
    - labels: array (n_samples,) with syndrome IDs
    - metadata: list of dictionaries with syndrome, subject, and image information
    """
    embeddings = []
    labels = []
    metadata = []
    
    for syndrome_id, subjects in data.items():
        for subject_id, images in subjects.items():
            for image_id, embedding in images.items():
                embeddings.append(embedding)
                labels.append(syndrome_id)
                metadata.append({
                    'syndrome_id': syndrome_id,
                    'subject_id': subject_id,
                    'image_id': image_id
                })
    
    return np.array(embeddings), np.array(labels), metadata

def exploratory_data_analysis(labels):
    """
    Performs exploratory data analysis
    """
    # Count samples per syndrome
    unique_labels, counts = np.unique(labels, return_counts=True)
    
    print("Exploratory Data Analysis:")
    print(f"Total number of samples: {len(labels)}")
    print(f"Number of different syndromes: {len(unique_labels)}")
    print(f"Distribution by syndrome:")
    for label, count in zip(unique_labels, counts):
        print(f"  Syndrome {label}: {count} samples")
    
    print(f"\nAverage samples per syndrome: {np.mean(counts):.2f}")
    print(f"Standard deviation of samples per syndrome: {np.std(counts):.2f}")
    print(f"Minimum samples per syndrome: {np.min(counts)}")
    print(f"Maximum samples per syndrome: {np.max(counts)}")
    
    # Plot distribution
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(unique_labels)), counts)
    plt.title('Sample Distribution by Syndrome')
    plt.xlabel('Syndrome Index')
    plt.ylabel('Number of Samples')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('results/distribution_syndromes.png')
    plt.close()

def visualize_embeddings_tsne(embeddings, labels, sample_size=1000):
    """
    Visualizes embeddings using t-SNE
    """
    # Sample to make t-SNE faster
    if len(embeddings) > sample_size:
        indices = np.random.choice(len(embeddings), sample_size, replace=False)
        sample_embeddings = embeddings[indices]
        sample_labels = labels[indices]
    else:
        sample_embeddings = embeddings
        sample_labels = labels
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(sample_embeddings) - 1))
    embeddings_2d = tsne.fit_transform(sample_embeddings)
    
    # Plot
    plt.figure(figsize=(12, 8))
    unique_labels = np.unique(sample_labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        mask = sample_labels == label
        plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                   c=[colors[i]], label=f'Syndrome {label}', alpha=0.7)
    
    plt.title('t-SNE Visualization of Embeddings (sample)')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('results/tsne_visualization.png')
    plt.close()
    
    print(f"\nt-SNE visualization completed with {len(sample_embeddings)} samples")

def calculate_metrics(y_true, y_pred, y_pred_proba, classes):
    """
    Calculates evaluation metrics
    """
    # AUC
    if len(classes) == 2:
        auc_score = roc_auc_score(y_true, y_pred_proba[:, 1])
    else:
        auc_score = roc_auc_score(y_true, y_pred_proba, multi_class='ovr')
    
    # F1-Score
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    # Accuracy
    accuracy = accuracy_score(y_true, y_pred)
    
    return auc_score, f1, accuracy

def knn_classification(embeddings, labels, distance_metric='euclidean'):
    """
    Implements KNN classification with different distance metrics
    """
    print(f"\nPerforming KNN classification with {distance_metric} metric...")
    
    # 10-fold cross-validation
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    results = {}
    
    # Test different k values (1 to 15)
    k_values = range(1, 16)
    cv_scores_auc = []
    cv_scores_f1 = []
    cv_scores_accuracy = []
    
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k, metric=distance_metric)
        
        # Store scores for each fold
        fold_auc = []
        fold_f1 = []
        fold_acc = []
        
        for train_idx, val_idx in skf.split(embeddings, labels):
            X_train, X_val = embeddings[train_idx], embeddings[val_idx]
            y_train, y_val = labels[train_idx], labels[val_idx]
            
            knn.fit(X_train, y_train)
            y_pred = knn.predict(X_val)
            
            # Calculate probabilities for AUC
            y_pred_proba = knn.predict_proba(X_val)
            
            # Calculate metrics
            auc_score, f1, acc = calculate_metrics(y_val, y_pred, y_pred_proba, np.unique(labels))
            
            fold_auc.append(auc_score)
            fold_f1.append(f1)
            fold_acc.append(acc)
        
        # Average scores across folds
        cv_scores_auc.append(np.mean(fold_auc))
        cv_scores_f1.append(np.mean(fold_f1))
        cv_scores_accuracy.append(np.mean(fold_acc))
    
    # Find optimal k
    optimal_k_idx = np.argmax(cv_scores_auc)
    optimal_k = k_values[optimal_k_idx]
    
    # Store results
    results['k_values'] = list(k_values)
    results['cv_scores_auc'] = cv_scores_auc
    results['cv_scores_f1'] = cv_scores_f1
    results['cv_scores_accuracy'] = cv_scores_accuracy
    results['optimal_k'] = optimal_k
    results['optimal_auc'] = cv_scores_auc[optimal_k_idx]
    results['optimal_f1'] = cv_scores_f1[optimal_k_idx]
    results['optimal_accuracy'] = cv_scores_accuracy[optimal_k_idx]
    
    print(f"Best k: {optimal_k} with AUC: {results['optimal_auc']:.4f}")
    
    return results

def plot_metrics_comparison(euclidean_results, cosine_results):
    """
    Plots metric comparison between Euclidean and Cosine distances
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 15))
    
    k_values = euclidean_results['k_values']
    
    # AUC
    axes[0].plot(k_values, euclidean_results['cv_scores_auc'], label='Euclidean', marker='o')
    axes[0].plot(k_values, cosine_results['cv_scores_auc'], label='Cosine', marker='s')
    axes[0].set_title('AUC Score by K Value')
    axes[0].set_xlabel('K Value')
    axes[0].set_ylabel('AUC Score')
    axes[0].legend()
    axes[0].grid(True)
    
    # F1-Score
    axes[1].plot(k_values, euclidean_results['cv_scores_f1'], label='Euclidean', marker='o')
    axes[1].plot(k_values, cosine_results['cv_scores_f1'], label='Cosine', marker='s')
    axes[1].set_title('F1-Score by K Value')
    axes[1].set_xlabel('K Value')
    axes[1].set_ylabel('F1-Score')
    axes[1].legend()
    axes[1].grid(True)
    
    # Accuracy
    axes[2].plot(k_values, euclidean_results['cv_scores_accuracy'], label='Euclidean', marker='o')
    axes[2].plot(k_values, cosine_results['cv_scores_accuracy'], label='Cosine', marker='s')
    axes[2].set_title('Accuracy by K Value')
    axes[2].set_xlabel('K Value')
    axes[2].set_ylabel('Accuracy')
    axes[2].legend()
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig('results/metric_comparison.png')
    plt.close()

def plot_roc_curves(embeddings, labels, euclidean_k, cosine_k):
    """
    Generates and plots ROC curves for both metrics
    """
    # Cross-validation to generate ROC curves - use just one fold to simplify
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    
    # Get a single fold to calculate ROC curves
    train_idx, val_idx = next(iter(skf.split(embeddings, labels)))
    X_train, X_val = embeddings[train_idx], embeddings[val_idx]
    y_train, y_val = labels[train_idx], labels[val_idx]
    
    # Convert labels to numeric format to use with label_binarize
    le = LabelEncoder()
    y_val_encoded = le.fit_transform(y_val)
    
    # Train and test with Euclidean distance
    euclidean_knn = KNeighborsClassifier(n_neighbors=euclidean_k, metric='euclidean')
    euclidean_knn.fit(X_train, y_train)
    euclidean_proba = euclidean_knn.predict_proba(X_val)
    
    # Train and test with Cosine distance
    cosine_knn = KNeighborsClassifier(n_neighbors=cosine_k, metric='cosine')
    cosine_knn.fit(X_train, y_train)
    cosine_proba = cosine_knn.predict_proba(X_val)
    
    # Convert labels to binary format for multiclass AUC calculation
    classes = le.fit_transform(np.unique(labels))
    y_val_binarized = np.zeros((len(y_val), len(classes)))
    for i, label in enumerate(y_val_encoded):
        y_val_binarized[i, label] = 1
    
    # Calculate ROC curves for each class then average
    # For Euclidean distance
    fpr_euclidean = dict()
    tpr_euclidean = dict()
    roc_auc_euclidean = dict()
    
    # For Cosine distance
    fpr_cosine = dict()
    tpr_cosine = dict()
    roc_auc_cosine = dict()
    
    # Calculate ROC for each class
    for i in range(len(classes)):
        if i < euclidean_proba.shape[1]:  # Check if class exists in probabilities
            fpr_euclidean[i], tpr_euclidean[i], _ = roc_curve(y_val_binarized[:, i], euclidean_proba[:, i])
            roc_auc_euclidean[i] = auc(fpr_euclidean[i], tpr_euclidean[i])
            
            fpr_cosine[i], tpr_cosine[i], _ = roc_curve(y_val_binarized[:, i], cosine_proba[:, i])
            roc_auc_cosine[i] = auc(fpr_cosine[i], tpr_cosine[i])
    
    # Calculate average ROC curves
    # First define a common FPR grid
    mean_fpr = np.linspace(0, 1, 100)
    
    # Interpolate TPR for each class and each metric
    all_tpr_euclidean = []
    all_tpr_cosine = []
    
    for i in range(len(classes)):
        if i in fpr_euclidean and i in fpr_cosine:  # Check if curves exist
            # Interpolate TPR for Euclidean distance
            interp_euclidean = interp1d(fpr_euclidean[i], tpr_euclidean[i], kind='linear', fill_value='extrapolate')
            tpr_interp_euclidean = interp_euclidean(mean_fpr)
            all_tpr_euclidean.append(tpr_interp_euclidean)
            
            # Interpolate TPR for Cosine distance
            interp_cosine = interp1d(fpr_cosine[i], tpr_cosine[i], kind='linear', fill_value='extrapolate')
            tpr_interp_cosine = interp_cosine(mean_fpr)
            all_tpr_cosine.append(tpr_interp_cosine)
    
    # Calculate averages
    if all_tpr_euclidean:
        mean_tpr_euclidean = np.mean(all_tpr_euclidean, axis=0)
        mean_tpr_euclidean[0] = 0.0  # Force start at (0,0)
        mean_auc_euclidean = auc(mean_fpr, mean_tpr_euclidean)
    else:
        mean_auc_euclidean = 0
    
    if all_tpr_cosine:
        mean_tpr_cosine = np.mean(all_tpr_cosine, axis=0)
        mean_tpr_cosine[0] = 0.0  # Force start at (0,0)
        mean_auc_cosine = auc(mean_fpr, mean_tpr_cosine)
    else:
        mean_auc_cosine = 0
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.plot(mean_fpr, mean_tpr_euclidean, label=f'Euclidean (AUC = {mean_auc_euclidean:.3f})', linewidth=2)
    plt.plot(mean_fpr, mean_tpr_cosine, label=f'Cosine (AUC = {mean_auc_cosine:.3f})', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', alpha=0.8)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Average ROC Curves - Metric Comparison')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.savefig('results/roc_curves_comparison.png')
    plt.close()  # Close to avoid display attempt
    
    print(f"\nAverage AUC - Euclidean: {mean_auc_euclidean:.4f}")
    print(f"Average AUC - Cosine: {mean_auc_cosine:.4f}")

def main():
    """
    Main function that executes the entire pipeline
    """
    print("Starting genetic syndrome classification project...")
    
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    # 1. Load data
    print("1. Loading data...")
    data = load_data('mini_gm_public_v0.1.p')
    
    # 2. Transform data
    print("2. Transforming data...")
    embeddings, labels, metadata = flatten_data(data)
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Number of labels: {len(labels)}")
    
    # 3. Exploratory analysis
    print("3. Performing exploratory data analysis...")
    exploratory_data_analysis(labels)
    
    # 4. Visualization with t-SNE
    print("4. Generating t-SNE visualization...")
    visualize_embeddings_tsne(embeddings, labels)
    
    # 5. KNN Classification with Euclidean distance
    print("5. Performing KNN classification with Euclidean distance...")
    euclidean_results = knn_classification(embeddings, labels, distance_metric='euclidean')
    
    # 6. KNN Classification with Cosine distance
    print("6. Performing KNN classification with Cosine distance...")
    cosine_results = knn_classification(embeddings, labels, distance_metric='cosine')
    
    # 7. Metric comparison
    print("7. Comparing metrics...")
    plot_metrics_comparison(euclidean_results, cosine_results)
    
    # 8. ROC curves
    print("8. Generating ROC curves...")
    plot_roc_curves(embeddings, labels, euclidean_results['optimal_k'], cosine_results['optimal_k'])
    
    # 9. Summary of results
    print("\n" + "="*50)
    print("RESULTS SUMMARY")
    print("="*50)
    print(f"Euclidean Distance - Best k: {euclidean_results['optimal_k']}, AUC: {euclidean_results['optimal_auc']:.4f}, F1: {euclidean_results['optimal_f1']:.4f}, Accuracy: {euclidean_results['optimal_accuracy']:.4f}")
    print(f"Cosine Distance - Best k: {cosine_results['optimal_k']}, AUC: {cosine_results['optimal_auc']:.4f}, F1: {cosine_results['optimal_f1']:.4f}, Accuracy: {cosine_results['optimal_accuracy']:.4f}")
    
    if euclidean_results['optimal_auc'] > cosine_results['optimal_auc']:
        print(f"\nThe Euclidean distance metric performed better with AUC of {euclidean_results['optimal_auc']:.4f}")
    else:
        print(f"\nThe Cosine distance metric performed better with AUC of {cosine_results['optimal_auc']:.4f}")

if __name__ == "__main__":
    main()