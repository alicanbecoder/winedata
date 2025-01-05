import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import LabelBinarizer
import os

# Initialize lists to store algorithm names, accuracies, and model details  
al = []
accuracy = []
models = []

def evaluate_model(algorithm_name, model, X_train, X_test, y_train, y_test, train_columns, folder_name, wine_type, models):
    # Start time tracking
    start_time = time.time()

    # Train the model and make predictions
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)

    # Calculate accuracy scores
    acc_model = model.score(X_test, y_test)
    train_acc_model = model.score(X_train, y_train)

    # Store the algorithm name and accuracy
    al.append(algorithm_name)
    accuracy.append(acc_model)

    # End time tracking
    end_time = time.time()
    perf_time = end_time - start_time  # Time in seconds

    # Print performance metrics
    print(f'For {algorithm_name} - {wine_type} Wine\n')
    print(f'Training Accuracy: {train_acc_model * 100:.4f} %\n')
    print(f'Testing Accuracy: {acc_model * 100:.4f} %\n')
    print(f'Accuracy Score: {accuracy_score(y_test, y_pred) * 100:.4f} %\n')
    print(f'Performance Time: {perf_time:.4f} seconds\n')

    # Create subplots for the confusion matrix and ROC curve
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap='RdGy', annot_kws={'size': 15}, square=True, fmt='.0f', ax=axes[0])
    axes[0].set_title('Confusion Matrix', fontsize=20)

    lb = LabelBinarizer()
    y_test_bin = lb.fit_transform(y_test)  # Binarize the test labels for multiclass
    for i in range(y_test_bin.shape[1]):  # loop through each class
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_prob[:, i])
        roc = auc(fpr, tpr)
        axes[1].plot(fpr, tpr, label=f'Class {i+1} (AUC = {roc:.2f})', color='red')

    axes[1].set_xlabel('False Positive Rate', fontsize=15)
    axes[1].set_ylabel('True Positive Rate', fontsize=15)
    axes[1].set_title(f'ROC Curves for Multiclass', fontsize=20)
    axes[1].legend()

    # Save the confusion matrix and ROC plot to PNG files
    plot_folder = os.path.join(folder_name, f'{wine_type}_{algorithm_name}')
    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)

    confusion_matrix_path = os.path.join(plot_folder, f'{wine_type}_{algorithm_name}_confusion_matrix.png')
    roc_curve_path = os.path.join(plot_folder, f'{wine_type}_{algorithm_name}_roc_curve.png')
    
    plt.tight_layout()
    fig.savefig(confusion_matrix_path, format='png')
    fig.savefig(roc_curve_path, format='png')
    print(f'Plots saved to {plot_folder}\n')

    # Plot permutation importance for feature evaluation
    if algorithm_name != 'K Nearest Neighbors':
        perm_importance = permutation_importance(model, X_test, y_test, n_repeats=30, random_state=42)
        feature_importance = perm_importance.importances_mean
        feature_names = train_columns

        plt.figure(figsize=(12, 6))
        plt.barh(feature_names, feature_importance, color='#b40000')
        plt.xlabel('Permutation Importance')
        plt.ylabel('Feature')
        plt.title(f'{algorithm_name} Permutation Importance for Features')

        perm_importance_path = os.path.join(plot_folder, f'{wine_type}_{algorithm_name}_perm_importance.png')
        plt.savefig(perm_importance_path, format='png')
        plt.close()
        print(f'Permutation Importance Plot saved to {perm_importance_path}\n')

    # Append the model evaluation results to the models list
    models.append((algorithm_name, train_acc_model, acc_model, accuracy_score(y_test, y_pred), perf_time)) 

    return models 


