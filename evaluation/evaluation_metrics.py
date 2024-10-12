from sklearn.metrics import f1_score, classification_report, accuracy_score
import pandas as pd
import matplotlib.pyplot as plt


def evaluate_model(y_true, model_predictions, model_name):
    """
    Evaluate the model with accuracy, classification report, and F1 scores.

    Parameters:
    - y_true: Ground truth labels.
    - model_predictions: Predictions from the model.
    - model_name: Name of the model (for reporting purposes).
    """
    # Calculate accuracy
    accuracy = accuracy_score(y_true, model_predictions)
    print(f"=== {model_name} ===")
    print(f"Cross-validated Accuracy: {accuracy:.4f}")

    # Print classification report
    print(f"=== Classification Report ({model_name}) ===")
    print(classification_report(y_true, model_predictions))

    return accuracy


def plot_f1_scores(y_true, model_predictions, class_labels=['n', 'y']):
    """
    Plot the average F1 scores for multiple models using cross-validation predictions.

    Parameters:
    - y_true: Ground truth labels
    - model_predictions: A dictionary where keys are model names (strings) and values are cross-validation predictions (arrays)
    - class_labels: The class labels (default is ['n', 'y'])
    """
    # Initialize a DataFrame to hold F1 scores for each model and class
    f1_results = {}

    for model_name, predictions in model_predictions.items():
        f1_scores = [f1_score(y_true, predictions, pos_label=class_label, average='binary') for class_label in
                     class_labels]
        f1_results[model_name] = f1_scores

    # Convert the results to a DataFrame for easier plotting
    df_res = pd.DataFrame(f1_results, index=class_labels)

    # Plot the F1 scores as a bar plot
    ax = df_res.plot.bar()
    ax.set_xlabel("Class")
    ax.set_ylabel("F1 Score")
    plt.title("F1 Scores by Model and Class")
    plt.show()
