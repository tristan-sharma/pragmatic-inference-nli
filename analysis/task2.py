from bert_nli import BertNLIModel
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from datasets import load_dataset
import numpy as np

def predict_relationship(model, premise, hypothesis):
    '''
    Returns the predicted relationship between the premise and hypothesis.
    Value is one of: ['entailment', 'neutral', 'contradiction']
    '''
    sent_pairs = [(premise, hypothesis)]
    label, _= model(sent_pairs)
    if label[0] == 'entail':
        return 'entailment'
    return label[0]

def add_prediction_columns(dataset, model):
    """
    Adds prediction columns to the dataset for baseline, critical, and non-critical scenarios.
    Args:
        dataset (pd.DataFrame): The dataset containing premise, hypothesis, and other relevant columns.
        model (BertNLIModel): The pre-trained BERT model for NLI.
    
    Returns:
        pd.DataFrame: The updated dataset with new prediction columns.
    """
    # Initialize lists to store predictions
    baseline_predictions = []
    critical_predictions = []
    noncritical_predictions = []

    for _, row in dataset.iterrows():
        # Extracting data from the row
        premise = row['Premise']
        hypothesis = row['Hypothesis']
        masked_premise_critical = row['Masked Premise (Critical)']
        masked_premise_noncritical = row['Masked Premise (Non-Critical)']

        # Making predictions
        baseline_pred = predict_relationship(model, premise, hypothesis)
        critical_pred = predict_relationship(model, masked_premise_critical, hypothesis)
        noncritical_pred = predict_relationship(model, masked_premise_noncritical, hypothesis)

        # Storing predictions
        baseline_predictions.append(baseline_pred)
        critical_predictions.append(critical_pred)
        noncritical_predictions.append(noncritical_pred)

    # Adding predictions to the dataset
    dataset['Baseline Prediction'] = baseline_predictions
    dataset['Critical Prediction'] = critical_predictions
    dataset['Non-Critical Prediction'] = noncritical_predictions

    dataset['Relationship Type'] = dataset['Relationship Type'].str.lower()

    return dataset

def get_relationship_accuracies(dataset):
    actual_relationships = dataset['Relationship Type']
    baseline_relationships = dataset['Baseline Prediction']
    critical_relationships = dataset['Critical Prediction']
    noncritical_relationships = dataset['Non-Critical Prediction']

    print("Baseline accuracy: {}".format(accuracy_score(actual_relationships, baseline_relationships)))
    print("Critical accuracy: {}".format(accuracy_score(actual_relationships, critical_relationships)))
    print("Non-critical accuracy: {}".format(accuracy_score(actual_relationships, noncritical_relationships)))

    results = []
    results.append({
        'Baseline': accuracy_score(actual_relationships, baseline_relationships),
        'Critical': accuracy_score(actual_relationships, critical_relationships),
        'Non-Critical': accuracy_score(actual_relationships, noncritical_relationships)
    })
    return pd.DataFrame(results)

def analyze_presupposition_types(dataset):
    presupposition_types = dataset['Presupposition Type'].unique()
    results = []

    for presupposition_type in presupposition_types:
        filtered_dataset = dataset[dataset['Presupposition Type'] == presupposition_type]

        # Use precomputed predictions
        baseline_accuracy = accuracy_score(filtered_dataset['Relationship Type'], filtered_dataset['Baseline Prediction'])
        critical_accuracy = accuracy_score(filtered_dataset['Relationship Type'], filtered_dataset['Critical Prediction'])
        noncritical_accuracy = accuracy_score(filtered_dataset['Relationship Type'], filtered_dataset['Non-Critical Prediction'])

        results.append({
            'Presupposition Type': presupposition_type,
            'Baseline Accuracy': baseline_accuracy,
            'Critical Accuracy': critical_accuracy,
            'Non-Critical Accuracy': noncritical_accuracy
        })

    return pd.DataFrame(results)

def plot_accuracy(dataset, name):
    accuracies = get_relationship_accuracies(dataset)
    accuracies.plot(kind='bar', title='Model Accuracy Comparison')
    plt.xlabel('Condition')
    plt.ylabel('Accuracy')
    plt.savefig(f'./task2plots/{name}model_accuracy_comparison.png')
    plt.close()

def plot_confusion_matrix(dataset, name, prediction_type): 
    cm = confusion_matrix(dataset['Relationship Type'], dataset[prediction_type])
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    # Define the class names
    class_names = ['Entailment', 'Neutral', 'Contradiction']
    sns.heatmap(cm, annot=True, fmt='.2f', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - {prediction_type}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(f'./task2plots/{name}confusion_matrix_{prediction_type}.png')
    plt.close()

def print_classification_report(dataset, prediction_type):
    report = classification_report(dataset['Relationship Type'], dataset[prediction_type], target_names=['Entailment', 'Neutral', 'Contradiction'])
    print(report)

def performance_by_relationship_type(dataset):
    relationship_types = dataset['Relationship Type'].unique()
    results = []

    for rel_type in relationship_types:
        filtered_dataset = dataset[dataset['Relationship Type'] == rel_type]
        baseline_accuracy = accuracy_score(filtered_dataset['Relationship Type'], filtered_dataset['Baseline Prediction'])
        critical_accuracy = accuracy_score(filtered_dataset['Relationship Type'], filtered_dataset['Critical Prediction'])
        noncritical_accuracy = accuracy_score(filtered_dataset['Relationship Type'], filtered_dataset['Non-Critical Prediction'])

        results.append({
            'Relationship Type': rel_type,
            'Baseline Accuracy': baseline_accuracy,
            'Critical Accuracy': critical_accuracy,
            'Non-Critical Accuracy': noncritical_accuracy
        })

    return pd.DataFrame(results)

def plot_accuracy_by_presupposition_type(dataset, file_name='accuracy_by_presupposition_type.png'):
    analysis = analyze_presupposition_types(dataset)
    analysis.plot(x='Presupposition Type', kind='bar', stacked=True)
    plt.ylabel('Accuracy')
    plt.title('Accuracy by Presupposition Type')
    plt.savefig(file_name)
    plt.close()

def main():
    multinli_inspired_dataset = pd.read_excel('../data/MultiNLIinspired-nli-dataset.xlsx', header=0)
    imppres_inspired_dataset = pd.read_excel('../data/IMPPRESinspired-nli-dataset.xlsx', header=0)

    # if the prediction columns already exist, skip this step
    if 'Baseline Prediction' not in multinli_inspired_dataset.columns:
        print("Adding prediction columns to the datasets...")
        model = BertNLIModel('../models/bert-base.state_dict')
        imppres_inspired_dataset = add_prediction_columns(imppres_inspired_dataset, model)
        multinli_inspired_dataset = add_prediction_columns(multinli_inspired_dataset, model)
        # Replace the data files with the updated ones
        imppres_inspired_dataset.to_excel('../data/IMPPRESinspired-nli-dataset.xlsx', index=False)
        multinli_inspired_dataset.to_excel('../data/MultiNLIinspired-nli-dataset.xlsx', index=False)

    print("IMPPRES-inspired dataset accuracies:")
    get_relationship_accuracies(imppres_inspired_dataset)
    print("\nMultiNLI-inspired dataset accuracies:")
    get_relationship_accuracies(multinli_inspired_dataset)

    print("\nIMPPRES-inspired dataset presupposition analysis:")
    imppres_presupposition_analysis = analyze_presupposition_types(imppres_inspired_dataset)
    print(imppres_presupposition_analysis)
    print("\nMultiNLI-inspired dataset presupposition analysis:")
    multinli_presupposition_analysis = analyze_presupposition_types(multinli_inspired_dataset)
    print(multinli_presupposition_analysis)

    plot_accuracy(imppres_inspired_dataset, 'IMPPRES-inspired ')
    plot_accuracy(multinli_inspired_dataset, 'MultiNLI-inspired ')
    plot_confusion_matrix(imppres_inspired_dataset, 'IMPPRES-inspired ', 'Baseline Prediction',)
    plot_confusion_matrix(imppres_inspired_dataset, 'IMPPRES-inspired ', 'Critical Prediction')
    plot_confusion_matrix(imppres_inspired_dataset, 'IMPPRES-inspired ', 'Non-Critical Prediction')

    print("Classification report for IMPPRES-inspired dataset:")
    print_classification_report(imppres_inspired_dataset, 'Baseline Prediction')

    print("Classification report for MultiNLI-inspired dataset:")
    print_classification_report(multinli_inspired_dataset, 'Baseline Prediction')

if __name__ == "__main__":
    main()
