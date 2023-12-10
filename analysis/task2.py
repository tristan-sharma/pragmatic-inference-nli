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

def plot_accuracy(dataset, name, dataset_title):
    accuracies_df = get_relationship_accuracies(dataset)
    accuracies_df = accuracies_df.T  # Transpose the DataFrame for correct plotting
    accuracies_df.reset_index(inplace=True)
    accuracies_df.columns = ['Condition', 'Accuracy']
    
    ax = accuracies_df.plot(x='Condition', y='Accuracy', kind='bar', title=f'Model Accuracy Comparison - {dataset_title}')
    plt.xlabel('Condition')
    plt.ylabel('Accuracy')
    ax.get_legend().remove()  # Remove legend if not needed
    plt.xticks(rotation=0)  # Set x-ticks rotation
    plt.tight_layout()
    plt.savefig(f'./task2plots/{name}_model_accuracy_comparison.png')
    plt.close()

def plot_confusion_matrix(dataset, name, prediction_type, dataset_title): 
    cm = confusion_matrix(dataset['Relationship Type'], dataset[prediction_type])
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    class_names = ['Entailment', 'Neutral', 'Contradiction']
    sns.heatmap(cm, annot=True, fmt='.2f', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - {prediction_type} - {dataset_title}')
    plt.ylabel('Actual', fontweight='bold')
    plt.xlabel('Predicted', fontweight='bold')
    plt.savefig(f'./task2plots/{name}_confusion_matrix_{prediction_type}.png')
    plt.close()

def print_classification_report(dataset, prediction_type, name, dataset_title):
    report = classification_report(dataset['Relationship Type'], dataset[prediction_type], target_names=['Entailment', 'Neutral', 'Contradiction'])
    file_path = f'./task2plots/{name}_classification_report_{prediction_type}.txt'
    with open(file_path, 'w') as file:
        file.write(f"Classification Report - {dataset_title}\n")
        file.write(report)
    print(f"Classification report for {dataset_title} saved to {file_path}")

def performance_by_relationship_type(dataset, name, dataset_title):
    results = []
    relationship_types = dataset['Relationship Type'].unique()
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
    df_results = pd.DataFrame(results)
    ax = df_results.plot(x='Relationship Type', kind='bar', stacked=True, figsize=(10, 6))
    plt.title(f'Performance by Relationship Type - {dataset_title}')
    plt.ylabel('Accuracy')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=10)  # Rotate labels and set smaller font size
    plt.tight_layout()  # Adjust layout to fit everything nicely
    plt.savefig(f'./task2plots/{name}_performance_by_relationship_type.png')
    plt.close()
    return df_results

def plot_accuracy_by_presupposition_type(dataset, name, dataset_title):
    analysis = analyze_presupposition_types(dataset)
    ax = analysis.plot(x='Presupposition Type', kind='bar', stacked=True, figsize=(10, 6))
    plt.ylabel('Accuracy')
    plt.title(f'Accuracy by Presupposition Type - {dataset_title}')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=10)  # Rotate labels and set smaller font size
    plt.tight_layout()  # Adjust layout to fit everything nicely
    plt.savefig(f'./task2plots/{name}_accuracy_by_presupposition_type.png')
    plt.close()
    return analysis

def process_dataset(dataset_name, model, dataset_title):
    # Load dataset
    dataset = pd.read_excel(f'../data/{dataset_name}.xlsx', header=0)

    # Check for pre-existing prediction columns
    if 'Baseline Prediction' not in dataset.columns:
        print(f"Adding prediction columns to {dataset_name}...")
        dataset = add_prediction_columns(dataset, model)
        dataset.to_excel(f'../data/{dataset_name}.xlsx', index=False)

    # Perform analyses
    print(f"\n{dataset_title} dataset accuracies:")
    get_relationship_accuracies(dataset)

    print(f"\n{dataset_title} dataset presupposition analysis:")
    presupposition_analysis = analyze_presupposition_types(dataset)
    print(presupposition_analysis)

    # Plotting
    plot_accuracy(dataset, dataset_name, dataset_title)
    plot_confusion_matrix(dataset, dataset_name, 'Baseline Prediction', dataset_title)
    plot_confusion_matrix(dataset, dataset_name, 'Critical Prediction', dataset_title)
    plot_confusion_matrix(dataset, dataset_name, 'Non-Critical Prediction', dataset_title)
    performance_by_relationship_type(dataset, dataset_name, dataset_title)
    plot_accuracy_by_presupposition_type(dataset, dataset_name, dataset_title)

    print(f"Classification report for {dataset_title} dataset:")
    print_classification_report(dataset, 'Baseline Prediction', dataset_name, dataset_title)
    print_classification_report(dataset, 'Critical Prediction', dataset_name, dataset_title)
    print_classification_report(dataset, 'Non-Critical Prediction', dataset_name, dataset_title)

def main():
    # Initialize the BERT NLI model
    model = BertNLIModel('../models/bert-base.state_dict')

    # Process each dataset
    process_dataset('IMPPRESinspired-nli-dataset', model, 'IMPPRES Set')
    process_dataset('MultiNLIinspired-nli-dataset-trainsplit', model, 'MultiNLI Train Split Set')
    process_dataset('MultiNLIinspired-nli-dataset-validation-split', model, 'MultiNLI Validation Split Set')

if __name__ == "__main__":
    main()