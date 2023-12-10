import pandas as pd
from transformers import BertTokenizer, TFBertForMaskedLM, BertForMaskedLM
import torch
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Function to get top k predictions for a masked word using BERT
def get_top_k_predictions(input_string, k=1, tokenizer=BertTokenizer.from_pretrained('bert-base-cased'), model=TFBertForMaskedLM.from_pretrained('bert-base-cased')):
    tokenized_inputs = tokenizer(input_string, return_tensors="tf")
    outputs = model(tokenized_inputs["input_ids"])
    mask_token_index = np.where(tokenized_inputs['input_ids'].numpy()[0] == tokenizer.mask_token_id)[0][0]
    mask_token_logits = outputs.logits[0, mask_token_index, :]
    top_k_indices = tf.math.top_k(mask_token_logits, k).indices.numpy()
    top_k_words = [tokenizer.decode([idx]) for idx in top_k_indices]

    return ', '.join(top_k_words)

# Function to concatenate premise and hypothesis with relationship indicator
def concatenate_with_indicator(premise, hypothesis, relationship_type):
    relationship_type_to_indicator = {
        "entailment": "This implies that",
        "neutral": "Additionally,",
        "contradiction": "On the contrary,"
    }
    indicator = relationship_type_to_indicator.get(relationship_type.lower(), "")
    # Adding [CLS] at the start and [SEP] at the end
    return f"[CLS] {premise} [SEP] {indicator} {hypothesis} [SEP]"

def concatenate_without_indicator(premise, hypothesis):
    # Adding [CLS] at the start and [SEP] at the end
    return f"[CLS] {premise} [SEP] {hypothesis} [SEP]"

# Function to add prediction columns to the dataset
def add_prediction_columns(dataset, model, tokenizer):
    # Columns to be added
    columns = [
        'Critical Masked, Indicator', 'Non-Critical Masked, Indicator',
        'Critical Masked, No Indicator', 'Non-Critical Masked, No Indicator'
    ]

    # Check if the columns already exist
    if 'Critical Masked, Indicator' in dataset.columns:
        print("Prediction columns already exist. Skipping predictions.")
        return dataset

    # Initialize new columns
    for col in columns:
        dataset[col] = None

    # Iterate over the dataset and make predictions
    for index, row in tqdm(dataset.iterrows()):
        premise_critical = row['Masked Premise (Critical)']
        premise_non_critical = row['Masked Premise (Non-Critical)']
        hypothesis = row['Hypothesis']
        relationship_type = row['Relationship Type']

        # Store predictions with indicators
        concatenated_sentence_critical = concatenate_with_indicator(premise_critical, hypothesis, relationship_type)
        concatenated_sentence_non_critical = concatenate_with_indicator(premise_non_critical, hypothesis, relationship_type)
        dataset.at[index, 'Critical Masked, Indicator'] = get_top_k_predictions(concatenated_sentence_critical, tokenizer=tokenizer, model=model)
        dataset.at[index, 'Non-Critical Masked, Indicator'] = get_top_k_predictions(concatenated_sentence_non_critical, tokenizer=tokenizer, model=model)

        # Store predictions without indicators
        concatenated_sentence_critical_no_ind = concatenate_without_indicator(premise_critical, hypothesis)
        concatenated_sentence_non_critical_no_ind = concatenate_without_indicator(premise_non_critical, hypothesis)
        dataset.at[index, 'Critical Masked, No Indicator'] = get_top_k_predictions(concatenated_sentence_critical_no_ind, tokenizer=tokenizer, model=model)
        dataset.at[index, 'Non-Critical Masked, No Indicator'] = get_top_k_predictions(concatenated_sentence_non_critical_no_ind, tokenizer=tokenizer, model=model)

    return dataset

def calculate_accuracy_for_masked_words(dataset):
    # Define the conditions to be checked
    conditions = [
        'Critical Masked, Indicator',
        'Non-Critical Masked, Indicator',
        'Critical Masked, No Indicator',
        'Non-Critical Masked, No Indicator'
    ]

    condition_to_word_column = {
        'Critical Masked, Indicator': 'Masked Word (Critical)',
        'Non-Critical Masked, Indicator': 'Masked Word (Non-Critical)',
        'Critical Masked, No Indicator': 'Masked Word (Critical)',
        'Non-Critical Masked, No Indicator': 'Masked Word (Non-Critical)'
    }
    
    # Dictionary to hold the count of correct predictions for each condition
    correct_counts = {condition: 0 for condition in conditions}
    total_counts = {condition: len(dataset) for condition in conditions}
    # Iterate over the dataset and count correct predictions
    for _, row in dataset.iterrows():
        for condition in conditions:
            # Split the predictions into a list
            predictions = row[condition].split(', ')
            word_column = condition_to_word_column[condition]
            # Check if the actual masked word is in the top 10 predictions
            if row[word_column] in predictions or row[word_column].lower() in predictions:
                correct_counts[condition] += 1
    
    # Calculate accuracy for each condition
    accuracies = {condition: correct_counts[condition] / total_counts[condition] for condition in conditions}
    
    # Convert the accuracies dictionary to a DataFrame
    accuracies_df = pd.DataFrame(accuracies.items(), columns=['Condition', 'Accuracy'])
    
    return accuracies_df

def plot_prediction_accuracy(accuracies_df, dataset_name, file_path, dataset_title):
    """
    Plots and saves the accuracies of word predictions for different conditions in the dataset.

    Args:
    accuracies_df (pd.DataFrame): DataFrame containing accuracies for different conditions.
    dataset_name (str): The name of the dataset for which the plot is being made.
    file_path (str): The file path to save the plot.
    """
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Condition', y='Accuracy', data=accuracies_df)
    plt.title(f'Word Prediction Accuracy - {dataset_title}')
    plt.xlabel('Condition')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{file_path}/{dataset_name}_accuracies")  # Save the plot to the specified file path
    plt.close()  # Close the plot to prevent displaying it

def calculate_accuracy_by_relationship_type(dataset):
    conditions = [
        'Critical Masked, Indicator',
        'Non-Critical Masked, Indicator',
        'Critical Masked, No Indicator',
        'Non-Critical Masked, No Indicator'
    ]

    condition_to_masked_word_column = {
        'Critical Masked, Indicator': 'Masked Word (Critical)',
        'Non-Critical Masked, Indicator': 'Masked Word (Non-Critical)',
        'Critical Masked, No Indicator': 'Masked Word (Critical)',
        'Non-Critical Masked, No Indicator': 'Masked Word (Non-Critical)'
    }

    # Initialize dictionary to hold accuracies for each condition and relationship type
    accuracies = {condition: {} for condition in conditions}

    # Group dataset by relationship type
    grouped_dataset = dataset.groupby('Relationship Type')

    for relationship_type, group_data in grouped_dataset:
        for condition in conditions:
            correct_predictions = group_data.apply(
                lambda row: row[condition_to_masked_word_column[condition]] in row[condition].split(', '),
                axis=1
            ).sum()
            accuracy = correct_predictions / len(group_data)
            accuracies[condition][relationship_type] = accuracy

    return accuracies

def plot_accuracy_by_relationship_type(accuracies, dataset_name, file_path, dataset_title):
    # Convert dictionary to DataFrame for plotting
    df = pd.DataFrame(accuracies).reset_index()
    df.rename(columns={'index': 'Relationship Type'}, inplace=True)
    df_melted = df.melt(id_vars=['Relationship Type'], var_name='Condition', value_name='Accuracy')

    plt.figure(figsize=(12, 6))
    sns.barplot(x='Condition', y='Accuracy', hue='Relationship Type', data=df_melted)
    plt.title(f'Prediction Accuracy by Relationship Type - {dataset_title}')
    plt.xlabel('Condition')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    plt.legend(title='Relationship Type')
    plt.tight_layout()
    plt.savefig(f"{file_path}/{dataset_name}_accuracy_by_relationship_type.png")
    plt.close()

def analyze_accuracy_by_presupposition_type(dataset):
    conditions = [
        'Critical Masked, Indicator',
        'Non-Critical Masked, Indicator',
        'Critical Masked, No Indicator',
        'Non-Critical Masked, No Indicator'
    ]

    condition_to_masked_word_column = {
        'Critical Masked, Indicator': 'Masked Word (Critical)',
        'Non-Critical Masked, Indicator': 'Masked Word (Non-Critical)',
        'Critical Masked, No Indicator': 'Masked Word (Critical)',
        'Non-Critical Masked, No Indicator': 'Masked Word (Non-Critical)'
    }

    results = []

    # Group dataset by presupposition type
    grouped_dataset = dataset.groupby('Presupposition Type')

    for presupposition_type, group_data in grouped_dataset:
        accuracies = {}
        for condition in conditions:
            correct_predictions = group_data.apply(
                lambda row: row[condition_to_masked_word_column[condition]] in row[condition].split(', '),
                axis=1
            ).sum()
            accuracy = correct_predictions / len(group_data)
            accuracies[condition] = accuracy

        results.append({
            'Presupposition Type': presupposition_type,
            **accuracies
        })

    return pd.DataFrame(results)

def plot_accuracy_by_presupposition_type(analysis_df, dataset_name, file_path, dataset_title):
    plt.figure(figsize=(12, 6))
    melted_df = analysis_df.melt(id_vars=['Presupposition Type'], var_name='Condition', value_name='Accuracy')
    sns.barplot(x='Presupposition Type', y='Accuracy', hue='Condition', data=melted_df)
    plt.title(f'Accuracy by Presupposition Type - {dataset_title}')
    plt.xlabel('Presupposition Type')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    plt.legend(title='Condition')
    plt.tight_layout()
    plt.savefig(f"{file_path}/{dataset_name}_accuracy_by_presupposition_type.png")
    plt.close()

# Main function to process the dataset
def process_dataset(dataset_name, model, tokenizer, dataset_title):
    dataset = pd.read_excel(f'../data/{dataset_name}.xlsx', header=0)
    dataset_with_predictions = add_prediction_columns(dataset, model, tokenizer)

    # Calculate accuracy for masked words
    accuracies_df = calculate_accuracy_for_masked_words(dataset_with_predictions)
    print(accuracies_df)

    plot_prediction_accuracy(accuracies_df, dataset_name, './task1plots', dataset_title)

    plot_accuracy_by_relationship_type(calculate_accuracy_by_relationship_type(dataset_with_predictions), dataset_name, './task1plots', dataset_title)

    presupposition_analysis = analyze_accuracy_by_presupposition_type(dataset_with_predictions)
    plot_accuracy_by_presupposition_type(presupposition_analysis, dataset_name, './task1plots', dataset_title)

    return dataset_with_predictions

def manual_test(sentence, tokenizer, model):
    """
    Perform a manual test by providing a sentence with a masked word.
    
    Args:
    sentence (str): The sentence with a '[MASK]' token where the word should be predicted.
    tokenizer (BertTokenizer): The tokenizer for BERT.
    model (TFBertForMaskedLM): The BERT model for masked language modeling.

    Returns:
    str: A string of top k predictions.
    """
    predictions = get_top_k_predictions(sentence, tokenizer=tokenizer, model=model)
    return predictions

def main():
    # Initialize the BERT model and tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    model = TFBertForMaskedLM.from_pretrained('bert-base-cased')

    processed_dataset_imppres = process_dataset('IMPPRESinspired-nli-dataset', model, tokenizer, 'IMPPRES Set')
    processed_dataset_imppres.to_excel(f'../data/IMPPRESinspired-nli-dataset.xlsx', index=False)

    processed_dataset_nli_train = process_dataset('MultiNLIinspired-nli-dataset-trainsplit', model, tokenizer, 'MultiNLI Train Split Set')
    processed_dataset_nli_train.to_excel(f'../data/MultiNLIinspired-nli-dataset-trainsplit.xlsx', index=False)

    processed_dataset_nli_val = process_dataset('MultiNLIinspired-nli-dataset-validation-split', model, tokenizer, 'MultiNLI Validation Split Set')
    processed_dataset_nli_val.to_excel(f'../data/MultiNLIinspired-nli-dataset-validation-split.xlsx', index=False)

# Example usage
if __name__ == "__main__":
    main()
