from bert_nli import BertNLIModel
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.metrics import confusion_matrix, accuracy_score
from datasets import load_dataset

dataset = pd.read_excel('../data/custom-nli-dataset.xlsx', header=0)
model = BertNLIModel('../models/bert-base.state_dict')

def predict_relationship(premise, hypothesis):
    '''
    Returns the predicted relationship between the premise and hypothesis.
    Value is one of: ['entailment', 'neutral', 'contradiction']
    '''
    sent_pairs = [(premise, hypothesis)]
    label, _= model(sent_pairs)
    if label[0] == 'entail':
        return 'entailment'
    return label[0]

multi_nli_dataset = load_dataset("multi_nli")
val_data = pd.DataFrame(multi_nli_dataset['validation_matched'])
val_sample = val_data.sample(n=100, random_state=1)

from sklearn.metrics import accuracy_score

def predict_dataset(dataframe, predict_function):
    predictions = []
    for index, row in dataframe.iterrows():
        prediction = predict_function(row['premise'], row['hypothesis'])
        predictions.append(prediction)
    return predictions

int_to_label = {
    0: 'entailment',
    1: 'neutral',
    2: 'contradiction'
}

predicted_labels = predict_dataset(val_sample, predict_relationship)
adjusted_predicted_labels = [int_to_label[label] for label in val_sample['label']]
print(adjusted_predicted_labels)
print(predicted_labels)
# Calculate the accuracy
accuracy = accuracy_score(adjusted_predicted_labels, predicted_labels)
print(f'Accuracy: {accuracy * 100:.2f}%')



# print(predict_relationship('Did Jo eat something?', 'Jo\'s cat went.'))


# actual_relationships = []
# baseline_relationships = []
# critical_relationships = []
# noncritical_relationships = []

# for index, row in dataset.iterrows():
#     original_premise = row['Premise']
#     masked_premise_critical = row['Masked Premise (Critical)']
#     masked_premise_noncritical = row['Masked Premise (Non-Critical)']
#     hypothesis = row['Hypothesis']
#     presupposition_type = row['Presupposition Type']
#     relationship_type = 'entail' if row['Relationship Type'] == 'entailment' else row['Relationship Type'].lower()
#     masked_word_critical = row['Masked Word (Critical)']
#     masked_word_noncritical = row['Masked Word (Non-Critical)']

#     baseline_relationship = predict_relationship(original_premise, hypothesis) # Baseline prediction without any masking
#     critical_relationship = predict_relationship(masked_premise_critical, hypothesis) # Prediction with critical masking
#     noncritical_relationship = predict_relationship(masked_premise_noncritical, hypothesis) # Prediction with non-critical masking

#     actual_relationships.append(relationship_type)
#     baseline_relationships.append(baseline_relationship)
#     critical_relationships.append(critical_relationship)
#     noncritical_relationships.append(noncritical_relationship)

# def calculate_proportion(list1, list2):
#     if len(list1) != len(list2):
#         raise ValueError("Lists must be of the same length")

#     same_count = sum(1 for x, y in zip(list1, list2) if x == y)
#     proportion = same_count / len(list1)
#     return proportion

# print("Baseline accuracy: {}".format(accuracy_score(actual_relationships, baseline_relationships)))
# print("Critical accuracy: {}".format(accuracy_score(actual_relationships, critical_relationships)))
# print("Non-critical accuracy: {}".format(accuracy_score(actual_relationships, noncritical_relationships)))

# print("Baseline proportion: {}".format(calculate_proportion(actual_relationships, baseline_relationships)))
# print("Critical proportion: {}".format(calculate_proportion(actual_relationships, critical_relationships)))
# print("Non-critical proportion: {}".format(calculate_proportion(actual_relationships, noncritical_relationships)))

