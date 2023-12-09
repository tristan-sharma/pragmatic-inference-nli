from bert_nli import BertNLIModel
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.metrics import confusion_matrix, accuracy_score
from datasets import load_dataset

multinli_inspired_dataset = pd.read_excel('../data/MultiNLIinspired-nli-dataset.xlsx', header=0)
imppres_inspired_dataset = pd.read_excel('../data/IMPPRESinspired-nli-dataset.xlsx', header=0)
model = BertNLIModel('../models/bert-base.state_dict')

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

def get_relationship_accuracies(dataset):

    actual_relationships = []
    baseline_relationships = []
    critical_relationships = []
    noncritical_relationships = []

    for index, row in dataset.iterrows():
        original_premise = row['Premise']
        masked_premise_critical = row['Masked Premise (Critical)']
        masked_premise_noncritical = row['Masked Premise (Non-Critical)']
        hypothesis = row['Hypothesis']
        presupposition_type = row['Presupposition Type']
        relationship_type = row['Relationship Type'].lower()
        masked_word_critical = row['Masked Word (Critical)']
        masked_word_noncritical = row['Masked Word (Non-Critical)']

        baseline_relationship = predict_relationship(model, original_premise, hypothesis) # Baseline prediction without any masking
        critical_relationship = predict_relationship(model, masked_premise_critical, hypothesis) # Prediction with critical masking
        noncritical_relationship = predict_relationship(model, masked_premise_noncritical, hypothesis) # Prediction with non-critical masking

        actual_relationships.append(relationship_type)
        baseline_relationships.append(baseline_relationship)
        critical_relationships.append(critical_relationship)
        noncritical_relationships.append(noncritical_relationship)

    print("Baseline accuracy: {}".format(accuracy_score(actual_relationships, baseline_relationships)))
    print("Critical accuracy: {}".format(accuracy_score(actual_relationships, critical_relationships)))
    print("Non-critical accuracy: {}".format(accuracy_score(actual_relationships, noncritical_relationships)))

    return actual_relationships, baseline_relationships, critical_relationships, noncritical_relationships

get_relationship_accuracies(imppres_inspired_dataset)