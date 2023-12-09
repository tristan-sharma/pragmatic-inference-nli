# Tests the BERT model with the pretrained weights on the MLI dataset against validation data from the MLI dataset.
from task2 import predict_relationship
from sklearn.metrics import accuracy_score
from datasets import load_dataset
import pandas as pd
from bert_nli import BertNLIModel

multi_nli_dataset = load_dataset("multi_nli")
val_data = pd.DataFrame(multi_nli_dataset['validation_matched'])
val_sample = val_data.sample(n=1000, random_state=1)
model = BertNLIModel('../models/bert-base.state_dict')

def predict_dataset(dataframe, predict_function):
    predictions = []
    for _, row in dataframe.iterrows():
        prediction = predict_function(model, row['premise'], row['hypothesis'])
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
