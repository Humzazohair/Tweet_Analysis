# main.py
# Simplified version for MP0 using global config values

import reader
import naive_bayes as nb
import csv

# === CONFIGURATION ===
TRAINING_DIR = 'data/WNUT-2020-Task-2-Dataset/train.tsv'
TEST_DIR = 'data/WNUT-2020-Task-2-Dataset/test.tsv'
VALIDATION_DIR = 'data/WNUT-2020-Task-2-Dataset/valid.tsv'
VALIDATION = True # set this if you want to use validation dataset instead of testing dataset
STEMMING = True
LOWERCASE = True
LAPLACE = 5
POS_PRIOR = 0.5
# ======================

def compute_accuracies(predicted_labels, test_labels):
    yhats = predicted_labels
    
    assert len(yhats) == len(test_labels), "predicted and gold label lists have different lengths"
    accuracy = sum([yhats[i] == test_labels[i] for i in range(len(yhats))]) / len(yhats)
    return accuracy

def print_value(label, value, numvalues):
    print(f"{label} {value} ({value/numvalues * 100:.2f}%)")

def print_stats(accuracy, numvalues):
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Total number of samples: {numvalues}")

def create_prediction_csv(predicted_labels, test_labels, test_id, output_file):  
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["Id", "Label"])
        writer.writeheader()
        for i in range(len(predicted_labels)):
            writer.writerow({"Id" : test_id[i], "Label" : predicted_labels[i]})

def main():
    train_labels, train_text, test_labels, test_text, test_id, validation_labels, validation_text, validation_id = nb.load_data(TRAINING_DIR, TEST_DIR, VALIDATION_DIR)

    if VALIDATION:
        predicted_labels = nb.naive_bayes(train_labels, train_text, validation_text, LAPLACE, POS_PRIOR)
        create_prediction_csv(predicted_labels, validation_labels, validation_id, "unigram_prediction.csv")
        accuracy = compute_accuracies(predicted_labels, validation_labels)
        print_stats(accuracy, len(validation_labels))
    else:
        predicted_labels = nb.naive_bayes(train_labels, train_text, test_text, LAPLACE, POS_PRIOR)
        create_prediction_csv(predicted_labels, test_labels, test_id, "unigram_prediction.csv")
        accuracy = compute_accuracies(predicted_labels, test_labels)
        print_stats(accuracy, len(test_labels))

if __name__ == "__main__":
    main()
