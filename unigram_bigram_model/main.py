import reader
import bigram_naive_bayes as bnb
import unigram_naive_bayes as unb
import csv

# === CONFIGURATION ===
TRAINING_DIR = 'data/WNUT-2020-Task-2-Dataset/train.tsv'
TEST_DIR = 'data/WNUT-2020-Task-2-Dataset/test.tsv'
VALIDATION_DIR = 'data/WNUT-2020-Task-2-Dataset/valid.tsv'
VALIDATION = False # set this if you want to use validation dataset instead of testing dataset
STEMMING = True    
LOWERCASE = True   
BIGRAM_LAPLACE = 4
UNIGRAM_LAPLACE = 2
POS_PRIOR = 0.5
# ======================

def compute_accuracies(predicted_labels, test_labels):
    yhats = predicted_labels
    assert len(yhats) == len(test_labels), "predicted and target label lists have different lengths"
    # Sums all the times that the predictions were right divided by how the length of the dataset
    accuracy = sum([yhats[i] == test_labels[i] for i in range(len(yhats))]) / len(yhats) 

    return accuracy

def print_value(label, value, numvalues):
    print(f"{label} {value} ({value/numvalues * 100:.2f}%)")

def print_stats(accuracy, numvalues):
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Total number of samples: {numvalues}")


def create_prediction_csv(predicted_labels, test_labels, test_id, output_file):  
    # this creates a csv with the predictions for kaggle submission
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["Id", "Label"])
        writer.writeheader()
        for i in range(len(predicted_labels)):
            writer.writerow({"Id" : test_id[i], "Label" : predicted_labels[i]})

def run_unigram_bigram_model(train_labels, train_text, validation_text):
    # Runs the combined model
    unigram_probabilities = unb.naive_bayes(train_labels, train_text, validation_text, UNIGRAM_LAPLACE, POS_PRIOR)
    bigram_probabilities = bnb.naive_bayes(train_labels, train_text, validation_text, BIGRAM_LAPLACE, POS_PRIOR)

    # Adds the log probabilities of tweets
    predictions = []
    for i in range(len(unigram_probabilities)):
            pos_prob = unigram_probabilities[i][0]  + bigram_probabilities[i][0] # The positive probability for a tweet from both models
            neg_prob = unigram_probabilities[i][1] + bigram_probabilities[i][1] # The negative probability for a tweet from both models
            predictions.append("INFORMATIVE" if pos_prob > neg_prob else "UNINFORMATIVE") # Predicts based on higher probability
    return predictions

def main():
    train_labels, train_text, test_labels, test_text, test_id, validation_labels, validation_text, validation_id = unb.load_data(TRAINING_DIR, TEST_DIR, VALIDATION_DIR)

    if VALIDATION:
        predicted_labels = run_unigram_bigram_model(train_labels, train_text, validation_text)
        create_prediction_csv(predicted_labels, validation_labels, validation_id, "unigram_bigram_prediction.csv")
        accuracy = compute_accuracies(predicted_labels, validation_labels)
        print_stats(accuracy, len(validation_labels))
    else:
        predicted_labels = run_unigram_bigram_model(train_labels, train_text, test_text)
        create_prediction_csv(predicted_labels, test_labels, test_id, "unigram_bigram_prediction.csv")
        accuracy = compute_accuracies(predicted_labels, test_labels)
        print_stats(accuracy, len(test_labels))

if __name__ == "__main__":
    main()
