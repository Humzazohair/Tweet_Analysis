# Tweet Analysis

This project's goal is to analyze tweets based on whether they are Informative or Uninformative. 

## Method

- First we created a unigram model and tested the accuracy
- Then we created a bigram model and tested the accuracy
- We realized that these two methods alone are not enough


## How to setup and run
1. Install the non-standard dependencies/libraries:
- `reader`
- `csv`
- `nltk`

2. Navigate to the parent directory: `Tweet_Analysis`

<br>

3. Select validation/test to run model on: Line 10 of `unigram_bigram_model/main.py`, set `Validation` to `True` or `False`

<br>

4. Run the code:

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`python3 unigram_bigram_model/main.py`

<br>

5. View the predictions in `validation_predictions.csv` if running on the validation data or `unigram_bigram_prediction.csv` if running on the test data

<br>

#### NOTE
- The folders `unigram_model` and `bigram_model` are implementations for the uncombined versions of the model. They are included as they were part of our process in coming up with the combined model approach.