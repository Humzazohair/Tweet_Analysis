import csv
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer

stemmer = PorterStemmer()
tokenizer = RegexpTokenizer(r'\w+')

def read_tsv_file(filepath, has_header=True):
    texts = []
    labels = []
    ids = []

    with open(filepath, "r", encoding="utf-8") as file:
        if has_header:
            reader = csv.DictReader(file, delimiter="\t")
        else:
            reader = csv.DictReader(file, delimiter="\t", fieldnames=["Id", "Text", "Label"])

        if has_header:
            next(reader)  # Skip the first row if it's actually the header but being treated as data

        for row in reader:
            id = row["Id"]
            ids.append(id)

            text_raw = row["Text"].strip().lower()
            tokens = tokenizer.tokenize(text_raw)
            stemmed_tokens = [stemmer.stem(word) for word in tokens if word.isalnum()]
            texts.append(stemmed_tokens)

            label_text = row.get("Label", "UNINFORMATIVE").strip().upper()
            label = "INFORMATIVE" if label_text == "INFORMATIVE" else "UNINFORMATIVE"
            labels.append(label)

    return labels, texts, ids
