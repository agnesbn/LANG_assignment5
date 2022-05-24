"""
Predict language
"""
import argparse, os
import pandas as pd
import numpy as np
import tensorflow as tf
# load saved model
from tensorflow.keras.models import load_model
# converts text into vector matrix
from sklearn.feature_extraction.text import CountVectorizer
# tokenizer
from nltk.tokenize import word_tokenize 
# Creates placeholders for categorical variables
from sklearn.preprocessing import LabelEncoder
import nltk
nltk.download('punkt')


#argparser: should allow a TXT as input
""" Basic functions """
# Argument parser
def parse_args():
    ap = argparse.ArgumentParser()
    # argument that decides whether the input is a single text or a whole directory
    ap.add_argument("-i",
                    "--input",
                    required = True,
                    help = "Whether you want to work with a single text or the whole directory")
    # name of target text (if relevant)
    ap.add_argument("-t",
                    "--text_name",
                    type=str,
                    default="arabic_wiki",
                    help = "The name of the text you want to work with")
    # directory name
    ap.add_argument("-d",
                    "--directory_name",
                    type=str,
                    default="language_examples",
                    help = "The name of the directory you want to work with")
    args = vars(ap.parse_args())
    return args

# Read TXT
def read_txt(directory_name, filename):
    filepath = os.path.join("in", directory_name, filename)
    with open(filepath, 'r') as file:
        data = file.read().replace('\n', ' ')
        data = data.split('  (Source')[0]
    return data

def write_txt(filename, language):
    inpath = os.path.join("in", "language_examples", filename)
    outpath = os.path.join("out", "language_predictions", filename) 
    with open(inpath, 'r') as input:
        with open(outpath, "w") as output:
            output.write(f"PREDICTION: The language below is {language.upper()}\n\n")
            for line in input:
                output.write(line)
    
    
""" Language prediction function """
def predict_language(sentence, filename):
    # read dataset which in .csv format
    data = pd.read_csv(os.path.join("in", "data", "dataset.csv"), encoding='utf-8').copy() 
    # number of of samples per language (category)
    data['language'].value_counts()
    # dropping duplicate samples
    data = data.drop_duplicates(subset='Text')
    data = data.reset_index(drop=True)
    # adding nonalphanumeric char to stopwords
    nonalphanumeric = ['\'', '.', ',', '\"', ':', ';', '!', '@', '#', '$', '%', '^', '&',
                       '*', '(', ')', '-', '_', '+', '=', '[', ']', '{', '}', '\\', '?', 
                       '/','>', '<', '|', ' '] 
    stopwords = nonalphanumeric
    # applying clean_text function to all rows in 'Text' column
    tokens = []
    for text in data['Text']:
        tokenised_text = word_tokenize(text)
        tokens.append(tokenised_text)
    words = []
    for l in tokens:
        lowered_text = []
        for word in l:
            if word not in stopwords:
                low_word = word.lower()
                lowered_text.append(low_word)
        words.append(lowered_text)
    data['clean_text'] = [" ".join(word) for word in words]
    # using LabelEncoder to get placeholder number values for categorical variabel 'language'
    le = LabelEncoder()
    data['language_encoded'] = le.fit_transform(data['language'])
    # list of languages encoded with thier respective indices representing their placeholder numbers
    lang_list = [i for i in range(22)]
    lang_list = le.inverse_transform(lang_list)
    lang_list = lang_list.tolist()
    # shuffling dataframe and resetting index
    data = data.sample(frac=1).reset_index(drop=True)
    # defining input variable
    # vectorizing input varible 'clean_text' into a matrix 
    X = data['clean_text']
    cv = CountVectorizer() # ngram_range=(1,2)
    X = cv.fit_transform(X)
    # loading the model
    model = load_model(os.path.join("utils", "language_identifcation_model.h5"))
    # using the model for prediction
    sent = sentence
    sent = cv.transform([sent])
    # avoid warning
    ans = model.predict(sent)
    ans = np.argmax(ans)
    answer = le.inverse_transform([ans])
    language = answer[0]
    print(f"[PREDICTION] {filename} : {language.upper()}")
    return language
                       
""" Main function """
def main():
    # parse arguments
    args = parse_args()
    if args["input"] == "single_text":
        filename = args["text_name"]
        directory_name = args["directory_name"]
        sentence = read_txt(directory_name, filename)
        language = predict_language(sentence, filename)
        write_txt(filename, language)
    elif args["input"] == "directory":
        directory_name = args["directory_name"]
        filepath = os.path.join("in", directory_name)
        text_names = os.listdir(filepath)
        text_names_clean = []
        for name in text_names:
            if name.endswith(".txt"):
                text_names_clean.append(name)
            else:
                pass
        count_texts = len(text_names_clean)
        counter = 0
        for name in text_names_clean:
            counter += 1
            filename = name
            sentence = read_txt(directory_name, filename)
            language = predict_language(sentence, filename)
            write_txt(filename, language)
            print(f"[INFO] {counter}/{count_texts} COMPLETE")
        return print("[INFO] FINISHED!")
    
    
if __name__=="__main__":
    main()
    