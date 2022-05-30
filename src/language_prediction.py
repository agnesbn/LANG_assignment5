"""
Predict language
"""
 # system
import os
 # argument parser
import argparse
 # data processing
import pandas as pd
import numpy as np
 # tensorflow
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
                    default="arabic_wiki.txt",
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
    # specify filepath
    filepath = os.path.join("in", directory_name, filename)
    # open file and while reading...
    with open(filepath, 'r') as file:
        # replace new line with space
        data = file.read().replace('\n', ' ')
        # do not include the '(Source: ...)' part
        data = data.split('  (Source')[0]
    # return the data
    return data

# Write predictions as TXT
def write_txt(filename, language):
    # define the filepath
    inpath = os.path.join("in", "language_examples", filename)
    # define the outpath
    outpath = os.path.join("out", "language_predictions", filename) 
    # open the input file
    with open(inpath, 'r') as input:
        # open the output file
        with open(outpath, "w") as output:
            # write prediction
            output.write(f"PREDICTION: The language below is {language.upper()}\n\n")
            # write input data below prediction
            for line in input:
                output.write(line)
    
    
""" Language prediction function """
def predict_language(sentence, filename):
    """
    Part 1: running some of the code from language_identification.py again
    """
    # read dataset
    data = pd.read_csv(os.path.join("in", "data", "dataset.csv"), encoding='utf-8').copy() 
    # drop duplicate samples
    data = data.drop_duplicates(subset='Text')
    data = data.reset_index(drop=True)
    # add nonalphanumeric characters to stopwords
    nonalphanumeric = ['\'', '.', ',', '\"', ':', ';', '!', '@', '#', '$', '%', '^', '&',
                       '*', '(', ')', '-', '_', '+', '=', '[', ']', '{', '}', '\\', '?', 
                       '/','>', '<', '|', ' '] 
    stopwords = nonalphanumeric
    # tokenise texts
    tokens = []
    for text in data['Text']:
        tokenised_text = word_tokenize(text)
        tokens.append(tokenised_text)
    # lower characters in texts
    words = []
    for l in tokens:
        lowered_text = []
        for word in l:
            if word not in stopwords:
                low_word = word.lower()
                lowered_text.append(low_word)
        words.append(lowered_text)
    # create 'cleaned_text' column
    data['clean_text'] = [" ".join(word) for word in words]
    # use placeholder number values to encode categorical 'language' variables 
    le = LabelEncoder()
    data['language_encoded'] = le.fit_transform(data['language'])
    # list of languages with thier encoded indices
    lang_list = [i for i in range(22)]
    lang_list = le.inverse_transform(lang_list)
    lang_list = lang_list.tolist()
    # shuffle dataframe and reset index
    data = data.sample(frac=1).reset_index(drop=True)
    # Define input variable
    # vectorise input varible 'clean_text' into a matrix
    X = data['clean_text']
    cv = CountVectorizer() # ngram_range=(1,2)
    X = cv.fit_transform(X)
    """
    Part 2: loading model and doing language prediction based on text input
    """
    # load the model
    model = load_model(os.path.join("utils", "language_identifcation_model.h5"))
    # use the model for prediction
    sent = sentence
    sent = cv.transform([sent])
    ans = model.predict(sent) 
    ans = np.argmax(ans)
    answer = le.inverse_transform([ans])
    # get most likely candidate 
    language = answer[0]
    # print the prediction
    print(f"[PREDICTION] {filename} : {language.upper()}")
    # return language for saving TXT
    return language
                       
""" Main function """
def main():
    # parse arguments
    args = parse_args()
    # if the input is a single text
    if args["input"] == "single_text":
        # get file and directory name
        filename = args["text_name"]
        directory_name = args["directory_name"]
        # read the TXT
        sentence = read_txt(directory_name, filename)
        # predict the language in the text
        language = predict_language(sentence, filename)
        # write results TXT
        write_txt(filename, language)
    # otherwise, if the input is a directory
    elif args["input"] == "directory":
        # get directory name
        directory_name = args["directory_name"]
        # get directory path
        filepath = os.path.join("in", directory_name)
        # get list of files in directory
        text_names = os.listdir(filepath)
        # make list of only TXT files in directory
        text_names_clean = []
        for name in text_names:
            if name.endswith(".txt"):
                text_names_clean.append(name)
            else:
                pass
        # count texts
        count_texts = len(text_names_clean)
        # initialise counter
        counter = 0
        # for each filename in list
        for name in text_names_clean:
            # add 1 to the counter
            counter += 1
            # specify filename
            filename = name
            # read the TXT
            sentence = read_txt(directory_name, filename)
            # predict the language in the text
            language = predict_language(sentence, filename)
            # write result TXT
            write_txt(filename, language)
            # print message
            print(f"[INFO] {counter}/{count_texts} COMPLETE")
        # print final message
        return print("[INFO] FINISHED!")
    
    
if __name__=="__main__":
    main()
    