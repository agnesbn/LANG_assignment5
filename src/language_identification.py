"""
Language identification
"""
""" Import relevant packages """
 # system
import os
 # data processing
import numpy as np
import pandas as pd
import json
 # plotting
import matplotlib.pyplot as plt
import seaborn as sns
 # from scikit-learn
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
 # natural language processing
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
nltk.download('punkt')
 # word cloud tool
from wordcloud import WordCloud
 # from tensorflow
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier 
 # argument parser
import argparse

""" Basic functions """
# Argument parser
def parse_args():
    # initialise argument parser
    ap = argparse.ArgumentParser()
    """
    Data saving arguments
    """
    # plot name argument
    ap.add_argument("-p",
                    "--plot_name",
                    default="history_plot",
                    help="The name you wish to save the history plot under")
    # report name argument
    ap.add_argument("-r", 
                    "--report_name", 
                    default="classification_report", 
                    help="The name you wish to save the classification report under")
    # confusion matrix plot name argument
    ap.add_argument("-c", 
                    "--cm_name", 
                    default="confusion_matrix", 
                    help="The name you wish to save the confusion matrix under")
    # language counts plot name argument
    ap.add_argument("-l", 
                    "--lc_plot_name", 
                    default="language_counts", 
                    help="The name you wish to save the language counts plot under")
    
    """
    Hyperparameters for model
    """
    # epoch argument
    ap.add_argument("-e",
                    "--epochs",
                    type=int,
                    default=8,
                    help = "The number of epochs the model runs for")
    # batch size argument
    ap.add_argument("-b",
                    "--batch_size",
                    type=int,
                    default=256,
                    help="Size of batches the data is processed by")
    # early stopping patience argument
    ap.add_argument("-s",
                    "--es_patience",
                    type=int,
                    default=1,
                    help="The patience of the early stopping callback function")
    # monitored metric argument
    ap.add_argument("-m",
                    "--monitor_metric",
                    type=str,
                    default="accuracy",
                    help="The metric that the early stopping callback function monitors by")
    args = vars(ap.parse_args())
    return args

# Save history plot
def save_history(H, epochs, plot_name):
    outpath = os.path.join("out", "model_evaluations", f"{plot_name}.png")
    plt.style.use("seaborn-colorblind")
    
    plt.figure(figsize=(12,6))
    plt.suptitle(f"Human Action Classification", fontsize=16)
    plt.subplot(1,2,1)
    plt.plot(np.arange(0, epochs), H.history["loss"], label="Train")
    plt.plot(np.arange(0, epochs), H.history["val_loss"], label="Validation", linestyle=":")
    plt.title("Loss curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.legend()
    
    plt.subplot(1,2,2)
    plt.plot(np.arange(0, epochs), H.history["accuracy"], label="Train")
    plt.plot(np.arange(0, epochs), H.history["val_accuracy"], label="Validation", linestyle=":")
    plt.title("Accuracy curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.legend()
    plt.savefig(os.path.join(outpath))

# Save the classification report as TXT
def report_to_txt(report, epochs, report_name, batch_size, es_patience, monitor_metric):
    # specify outpath
    outpath = os.path.join("out", "model_evaluations", f"{report_name}.txt")
    # open TXT
    with open(outpath,"w") as file:
        # write headings
        file.write(f"Classification report\nData: Language Identification dataset\nEpochs: {epochs}\nBatch size: {batch_size}\nPatience of Early Stopping: {es_patience}\nMonitor metric: {monitor_metric}\n")
        # write report
        file.write(str(report))

""" Language identification function """
def language_identification(plot_name, report_name, cm_name, lc_plot_name, epochs, batch_size, es_patience, monitor_metric):
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
    # save plot of number of samples in each language
    plt.figure(figsize=(10,10))
    plt.title('Language Counts')
    ax = sns.countplot(y=data['language'], data=data)
    plt.savefig(os.path.join("out", "model_evaluations", f"{lc_plot_name}.png"))
    # shuffle dataframe and reset index
    data = data.sample(frac=1).reset_index(drop=True)
    # Define input variable
    # vectorise input varible 'clean_text' into a matrix 
    X = data['clean_text']
    cv = CountVectorizer()
    X = cv.fit_transform(X)
    # change datatype of the number into uint8 to consume less memory
    X = X.astype('uint8') # uint8 and float32
    # define target variable
    y = data['language_encoded']
    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)
    # convert data into numpy array supported by tensorflow
    X_train = X_train.toarray()
    X_test = X_test.toarray()
    # define hyperparameters
    INPUT_SIZE = X_train.shape[1]
    OUTPUT_SIZE = len(data['language_encoded'].unique())
    EPOCHS = epochs
    BATCH_SIZE = batch_size
    # configure early stopping
    es = EarlyStopping(monitor=monitor_metric, patience=es_patience)
    # create the MLP model
    model = Sequential([
        Dense(256, activation='softsign', kernel_initializer='glorot_uniform', input_shape=(INPUT_SIZE,)),
        Dense(128, activation='softsign', kernel_initializer='glorot_uniform'),
        Dense(64, activation='softsign', kernel_initializer='glorot_uniform'),
        Dense(OUTPUT_SIZE, activation='softmax')])
    # compile the MLP model
    model.compile(optimizer='Adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    # summary of the MLP model
    model.summary()
    # fit the model with earlystopping callback to avoid overfitting 
    H = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.3, callbacks=[es], verbose=1)
    y_pred_prob = model.predict(X_test) # returns an array containing probability for each category being output
    # get most likely category for each input
    y_pred = []
    for i in y_pred_prob:
        out = np.argmax(i)
        y_pred.append(out)
    y_pred = np.array(y_pred)
    # create confusion matrix - normalise data
    cm = confusion_matrix(y_test, y_pred, normalize="true")
    # make heat map of confusion matrix with annotations
    plt.figure(figsize=(12,10))
    plt.title('Confusion Matrix - MLP Model', Fontsize=20)
    sns.heatmap(cm, xticklabels=lang_list, yticklabels=lang_list, cmap='rocket_r', linecolor='white', linewidth=.005, annot=True)
    plt.xlabel('Predicted Language', fontsize=15)
    plt.ylabel('True Language', fontsize=15)
    # save figure
    plt.savefig(os.path.join("out", "model_evaluations", f"{cm_name}.png"))
    # make classification report
    report = classification_report(y_test, y_pred, target_names=lang_list)
    # get actual number of epochs run
    n_epochs = len(H.history['loss'])
    # save classification report as TXT
    report_to_txt(report, n_epochs, report_name, batch_size, es_patience, monitor_metric)
    # save history plot
    save_history(H, n_epochs, plot_name=plot_name)
    # saving the model
    model.save(os.path.join("utils", "language_identifcation_model.h5"))
    # print the report in the terminal
    return print(report)


""" Main function """
def main():
    # parse arguments
    args = parse_args()
    # get hyperparameters from argparse
    plot_name = args["plot_name"]
    report_name = args["report_name"]
    cm_name = args["cm_name"]
    lc_plot_name = args["lc_plot_name"]
    epochs = args["epochs"]
    batch_size = args["batch_size"]
    es_patience = args["es_patience"]
    monitor_metric = args["monitor_metric"]
    # run the language identification function
    language_identification(plot_name, report_name, cm_name, lc_plot_name, 
                            epochs, batch_size, es_patience, monitor_metric)
    
if __name__=="__main__":
    main()