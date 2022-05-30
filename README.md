# Assignment 5 - Language Detection
The portfolio for __Language Analytics S22__ consists of 5 projects (4 class assignments and 1 self-assigned project). This is the __fifth and final assignment__ in the portfolio. 

## 1. Contribution
This final project was not made in collaboration with others from the course. I did, however, find a lot of inspiration for how to work with the data from [this Kaggle notebook](https://www.kaggle.com/code/dariussingh/nlp-dl-language-identification).

## 2. Methods
Using the data, I wish to perform two tasks. First, I train a neural network model to do language classification. Second, I load the model that has been trained and use to do classification on completely new and unseen data. Thus, the order that the code is run is important – first, the 

### Train language classification model


### Perform language prediction
I tested the prediction on 4-5 lines of text from wikipedia articles in the given languages. 
- Arabic: [https://ar.wikipedia.org/wiki/اللغة_العربية](https://ar.wikipedia.org/wiki/اللغة_العربية).
- Chinese: [https://zh.wikipedia.org/zh-cn/汉语](https://zh.wikipedia.org/zh-cn/汉语).
    - the article is in Simplified Chinese, though the Chinese data from Kaggle seems to include both Traditional and Simplified Chinese – as far as I could tell from checking a few of the texts using a [tool to tell if a text is simplified or traditional Chinese](https://www.chineseconverter.com/en/convert/find-out-if-simplified-or-traditional-chinese).
    - as Japanese uses Traditional Chinese characters, the predictions are likely to be most accurate when using simplified Chinese input.
- Dutch: [https://nl.wikipedia.org/wiki/Nederlands](https://nl.wikipedia.org/wiki/Nederlands).
- English: [https://en.wikipedia.org/wiki/English_language](https://en.wikipedia.org/wiki/English_language).
- Estonian: [https://et.wikipedia.org/wiki/Eesti_keel](https://et.wikipedia.org/wiki/Eesti_keel).
- French: [https://fr.wikipedia.org/wiki/Français](https://fr.wikipedia.org/wiki/Français).
- Hindi: [https://hi.wikipedia.org/wiki/हिन्दी](https://hi.wikipedia.org/wiki/हिन्दी).
- Indonesian: [https://id.wikipedia.org/wiki/Bahasa_Indonesia](https://id.wikipedia.org/wiki/Bahasa_Indonesia).
- Japanese: [https://ja.wikipedia.org/wiki/日本語](https://ja.wikipedia.org/wiki/日本語).
- Korean: [https://ko.wikipedia.org/wiki/한국어](https://ko.wikipedia.org/wiki/한국어).
- Latin: [https://la.wikipedia.org/wiki/Lingua_Latina](https://la.wikipedia.org/wiki/Lingua_Latina).
- Persian: [https://fa.wikipedia.org/wiki/زبان_فارسی](https://fa.wikipedia.org/wiki/زبان_فارسی).
- Portugese: [https://pt.wikipedia.org/wiki/L%C3%ADngua_portuguesa](https://pt.wikipedia.org/wiki/L%C3%ADngua_portuguesa).
- Pushto (Pashto): [https://ps.wikipedia.org/wiki/پښتو](https://ps.wikipedia.org/wiki/پښتو).
- Romanian: [https://ro.wikipedia.org/wiki/Limba_română](https://ro.wikipedia.org/wiki/Limba_română).
- Russian: [https://ru.wikipedia.org/wiki/Русский_язык](https://ru.wikipedia.org/wiki/Русский_язык).
- Spanish: [https://es.wikipedia.org/wiki/Idioma_español](https://es.wikipedia.org/wiki/Idioma_español).
- Swedish: [https://sv.wikipedia.org/wiki/Svenska](https://sv.wikipedia.org/wiki/Svenska).
- Tamil: [https://ta.wikipedia.org/wiki/தமிழ்](https://ta.wikipedia.org/wiki/தமிழ்).
- Thai: [https://th.wikipedia.org/wiki/ภาษาไทย](https://th.wikipedia.org/wiki/ภาษาไทย).
- Turkish: [https://tr.wikipedia.org/wiki/Türkçe](https://tr.wikipedia.org/wiki/Türkçe).
- Urdu: [https://ur.wikipedia.org/wiki/اردو](https://ur.wikipedia.org/wiki/اردو).
(All Wikipedia pages were accessed and gathered 24 May 2022)


## 3. Usage
### Install packages
Before running the script, you have to install the relevant packages. To do this, run the following from the command line:
```
sudo apt update
pip install --upgrade pip
pip install pandas numpy scikit-learn tensorflow nltk wordcloud
```

### Get the data
- Download the data here: https://www.kaggle.com/datasets/zarajamshaid/language-identification-datasst.
- Place the data CSV in the `in/data` folder, so that the path to the input data is `in/data/dataset.csv`.

### Language classification
Make sure your current directory is `LANG_assignment5` and then, run:
```
python src/language_classification.py (--plot_name <PLOT NAME> --report_name <REPORT NAME> 
--cm_name <CONFUSION MATRIX NAME> --lc_plot_name <LANGUAGE COUNTS PLOT NAME>)
```

__Input__:
- `<PLOT NAME>`: The name you wish to save the history plot under. The default is `history_plot`.
- `<REPORT NAME>`: The name you wish to save the classification report under. The default is `classification_report`.
- `<CONFUSION MATRIX NAME>`: The name you wish to save the confusion matrix under. The default is `confusion_matrix`.
- `<LANGUAGE COUNTS PLOT NAME>`: The name you wish to save the language counts plot under. The default is `language_counts`. 

The classification report and the different result plots are saved in [`out/model_evaluations`](https://github.com/agnesbn/LANG_assignment5/tree/main/out/model_evaluations) and the model is saved as `language_identifcation_model.h5` in [`utils`](https://github.com/agnesbn/LANG_assignment5/tree/main/utils).

### Language prediction
- To do the language prediction, place a collection of TXT files which contain the different languages in the `in/language_examples` folder.
    - The specific files I used to test this on will be provided on `Digital Eksamen`.
- After the data has been placed in the folder, make sure your current directory is `LANG_assignment5` and from the command line, run:
```
python src/language_prediction.py --input <INPUT> (--text_name <TEXT NAME> --directory_name <DIRECTORY NAME>)
```
__Required input__:
- `<INPUT>`: Whether your input is a directory or a single file. To run language prediction for all files in a directory, put in `directory`, and to do the task for a single file, put in `single_text`.

__Optional input__:
- `<TEXT NAME>`: The name of the TXT file, you wish to run the language prediction on (if relevant). The default is `arabic_wiki.txt`.
- `<DIRECTORY NAME>`: The name of the directory with the files, that you wish to run the language prediction on (if relevant). The default is `language_examples`.

The results are printed in the command line and saved in [`out/language_predictions`](https://github.com/agnesbn/LANG_assignment5/tree/main/out/language_predictions).

## 4. Discussion of results

As you can tell from the plot of the number of samples per language there were a few that contained duplicates.

![](out/model_evaluations/language_counts.png)




The confusion matrix
![](out/model_evaluations/confusion_matrix.png)


The history plot
![](out/model_evaluations/history_plot.png)


