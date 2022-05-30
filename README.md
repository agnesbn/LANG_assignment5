# Assignment 5 - Language Detection
The portfolio for __Language Analytics S22__ consists of 5 projects (4 class assignments and 1 self-assigned project). This is the __fifth and final assignment__ in the portfolio. 

## 1. Contribution
The initial assignment was made partly in collaboration with others from the course, but the final code is my own. I made several adjustments to the code since I first handed it in.

Utils made by Ross.

https://www.kaggle.com/code/mdzisun/language-detection-system

and

https://www.kaggle.com/code/dariussingh/nlp-dl-language-identification

## 2. Methods
Using the data, I wish to perform two tasks. First, I train a neural network model to do language classification. Second, I load the model that has been trained and use to do classification on completely new and unseen data. Thus, the order that the code is run is important – first, the 

### Train language identification model


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




## 3. Usage
### Install packages
Before running the script, you have to install the relevant packages. To do this, run the following from the command line:
```
sudo apt update
pip install --upgrade pip
pip install pandas numpy scikit-learn tensorflow nltk wordcloud
```

### Get the data
Download here: https://www.kaggle.com/datasets/zarajamshaid/language-identification-datasst.
### Main task


### Bonus task


## 4. Discussion of results

As you can tell from the plot of the number of samples per language there were a few that contained duplicates.

![](out/model_evaluations/language_counts.png)




The confusion matric
![](out/model_evaluations/confusion_matrix.png)


The history plot
![](out/model_evaluations/history_plot.png)


