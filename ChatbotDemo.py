import pandas as pd
import re

### Đọc dataset
data= pd.read_csv("DataClothesShop.csv")
###
list_word=[]
list_question=list(data["question"])
for phrase in list_question:
	words_phrase=phrase.split()
	for words in words_phrase:
		list_word.append(words)


list_word=[word for phrase in list(data["question"]) for word in phrase.split()] #tách từ bỏ vào mảng list_word=[]
#print(list_word)

from collections import Counter
word_count=Counter(list_word) #đếm từ
#print (word_count)
#print (word_count.most_common(10))

###Vẽ đồ thị thể hiện tần số xuất hiện của các từ
import numpy as np
import matplotlib.pyplot as plt
value,label = zip(*word_count.items())
label=[]
value=[]
for T in word_count.most_common(15):
    label.append(T[0])
    value.append(T[1])

index = np.arange(len(label))
width = 0.3

plt.bar(index, value, width)
plt.xticks(index + width * 0.08, label)
plt.show()

### Bỏ dấu câu
def remove_punctuation(data):    
    txt=[w.lower() for w in data.split()] 
    txt=[x for w in txt for x in re.sub(r'[^\s\w]',' ',w).split()]    
    return txt

data["removed_punctuation"]=data["question"].apply(remove_punctuation)


#print (data["removed_punctuation"])

###Bỏ các stopword
from nltk.corpus import stopwords
stopword = set(stopwords.words('english'))
#print (stopword)
def remove_stopword(data):
	list_word_stop=[w for w in data if w not in stopword]
	return list_word_stop

data["removed_stopword"]=data["removed_punctuation"].apply(remove_stopword)
#print (data["removed_stopword"])

###Chuyển về từ gốc
from nltk.stem.porter import PorterStemmer
PS=PorterStemmer()
def Stemming(data):
	word_stem=[PS.stem(w) for w in data] 
	return word_stem

data["stem_data"]=data["removed_stopword"].apply(Stemming)
#print (data["stem_data"])

###
def Recreate(data):
	word=" ".join(data)
	return word

data["modified_phrase"]=data["stem_data"].apply(Recreate)
#print (data["modified_phrase"])

###data sau khi xử lý
def CleanData(data):
    data_removed_punctuation=remove_punctuation(data)
    data_remove_stopword=remove_stopword(data_removed_punctuation)
    data_stemmed=Stemming(data_remove_stopword)
    final_Data=Recreate(data_stemmed)
    return final_Data

###chuẩn bị data cho model, chuyển data về dạng Bag of Words,TF
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline, FeatureUnion

pre_Data = Pipeline([('vectoizer', CountVectorizer()),('tfidf', TfidfTransformer())])
Data_Train = pre_Data.fit_transform(data["modified_phrase"]).toarray()

#print(Data_Train)

classify=data["classification"]

###Text Classification
###Sử dụng Naive Bayes
from sklearn.naive_bayes import MultinomialNB
Classification1 = MultinomialNB().fit(Data_Train, classify)
question="How much is it?"

P=pre_Data.transform([CleanData(question)])
predict1=Classification1.predict(P)
#print (predict1)

#Đánh giá giải thuật ~0.53
#from sklearn.model_selection import train_test_split
#from sklearn.metrics import accuracy_score
#import numpy as np
#from sklearn.metrics import classification_report
#X = Data_Train
#y = classify
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 38)
#Classification1.fit(X_train, y_train)

#y_pred = Classification1.predict(X_test)
#print('AccuracyNavie %s' % accuracy_score(y_pred, y_test))
#print(classification_report(y_test, y_pred))

###Sử dụng hồi quy logistic
from sklearn.linear_model import LogisticRegression
Classification2 = LogisticRegression().fit(Data_Train, classify)

P=pre_Data.transform([CleanData(question)])
predict2=Classification2 .predict(P)
#print (predict2)

###Đánh giá giải thuật ~0.615
#from sklearn.model_selection import train_test_split
#from sklearn.metrics import accuracy_score
#import numpy as np
#from sklearn.metrics import classification_report
#X = Data_Train
#y = classify
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 38)
#Classification2.fit(X_train, y_train)

#y_pred = Classification2.predict(X_test)
#print('AccuracyLogic %s' % accuracy_score(y_pred, y_test))
#print(classification_report(y_test, y_pred))

###Sử dụng Decision Trees
from sklearn.tree import DecisionTreeClassifier
Classification3 = DecisionTreeClassifier().fit(Data_Train, classify)

P=pre_Data.transform([CleanData(question)])
predict3=Classification3.predict(P)
#print (predict3)

##Đánh giá giải thuật ~0.653
#from sklearn.model_selection import train_test_split
#from sklearn.metrics import accuracy_score
#import numpy as np
#from sklearn.metrics import classification_report
#X = Data_Train
#y = classify
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 35)
#Classification3.fit(X_train, y_train)

#y_pred = Classification3.predict(X_test)
#print('AccuracyTree %s' % accuracy_score(y_pred, y_test))
#print(classification_report(y_test, y_pred))

###Gộp 3 giải thuật để ra một giải thuật cuối cùng.
def PredictQuestion(text):
    P=pre_Data.transform([CleanData(text)])
    predict1=Classification1 .predict(P)
    #print (predict1)

    predict2=Classification2.predict(P)
    #print (predict2)
    
    predict3=Classification3.predict(P)
    #print (predict3)
    
    final_predict=[]
    final_predict=list(predict1)+list(predict2)+list(predict3)
    final_predict = Counter(final_predict)
    #print (final_predict.most_common(1)[0][0])

    return final_predict.most_common(1)[0][0]



###Bây giờ chúng ta sẽ tạo câu trả lời này khi câu hỏi đầu vào được phân loại cho lớp tương ứng.
anwser_of_group ={"time":["Every day, we work from 8:00am to 10:00PM, all day of weeks.Please feel free to contact us even out of service time, we will reply as soon as possible. Thank you!"],
                  "address":["Sorry, our store is not available in physic, we sale online. We will open the store soon. Hope you keep following us. It would be our pleasure to serve you."],
                  "price":["Oh. Seems like my boss is not here at the moment, I can't ask about the price. Please wait, I'll reply you right."],
                  "size S":["I think size S is suitable for you! You can choose size S."],
                  "size M": ["I think size M is suitable for you! You can choose size M."],
                  "size L": ["I think size L is suitable for you! You can choose size L."],
                  "size XL": ["I think size XL is suitable for you! You can choose size XL."],
                  "size XXL": ["I think size XXL is suitable for you! You can choose size XXL."],
                  "shop": ["Our products are the newest trend and fashionable. You can click the album for more details."],
                  "shipping": ["For shipping, please leave your phone and address here. My boss will advisory for you."],
                  "hello": ["Sorry, I sales, I do not confide."],
                  "love": ["Your satisfaction is my pleasure"]}

###Chọn câu trả lời từ anwser_of_group theo class
import random
def answer_chatbot(predict_class):
    answer=random.choice(anwser_of_group [predict_class])
    return answer

###Bot trò chuyện
print('Hello, my love! I am Clothesellbot, Can I help you?')
while True:
    question = input("Leave your question below: ")
    if question.strip()!= 'bye':
        perdiction=PredictQuestion(question)
        answer = answer_chatbot(perdiction)
        print('Chatbot Answer: ',answer)

    if question.strip()== 'bye':
        print('Chatbot Answer: Bye bye')
        break
