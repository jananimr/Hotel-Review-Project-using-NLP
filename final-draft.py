#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# # Loading the Data Set

# In[2]:


data_new=pd.read_csv("C:/Users/khann/Downloads/tripadvisor (5).csv")
data_new.head()


# # Exploratory Data Analysis

# In[3]:


data_new.info()


# In[4]:


data_new.nunique()


# In[5]:


data =data_new.drop([6130])


# In[6]:


data1=data.iloc[:,:6]
data1


# In[7]:


data2=data.iloc[:,6:11]


# In[8]:


data2


# In[9]:


data3=data.iloc[:,11:16]
data3


# In[10]:


data4=data.iloc[:,16:]
data4


# In[11]:


data.drop(data.iloc[:, 11:], inplace = True, axis = 1)


# In[12]:


data.drop(data.columns[[0, 1, 2, 4]], axis = 1, inplace = True)


# In[13]:


data


# In[14]:


data.drop(data.columns[[2, 3, 4]], axis = 1, inplace = True)


# In[15]:


data


# In[16]:


data.rename(columns = {'ui_header_link':'Name', 'default':'Place',
                              'ocfR3SKN':'Review_title', 'IRsGHoPm':'Review'}, inplace = True)
data


# In[17]:


data.nunique()


# In[18]:


data_tm=data[['Review_title', 'Review']]
data_tm


# In[19]:


text=data[['Review']]


# In[20]:


text


# In[21]:


text1=text.dropna()


# In[22]:


text1.info()


# ## Feature Extraction

# In[23]:


import numpy as np 
import string 
import spacy 

from matplotlib.pyplot import imread
from matplotlib import pyplot as plt
from wordcloud import WordCloud
get_ipython().run_line_magic('matplotlib', 'inline')


# # Tonekenization

# In[24]:


import nltk
from nltk.tokenize import word_tokenize
reviews = text1.Review.str.cat(sep=' ')


# In[25]:


#function to split text into word
tokens = word_tokenize(reviews)

vocabulary = set(tokens)
print(len(vocabulary))
frequency_dist = nltk.FreqDist(tokens)
sorted(frequency_dist,key=frequency_dist.__getitem__, reverse=True)[0:50]


# # Word Cloud of most Frequest words

# In[26]:


import matplotlib.pyplot as plt
wordcloud = WordCloud().generate_from_frequencies(frequency_dist)
plt.imshow(wordcloud)
plt.axis("off")
plt.show()


# In[27]:


from PIL import Image


# In[28]:


from nltk.corpus import stopwords
from nltk import sent_tokenize, word_tokenize
from wordcloud import WordCloud, STOPWORDS

def wc(text1,bgcolor,title):
    plt.figure(figsize = (100,100))
    wc = WordCloud(background_color = bgcolor, max_words = 1000,  max_font_size = 50)
    wc.generate(' '.join(text1))
    plt.imshow(wc)
    plt.axis('off')


# In[29]:


from nltk.corpus import stopwords
nltk.download('stopwords')


# In[30]:


#pip install stop-words


# In[31]:


from collections import Counter
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
import re
import seaborn as sns

top_N = 100

a = text1['Review'].str.lower().str.cat(sep=' ')

# removes punctuation,numbers and returns list of words
b = re.sub('[^A-Za-z]+', ' ', a)

#remove all the stopwords from the text
stop_words = list(get_stop_words('en'))         
nltk_words = list(stopwords.words('english'))   
stop_words.extend(nltk_words)

word_tokens = word_tokenize(b)
filtered_sentence = [w for w in word_tokens if not w in stop_words]
filtered_sentence = []
for w in word_tokens:
    if w not in stop_words:
        filtered_sentence.append(w)

# Remove characters which have length less than 2  
without_single_chr = [word for word in filtered_sentence if len(word) > 2]

# Remove numbers
cleaned_data_review = [word for word in without_single_chr if not word.isnumeric()]        

# Calculate frequency distribution
word_dist = nltk.FreqDist(cleaned_data_review)
rslt = pd.DataFrame(word_dist.most_common(top_N),
                    columns=['Word', 'Frequency'])

plt.figure(figsize=(10,10))
sns.set_style("whitegrid")
ax = sns.barplot(x="Word",y="Frequency", data=rslt.head(7))


# In[32]:


wc(cleaned_data_review,'black','Common Words' )


# # Sentiment Analysis

# In[33]:


from textblob import TextBlob

bloblist_desc = list()

text1_descr_str=text1['Review'].astype(str)
for row in text1_descr_str:
    blob = TextBlob(row)
    bloblist_desc.append((row,blob.sentiment.polarity, blob.sentiment.subjectivity))
    text1_polarity_desc = pd.DataFrame(bloblist_desc, columns = ['sentence','sentiment','polarity'])
 
def f(text1_polarity_desc):
    if text1_polarity_desc['sentiment'] > 0.25:
        val = "1"
    elif text1_polarity_desc['sentiment'] == 0:
        val = "0"
    else:
        val = "-1"
    return val

text1_polarity_desc['Sentiment_Type'] = text1_polarity_desc.apply(f, axis=1)

plt.figure(figsize=(10,10))
sns.set_style("whitegrid")
ax = sns.countplot(x="Sentiment_Type", data=text1_polarity_desc)


# In[34]:


text1_polarity_desc


# In[35]:


data2.rename(columns = {'ocfR3SKN':'review_title'}, inplace = True)


# In[36]:


extracted_col = data2["review_title"]
display(extracted_col)


# In[37]:


text1_polarity_desc = text1_polarity_desc.join(extracted_col)
display(text1_polarity_desc)


# In[38]:


text1_polarity_desc.dropna(inplace=True)


# In[39]:


text1_polarity_desc


# In[40]:


# split df - positive and negative sentiment:
positive = text1_polarity_desc[text1_polarity_desc['sentiment'] > 0]
negative = text1_polarity_desc[text1_polarity_desc['sentiment'] < 0]


# # Positive Sentiment WordCloud

# In[41]:


stopwords = set(STOPWORDS)
stopwords.update(["br", "href"])

pos = " ".join(review for review in positive.review_title)
wordcloud2 = WordCloud(stopwords=stopwords).generate(pos)
plt.imshow(wordcloud2, interpolation='bilinear')
plt.axis("off")
plt.show()


# # Negative Sentiment WordCloud

# In[42]:


stopwords = set(STOPWORDS)
stopwords.update(["br", "href","Great","Best","Good","Nice"])

                                                                      ## number of positive words were used in negative context hence to be removed because they were included in negative sentiment

neg = " ".join(review for review in negative.review_title)
wordcloud3 = WordCloud(stopwords=stopwords).generate(neg)
plt.imshow(wordcloud3, interpolation='bilinear')
plt.axis("off")
plt.savefig('wordcloud33.png')
plt.show()


# In[43]:


text1_polarity_desc['word_count'] = text1_polarity_desc['sentence'].apply(lambda x: len(str(x).split(" ")))
text1_polarity_desc['char_count'] = text1_polarity_desc['sentence'].str.len() 
text1_polarity_desc['sentence'] = text1_polarity_desc['sentence'].apply(lambda x: " ".join(x.lower() for x in x.split()))
text1_polarity_desc.head()


# In[44]:


text1_polarity_desc['sentence'] = text1_polarity_desc['sentence'].str.replace('[^\w\s]','')


# In[45]:


text1_polarity_desc['sentence'] = text1_polarity_desc['sentence'].apply(lambda x: " ".join(x for x in x.split() if x not in stopwords))
text1_polarity_desc.head()


# # Removing Punctuation

# In[46]:


def remove_punctuation(sentence):
    final = "".join(u for u in sentence if u not in ("?", ".", ";", ":",  "!",'"'))
    return final

text1_polarity_desc['sentence'] = text1_polarity_desc['sentence'].apply(remove_punctuation)

text1_polarity_desc = text1_polarity_desc.dropna(subset=['review_title'])

text1_polarity_desc['review_title'] = text1_polarity_desc['review_title'].apply(remove_punctuation)


# In[47]:


text1_polarity_desc.head()


# In[48]:


df = text1_polarity_desc[['sentence','review_title','Sentiment_Type']]
df.head()


# # Most Frequent Words

# In[49]:


pd.options.mode.chained_assignment = None

freq = pd.Series(' '.join(df['sentence']).split()).value_counts()[:20]
freq = list(freq.index)
df['sentence'] = df['sentence'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))


# In[50]:


freq


# # Least Frequent Words

# In[51]:


pd.options.mode.chained_assignment = None

low_freq = pd.Series(' '.join(df['sentence']).split()).value_counts()[-20:]
freq1 = list(low_freq.index)
df['sentence'] = df['sentence'].apply(lambda x: " ".join(x for x in x.split() if x not in freq1))


# In[52]:


freq1


# # Bigrams Creation

# In[53]:


import collections
from collections import Counter

counts_Reviews = collections.Counter()
for i in df['sentence']:
    words_A = word_tokenize(i)
    counts_Reviews.update(nltk.bigrams(words_A))    
Bigram_counts_Reviews = counts_Reviews.most_common(10)
Bigram_counts_Reviews


# 

# # TfidfVectorizer 

# In[54]:


from sklearn.feature_extraction.text import TfidfVectorizer
tfidf=TfidfVectorizer()
TFIDF=tfidf.fit_transform(df['sentence'])
print(TFIDF)


# In[55]:


# Lemmatization
from textblob import Word
df['sentence']= df['sentence'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))


# In[56]:


import re
pattern = r"((?<=^)|(?<= )).((?=$)|(?= ))"
df['sentence']= df['sentence'].apply(lambda x:(re.sub(pattern, '',x).strip()))
df['sentence'].head()


# In[57]:


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import cohen_kappa_score


# In[58]:


tv=TfidfVectorizer()


# In[59]:


df


# In[60]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import cohen_kappa_score


# In[61]:


tv=TfidfVectorizer()


# In[62]:


X=df.iloc[:,0]
y=df.iloc[:,2]


# In[63]:


X=tv.fit_transform(df.sentence)
X


# In[64]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)


# # LOGISTIC REGRESSION without Smote

# In[65]:


model=LogisticRegression()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)


# In[66]:


from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
print('Testing accuracy:',accuracy_score(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))


# In[67]:


n_errors_Log=print((y_pred!=y_test).sum())
cohen_kappa_score(y_test,y_pred)


# In[68]:


print('Training accuracy:',accuracy_score(y_train,model.predict(X_train)))


# # Decision Tree Classifier

# In[69]:


from sklearn.tree import DecisionTreeClassifier

model1=DecisionTreeClassifier(criterion='entropy')
model1.fit(X_train,y_train)


# In[70]:


y_pred1=model1.predict(X_test)


# In[71]:


from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

print('Testing accuracy:',accuracy_score(y_test,y_pred1))


# In[72]:


print(confusion_matrix(y_test,y_pred1))


# In[73]:


print(classification_report(y_test,y_pred1))


# In[74]:


n_errors_Dec=print((y_pred1!=y_test).sum())
cohen_kappa_score(y_test,y_pred1)


# In[75]:


print('Training accuracy:',accuracy_score(y_train,model1.predict(X_train)))


# # RANDOM FOREST CLASSIFIER

# In[76]:


from sklearn.ensemble import RandomForestClassifier
model2=RandomForestClassifier(n_estimators=100)
model2.fit(X_train,y_train)
y_pred2=model2.predict(X_test)


# In[77]:


from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
print('Testing accuracy:',accuracy_score(y_test,y_pred2))
print(confusion_matrix(y_test,y_pred2))
print(classification_report(y_test,y_pred2))


# In[78]:


n_errors_Ran=print((y_pred2!=y_test).sum())
cohen_kappa_score(y_test,y_pred2)


# In[79]:


print('Training accuracy:',accuracy_score(y_train,model2.predict(X_train)))


# # Extratreeclassifier

# In[80]:


from sklearn.ensemble import ExtraTreesClassifier 

model3=ExtraTreesClassifier()
model3.fit(X_train,y_train)
y_pred3=model3.predict(X_test)


# In[81]:


from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
print('Testing accuracy:',accuracy_score(y_test,y_pred3))
print(confusion_matrix(y_test,y_pred3))
print(classification_report(y_test,y_pred3))


# In[82]:


n_errors_ext=print((y_pred3!=y_test).sum())
cohen_kappa_score(y_test,y_pred3) 


# In[83]:


print('Training accuracy:',accuracy_score(y_train,model3.predict(X_train)))


# # Support Vector Machine

# In[84]:


from sklearn.svm import SVC  

model4=SVC()
model4.fit(X_train,y_train)

y_pred4=model4.predict(X_test)


# In[85]:


from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

print('Testing accuracy:',accuracy_score(y_test,y_pred4))
print(confusion_matrix(y_test,y_pred4))
print(classification_report(y_test,y_pred4))


# In[86]:


n_errors_svc=print((y_pred4!=y_test).sum())
cohen_kappa_score(y_test,y_pred4)


# In[87]:


print('Training accuracy:',accuracy_score(y_train,model4.predict(X_train)))


# # NEURAL NETWORK

# In[88]:


from sklearn.neural_network import MLPClassifier 

model5=MLPClassifier(hidden_layer_sizes=(5,5))
model5.fit(X_train,y_train)
y_pred5=model5.predict(X_test)


# In[89]:


from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

print('Testing accuracy:',accuracy_score(y_test,y_pred5))
print(confusion_matrix(y_test,y_pred5))
print(classification_report(y_test,y_pred5))


# In[90]:


n_errors_nn=print((y_pred5!=y_test).sum())
cohen_kappa_score(y_test,y_pred5)
print('Training accuracy:',accuracy_score(y_train,model5.predict(X_train)))


# In[91]:


print('Training accuracy:',accuracy_score(y_train,model5.predict(X_train)))


# # Bagging Classifier

# In[92]:


from sklearn.ensemble import BaggingClassifier 

model6=BaggingClassifier(DecisionTreeClassifier(criterion='entropy'))
model6.fit(X_train,y_train)

y_pred6=model6.predict(X_test)


# In[93]:


from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

print('Testing accuracy:',accuracy_score(y_test,y_pred6))
print(confusion_matrix(y_test,y_pred6))
print(classification_report(y_test,y_pred6))


# In[94]:


n_errors_BC=print((y_pred6!=y_test).sum())
cohen_kappa_score(y_test,y_pred6) 


# In[95]:


print('Training accuracy:',accuracy_score(y_train,model6.predict(X_train)))


# # Extreme Grdient Boosting Algorithm

# In[96]:


import xgboost as xgb 

model7=xgb.XGBClassifier()
model7.fit(X_train,y_train)


# In[97]:


y_pred7=model7.predict(X_test)


# In[98]:


from sklearn.metrics import accuracy_score,classification_report,confusion_matrix


# In[99]:


print('Testing accuracy:',accuracy_score(y_test,y_pred7))
print(confusion_matrix(y_test,y_pred7))
print(classification_report(y_test,y_pred7))


# In[100]:


n_errors_BC=print((y_pred7!=y_test).sum())
cohen_kappa_score(y_test,y_pred7) 


# In[101]:


print('Training accuracy:',accuracy_score(y_train,model7.predict(X_train)))


# # SMOTE

# In[102]:


#.We can observe majority class for no class and monority class for yes class
#So we need to solve class imbalance problem with the help of SMOTE

from imblearn.over_sampling import SMOTE


# In[103]:


sm=SMOTE(random_state=444)
X_train_res,y_train_res=sm.fit_resample(X_train,y_train)
print('Original dataset shape %s' % Counter(y_train))
print('Resampled dataset shape %s' % Counter(y_train_res))


# # LogisticRegression after SMOTE

# In[104]:


from sklearn.linear_model import LogisticRegression 

model9=LogisticRegression()
model9.fit(X_train_res,y_train_res)
y_pred9=model9.predict(X_test)


# In[105]:


from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
print('Testing accuracy:',accuracy_score(y_test,y_pred9))
print(confusion_matrix(y_test,y_pred9))
print(classification_report(y_test,y_pred9))


# In[106]:


n_errors_Log_SM=print((y_pred9!=y_test).sum())
cohen_kappa_score(y_test,y_pred9)


# In[107]:


print('Training accuracy:',accuracy_score(y_train,model9.predict(X_train)))


# # DECISION TREE with Smote

# In[108]:


from sklearn.tree import DecisionTreeClassifier

model10=DecisionTreeClassifier(criterion='entropy')
model10.fit(X_train_res,y_train_res)


# In[109]:


y_pred10=model10.predict(X_test)


# In[110]:


from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

print('Testing accuracy:',accuracy_score(y_test,y_pred10))
print(confusion_matrix(y_test,y_pred10))
print(classification_report(y_test,y_pred10))


# In[111]:


n_errors_Dec_SM=print((y_pred10!=y_test).sum())
cohen_kappa_score(y_test,y_pred10)


# In[112]:


print('Training accuracy:',accuracy_score(y_train,model10.predict(X_train)))


# # RANDOM FOREST CLASSIFIER with Smote

# In[113]:


from sklearn.ensemble import RandomForestClassifier
model11=RandomForestClassifier(n_estimators=100)
model11.fit(X_train_res,y_train_res)


# In[114]:


y_pred11=model11.predict(X_test)


# In[115]:


from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
print('Testing accuracy:',accuracy_score(y_test,y_pred11))
print(confusion_matrix(y_test,y_pred11))
print(classification_report(y_test,y_pred11))


# In[116]:


n_errors_Ran_SM=print((y_pred11!=y_test).sum())
cohen_kappa_score(y_test,y_pred11)


# In[117]:


print('Training accuracy:',accuracy_score(y_train,model11.predict(X_train)))


# # Extratreeclassifier with Smote

# In[118]:


from sklearn.ensemble import ExtraTreesClassifier 

model12=ExtraTreesClassifier()
model12.fit(X_train_res,y_train_res)
y_pred12=model12.predict(X_test)


# In[119]:


from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
print('Testing accuracy:',accuracy_score(y_test,y_pred12))
print(confusion_matrix(y_test,y_pred12))
print(classification_report(y_test,y_pred12))
n_errors_ext_SM=print((y_pred12!=y_test).sum())


# In[120]:


cohen_kappa_score(y_test,y_pred12) 


# In[121]:


print('Training accuracy:',accuracy_score(y_train,model12.predict(X_train)))


# # SUPPORT VECTOR MACHINE With SMOTE

# In[122]:


from sklearn.svm import SVC


# In[123]:


model13=SVC()
model13.fit(X_train_res,y_train_res)


# In[124]:


y_pred13=model13.predict(X_test)


# In[125]:


from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

print('Testing accuracy:',accuracy_score(y_test,y_pred13))
print(confusion_matrix(y_test,y_pred13))
print(classification_report(y_test,y_pred13))


# In[126]:


n_errors_svc_SM=print((y_pred13!=y_test).sum())
cohen_kappa_score(y_test,y_pred13)


# In[127]:


print('Training accuracy:',accuracy_score(y_train,model13.predict(X_train)))


# # Neural_Networks With Smote

# In[128]:


from sklearn.neural_network import MLPClassifier 

model14=MLPClassifier(hidden_layer_sizes=(5,5))
model14.fit(X_train_res,y_train_res)
y_pred14=model14.predict(X_test)


# In[129]:


from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

print('Testing accuracy:',accuracy_score(y_test,y_pred14))
print(confusion_matrix(y_test,y_pred14))
print(classification_report(y_test,y_pred14))


# In[130]:


n_errors_nn_SM=print((y_pred14!=y_test).sum())
cohen_kappa_score(y_test,y_pred14)


# In[131]:


print('Training accuracy:',accuracy_score(y_train,model14.predict(X_train)))


# # Bagging classifier with Smote

# In[132]:


from sklearn.ensemble import BaggingClassifier 

model15=BaggingClassifier(DecisionTreeClassifier(criterion='entropy'))
model15.fit(X_train_res,y_train_res)


# In[133]:


y_pred15=model15.predict(X_test)


# In[134]:


from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

print('Testing accuracy:',accuracy_score(y_test,y_pred15))
print(confusion_matrix(y_test,y_pred15))
print(classification_report(y_test,y_pred15))


# In[135]:


n_errors_BC_SM=print((y_pred15!=y_test).sum())
cohen_kappa_score(y_test,y_pred15) 


# In[136]:


print('Training accuracy:',accuracy_score(y_train,model15.predict(X_train)))


# # Extreme Grdient Boosting Algorithm after smoot

# In[137]:


import xgboost as xgb 

model16=xgb.XGBClassifier()
model16.fit(X_train_res,y_train_res)


# In[138]:


y_pred16=model16.predict(X_test)


# In[139]:


from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

print('Testing accuracy:',accuracy_score(y_test,y_pred16))
print(confusion_matrix(y_test,y_pred16))
print(classification_report(y_test,y_pred16))


# In[140]:


n_errors_BC_SM=print((y_pred16!=y_test).sum())
cohen_kappa_score(y_test,y_pred16) #0.5220


# In[141]:


print('Training accuracy:',accuracy_score(y_train,model16.predict(X_train)))


# # Linear SVC

# In[142]:


from sklearn.svm import LinearSVC

model18= LinearSVC()

model18.fit(X_train,y_train)


# In[143]:


y_pred18=model18.predict(X_test)


# In[144]:


from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

print('Testing accuracy:',accuracy_score(y_test,y_pred18))
print(confusion_matrix(y_test,y_pred18))
print(classification_report(y_test,y_pred18))


# In[145]:


n_errors_LSVC=print((y_pred18!=y_test).sum())
cohen_kappa_score(y_test,y_pred18) 


# In[146]:


print('Training accuracy:',accuracy_score(y_train,model18.predict(X_train)))


# # Linear SVC _after smote

# In[147]:


from sklearn.svm import LinearSVC

model19= LinearSVC()

model19.fit(X_train_res,y_train_res)


# In[148]:


y_pred19=model19.predict(X_test)


# In[149]:


from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

print('Testing accuracy:',accuracy_score(y_test,y_pred19))
print(confusion_matrix(y_test,y_pred19))
print(classification_report(y_test,y_pred19))


# In[150]:


n_errors_LSVCSM=print((y_pred19!=y_test).sum())
cohen_kappa_score(y_test,y_pred19) 


# In[151]:


print('Training accuracy:',accuracy_score(y_train,model19.predict(X_train)))


# # Naive Bayes_before SMOTE

# In[152]:


from sklearn.naive_bayes import MultinomialNB

model20= MultinomialNB()

model20.fit(X_train,y_train)


# In[153]:


y_pred20=model20.predict(X_test)


# In[154]:


from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

print('Testing accuracy:',accuracy_score(y_test,y_pred20))
print(confusion_matrix(y_test,y_pred20))
print(classification_report(y_test,y_pred20))


# In[155]:


n_errors_MNB=print((y_pred20!=y_test).sum())
cohen_kappa_score(y_test,y_pred20) 


# In[156]:


print('Training accuracy:',accuracy_score(y_train,model20.predict(X_train)))


# # Naive Bayes _after smote

# In[157]:


from sklearn.naive_bayes import MultinomialNB

model21= MultinomialNB()

model21.fit(X_train_res,y_train_res)


# In[158]:


y_pred21=model21.predict(X_test)


# In[159]:


from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

print('Testing accuracy:',accuracy_score(y_test,y_pred21))
print(confusion_matrix(y_test,y_pred21))
print(classification_report(y_test,y_pred21))


# In[160]:


n_errors_MNBSM=print((y_pred21!=y_test).sum())
cohen_kappa_score(y_test,y_pred21) 


# In[161]:


print('Training accuracy:',accuracy_score(y_train,model21.predict(X_train)))


# # After analysing all the models with and without smote,
# 
# # The Final Model to be considered for deploying is Linear SVC
# # as it showed better values of accuracy precision and recall
# 

# In[ ]:




