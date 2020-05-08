from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from math import log, sqrt
import pandas as pd
import numpy as np
import os
import io
import email
import re  # regular expression. it is used to perform word searches.

def read_files(path):
    #this function is reading the actual email files.
    #the function will cycle through all the files in the folder specified.
    for root, dirnames, filenames in os.walk(path):
        for filename in filenames:
            path = os.path.join(root,filename)
            
            inBody = False
            lines = []
            #this uses IO functions to read the files by specifying the encoding type
            f = io.open(path, 'r', encoding = 'latin1')
            #this cycles through each line and fetches the text therein.
            for line in f:
                if inBody:
                    lines.append(line)
                elif line == '\n':
                    inBody = True
            f.close()
            #join the text into a single message.
            message = '\n'.join(lines)
            #return the message
            yield path, message
                    


# ### This function is for creating dataframes from the dataset in question... it also classifies the emails into either ham or spam

#this classifies the emails into spam or ham depending on predetermined status.
def dataFrameFromDirectory(path, classification):
    #instantiate the structure of the dataframe needed.
    rows = []
    index = []
    #store the message alongside its classification.
    for filename, message in read_files(path):
        rows.append({'message': message, 'label': classification})
        index.append(filename)
        #return the dataframe to be used for manipulation
    return pd.DataFrame(rows, index=index)



# The function below extracts all text from html text present in all emails.

#this is for scraping the emails, since most are in the form of html based content.
from bs4 import BeautifulSoup

def func(df):
    soup = BeautifulSoup(df['message'], "html.parser").find()
    #check emails with html syntax
    if bool(soup):
        soup = BeautifulSoup(df['message'], "html.parser")
        #extract text only from the whole email.

        text = soup.find_all(text=True)

        #return all the words found.
        text = ''.join(word for word in  text)
        df['message'] = text

        return text
    else:

        return df['message']


# def using_emailing_function(df):
#
# #     msg = email.message_from_string(df['message'])
# #     for part in msg.walk():
# #         print(df['message'])
#
#     msg = email.message_from_string(df['message'])
#     if msg.is_multipart():
#         for payload in msg.get_payload():
#             # if payload.is_multipart(): ...
# #             print(payload.get_payload())
#             soup = BeautifulSoup(payload.get_payload(), "lxml").find()
#             if bool(soup):
#                 # print(payload.get_payload())
#             else:
#                 # print(payload.get_payload())
#     else:
#         soup = BeautifulSoup(msg.get_payload(), "lxml").find()
#         if bool(soup):
#                 # print(msg.get_payload())
# #         else:
# #             print(msg.get_payload())

#tokenizing and processing the words harnessed.
def process_message(message, lower_case = True, stem = True, stop_words = True, gram = 2):
    #lower case all the letters in the message.
    if lower_case:
        message = message.lower()
    #break down the words into tokens.
    words = word_tokenize(message)
    words = [w for w in words if len(w) > 2]
#     print(words)
    if gram > 1:
        w = []
        for i in range(len(words) - gram + 1):
            w += [' '.join(words[i:i + gram])]
        return w
    if stop_words:
        #fetch all words that do not have meaning in this case: stopwords.
        sw = stopwords.words('english')
        #create a list of words containing only the words without stopwords
        words = [word for word in words if word not in sw]
    if stem:
        stemmer = PorterStemmer()
        words = [stemmer.stem(word) for word in words]   
#     print(words)
    return words


#this whole function is the bayesian classifier. the calc_prob calculates the probability of a word bein either spam or ham.
#the classify function returns true if the threshold for it being spam has been reached. otherwise, it is false, meaning ham.
class SpamClassifier(object):
    def __init__(self, train_data, method = 'tf-idf'):
        #initialize the data to be used for processing. i.e training data.
        self.mails, self.labels = train_data['message'], train_data['label']
        self.method = method

    def train(self):
        #this function trains our model.
        self.calc_TF_and_IDF()
        if self.method == 'tf-idf':
            self.calc_TF_IDF()
        else:
            self.calc_prob()

    def calc_prob(self):
        # initialize variables to calculate probabilities. i.e probability of being ham or spam. using normal probabilistic theory.
        self.prob_spam = dict()
        self.prob_ham = dict()
        for word in self.tf_spam:
            #if word is spam create a dictionary with the key being the word and the value being the probability of it being spam.
            self.prob_spam[word] = (self.tf_spam[word] + 1) / (self.spam_words +                                                                 len(list(self.tf_spam.keys())))
        for word in self.tf_ham:
            #if word is ham, create a dictionary with the key being the ham word and the value being its probability of being ham.
            self.prob_ham[word] = (self.tf_ham[word] + 1) / (self.ham_words +                                                                 len(list(self.tf_ham.keys())))
        self.prob_spam_mail, self.prob_ham_mail = self.spam_mails / self.total_mails, self.ham_mails / self.total_mails 


    def calc_TF_and_IDF(self):
        noOfMessages = self.mails.shape[0]
        self.spam_mails, self.ham_mails = self.labels.value_counts()[1], self.labels.value_counts()[0]
        self.total_mails = self.spam_mails + self.ham_mails
        self.spam_words = 0
        self.ham_words = 0
        self.tf_spam = dict()
        self.tf_ham = dict()
        self.idf_spam = dict()
        self.idf_ham = dict()
        self.missing_index = [number for number in range(0,noOfMessages) if number not in self.mails.index ]
        for i in range(noOfMessages):
            if i not in self.missing_index:
                message_processed = process_message(self.mails[i])
                count = list() #To keep track of whether the word has ocured in the message or not.
                               #For IDF
                for word in message_processed:
                    if self.labels[i]:
                        self.tf_spam[word] = self.tf_spam.get(word, 0) + 1
                        self.spam_words += 1
                    else:
                        self.tf_ham[word] = self.tf_ham.get(word, 0) + 1
                        self.ham_words += 1
                    if word not in count:
                        count += [word]
                for word in count:
                    if self.labels[i]:
                        self.idf_spam[word] = self.idf_spam.get(word, 0) + 1
                    else:
                        self.idf_ham[word] = self.idf_ham.get(word, 0) + 1

    def calc_TF_IDF(self):
        self.prob_spam = dict()
        self.prob_ham = dict()
        self.sum_tf_idf_spam = 0
        self.sum_tf_idf_ham = 0
        for word in self.tf_spam:
            self.prob_spam[word] = (self.tf_spam[word]) * log((self.spam_mails + self.ham_mails)                                                           / (self.idf_spam[word] + self.idf_ham.get(word, 0)))
            self.sum_tf_idf_spam += self.prob_spam[word]
        for word in self.tf_spam:
            self.prob_spam[word] = (self.prob_spam[word] + 1) / (self.sum_tf_idf_spam + len(list(self.prob_spam.keys())))
            
        for word in self.tf_ham:
            self.prob_ham[word] = (self.tf_ham[word]) * log((self.spam_mails + self.ham_mails)                                                           / (self.idf_spam.get(word, 0) + self.idf_ham[word]))
            self.sum_tf_idf_ham += self.prob_ham[word]
        for word in self.tf_ham:
            self.prob_ham[word] = (self.prob_ham[word] + 1) / (self.sum_tf_idf_ham + len(list(self.prob_ham.keys())))
            
    
        self.prob_spam_mail, self.prob_ham_mail = self.spam_mails / self.total_mails, self.ham_mails / self.total_mails 
       #the method that does the actual classifying.             
    def classify(self, processed_message):
        pSpam, pHam = 0, 0
        for word in processed_message:                
            if word in self.prob_spam:
                pSpam += log(self.prob_spam[word])
            else:
                try:
                    if self.method == 'tf-idf':
                        pSpam -= log(self.sum_tf_idf_spam + len(list(self.prob_spam.keys())))
                    else:
                        pSpam -= log(self.spam_words + len(list(self.prob_spam.keys())))
                except:
                    print('oops')
            if word in self.prob_ham:
                pHam += log(self.prob_ham[word])
            else:
                try:
                    if self.method == 'tf-idf':
                        pHam -= log(self.sum_tf_idf_ham + len(list(self.prob_ham.keys()))) 
                    else:
                        pHam -= log(self.ham_words + len(list(self.prob_ham.keys())))
                except:
                    print('oops')
            pSpam += log(self.prob_spam_mail)
            pHam += log(self.prob_ham_mail)
            #if probability of spam is higher, it returns true.
            return pSpam >= pHam
    
    def predict(self, testData):
        result = dict()
        for (i, message) in enumerate(testData):
            processed_message = process_message(message)
        
            result[i] = int(self.classify(processed_message))
        return result

def metrics(labels, predictions): #Confusion matrix function
    true_pos, true_neg, false_pos, false_neg = 0, 0, 0, 0
    for i in range(len(labels)):
        true_pos += int(labels[i] == 1 and predictions[i] == 1)
        true_neg += int(labels[i] == 0 and predictions[i] == 0)
        false_pos += int(labels[i] == 0 and predictions[i] == 1)
        false_neg += int(labels[i] == 1 and predictions[i] == 0)
    precision = true_pos / (true_pos + false_pos)
    recall = true_pos / (true_pos + false_neg)
    Fscore = 2 * precision * recall / (precision + recall)
    accuracy = (true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg)

    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F-score: ", Fscore)
    print("Accuracy: ", accuracy)


def main():
    # aggregate all the emails into a single dataframe. The key is: ham = 0, spam = 1

    # fetch the data from the source files.
    # here you run the datasets sequentially for validation by specifying file name.

    data = pd.DataFrame({'message': [], 'label': []})
    ham_folder = input("input ham folder name: ")
    spam_folder = input("input spam folder name: ")
    data = data.append(dataFrameFromDirectory("datasets/"+str(ham_folder), 0))
    data = data.append(dataFrameFromDirectory("datasets/"+ str(spam_folder), 1))


    # The below cell shows the number of rows in our dataframe.

    total_mails = data['message'].shape[0]
    print("total mails: ", total_mails)

    new_data = data.reset_index()

    #this removes all text that is not alphanumeric i.e special characters.
    new_data[new_data['message'].str.isalnum()]

    # split the data into the pareto principle for unbiased model testing.
    # so 80% will be used for training and 20% for testing.
    # this will be accomplished by using a random number generator to randomize the order by implementing a uniform distribution randomizer. this removes all chances of having a biased model on account of sequential data.

    # split the data into training and testing sets
    # initializing empty lists to hold training and testing data respectively
    train_index, test_index = list(), list()
    for i in range(data['message'].shape[0]):
        # make use of the uniform random distribution to alleviate contiguousness of pseudo-random number generators.
        # 80% for training and 20% for testing according to the pareto principle.
        if np.random.uniform(0, 1) < 0.80:
            train_index.append(i)
        else:
            test_index.append(i)
    # training data dataframe.
    train_data = new_data.loc[train_index]
    # test data dataframe
    test_data = new_data.loc[test_index]

    # train_data.apply(using_emailing_function, axis=1)

    # implement the data cleaning of the html based messages

    # here we  apply the scraping function to our data. hence cleaning it.
    train_data['message'] = train_data.apply(func, axis=1)
    test_data['message'] = test_data.apply(func, axis=1)
    # its just expecting the data to come from a website and not a dataframe.



    train_data['message'] = train_data['message'].map(lambda x: re.sub(r'\W+', ' ', x))
    test_data['message'] = test_data['message'].map(lambda x: re.sub(r'\W+', ' ', x))

    train_data.reset_index(drop=True, inplace=True)
    test_data.reset_index(drop=True, inplace=True)

    print("processing for bow....")
    train_data['message'].str.isalnum()
    # this checks for the metrics
    sc_bow = SpamClassifier(train_data, 'bow')
    sc_bow.train()
    preds_bow = sc_bow.predict(test_data['message'])
    metrics(test_data['label'], preds_bow)

    print("processing for tf-idf....")
    # this checks for the metrics based on tf-idf
    sc_tf_idf = SpamClassifier(train_data, 'tf-idf')
    sc_tf_idf.train()
    preds_tf_idf = sc_tf_idf.predict(test_data['message'])
    metrics(test_data['label'], preds_tf_idf)

    message = input("input the message to be classified: ")
    pm = process_message(str(message))
    if(sc_tf_idf.classify(pm)):
        print("The message is spam")
    else:
        print("The message is ham")


if __name__ == '__main__':
    main()