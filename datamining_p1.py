#manu gupta 1001599943
import os
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from math import log10, sqrt

filename = './debate.txt'
i = 0
stemmer = PorterStemmer()
tokenized_list = []
para = ""



def paratoken(para):
    tokenizer = RegexpTokenizer(r'[a-zA-Z]+')  # tokenize the paragraph
    tokens = tokenizer.tokenize(para)
    # stemming and removing the stop words simultaneously

    stop_words = set(stopwords.words('english'))
    filtered_sentence = [w for w in tokens if not w in stop_words]      #remove the stop words from the file
    filtered_sentence = []
    for w in tokens:
        if w not in stop_words:
            filtered_sentence.append(stemmer.stem(w))                   #stem and store it and return
    return filtered_sentence

#open and read the file "debate.txt"
with open(filename, "r") as input:
    para = input.read().split("\n\n")                                       #split the file into lists of lists, to see them as differernt document


while i < len(para):                                                    #tokenizing the file
    tokenized_list.append(paratoken(para[i]))
    i += 1
j = 0



#calculate the idf of each token
def getidf(token):
    doc_frequency = []
    doc_count = 0
    token_count = 0
    for x in tokenized_list:
        doc_count += 1
        for words in range(len(x)):                 #check if token is present in the tokenized-list of the file
            if x[words] == token:
                token_count += 1
            if words == len(x) - 1:
                # print("break")
                if token_count != 0:                   #append the list only if the token is found, token_count>0
                    doc_frequency.append([token_count, doc_count])
                    token_count = 0

                                                        # if list is empty , no match , return idf=-1, else apply the formula and return idf
    if len(doc_frequency) == 0:
        idf = -1
    else:
        idf = log10(len(para) / len(doc_frequency))

    return idf

                                                        #function to find query vector
def getqvec(qstring):                                   #accept the query string
    #print("idhar")
    i = 0.0
    qstring = qstring.lower()                           #convert in to lower case
    tokenizer = RegexpTokenizer(r'[a-zA-Z]+')           #tokenize it
    tokens = tokenizer.tokenize(qstring)
    stopset = set(stopwords.words('english'))
    d1 = [w for w in tokens if not w in stopset]        #remove the stopwords
    stemmer = PorterStemmer()
    d1 = [stemmer.stem(d) for d in d1]

    idfMap = {}
    for term in d1:                                     #find idf of each term
        idf = getidf(term)
        if idf == -1:                                   #if idf=-1, take it as 1 and recalculate by log(no. of files)/1
            idf = log10(len(tokenized_list)) / 1
        idfMap[term] = idf

    sum = 0
    normalizedIdfMap = {}
    for key in idfMap:                                     # the normalization of the idf recieved
        sum = sum + idfMap[key] * idfMap[key]               #sum of all the idf recieved in the file and square root it

    normalizedLength = sqrt(sum)
    for key in idfMap:
        normalizedIdfMap[key] = idfMap[key]/normalizedLength    #divide the rooted value to the idf of the required value

    return  normalizedIdfMap                                    #return the final normalized idf






def query(qstring):                 #accept the query the string

    tempList = []                       #calculate the tf idf for ach term in every document
    for terms in tokenized_list:
        tempListMap = {}
        for words in terms:
            idf1 = getidf(words)
            if idf1 == -1:
                idf1 = log10(len(tokenized_list)) / 1
            tempListMap[words] = idf1
        tempList.append(tempListMap)


                                                #normalize the value of the tf-idf recieved from above
    normalizedmap = {}
    normalizedlenMap = {}
    i =0
    for document in tempList:
        sum = 0
        for term in document:

            sum = sum + document[term] * document[term]
        normalizedlen = sqrt(sum)
        normalizedlenMap[i] = normalizedlen
        i = i + 1



    tfIDFList = []
    i = 0
    for document in tempList:
        tempMap = {}
        for term in document:
            val = document[term]/ normalizedlenMap[i]
            tempMap[term] = val
        tfIDFList.append(tempMap)
        i = i + 1
    print("idf for Document")
    print(tfIDFList)


    idf_qstring = getqvec(qstring)              #normalize along with tf-idf of the query string given
    cosPreValList = []
    for document in tfIDFList:
        val = 0
        for queryTerm in idf_qstring:               #calculate cost of each document for cosine similarity
            if queryTerm in document:
                val = val + (idf_qstring[queryTerm] * document[queryTerm])
        cosPreValList.append(val)


    highestValue = max(cosPreValList)           #the document with highest value prints along with the value
    if highestValue > 0.00:
        highestValueIndex = cosPreValList.index(highestValue)
        return para[highestValueIndex], highestValue
    else:
        str =""
        return str,highestValue





#print("%s%.4f" % query("The alternative, as cruz has proposed, is to deport 11 million people from this country"))
#print("%s%.4f" % query("unlike any other time, it is under attack"))
#print("%s%.4f" % query("vector entropy"))
#print("%s%.4f" % query("clinton first amendment kavanagh"))
#print(getqvec("unlike any other time, it is under attack"))
#print("%.4f" % getidf(stemmer.stem("immigration")))