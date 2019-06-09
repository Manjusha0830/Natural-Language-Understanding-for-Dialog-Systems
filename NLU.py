
from collections import defaultdict
import pandas as p
from sklearn import datasets, linear_model
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import re
import sys

def replaceIOBWithNum(x):
    iobChar = x[-1]
    if iobChar=='B':
        return 2
    elif iobChar =='I':
        return 1
    else:
        return 0
def namely(trainRead, testRead):

    #Reading data from NLU.train file
    #trainRead = open(r'C:\Users\manju\Desktop\Semester1\NLP\Assignment4\NLU.train','r')
    wordsList1 = trainRead.read() 
    trainRead.close()

    #Reading data from NLU.test file
    #testRead = open(r'C:\Users\manju\Desktop\Semester1\NLP\Assignment4\NLU.test','r')
    wordsList2 = testRead.read() 
    testRead.close()

    #traintextDict = dict()
    
    nameList = []
    traintextDict = list()
    trainfeatureDict = defaultdict(list)
    testtextDict = list()
    testfeatureDict = defaultdict(list)

    for nlupara in wordsList1.split("\n\n"):
        
        if("<class" in nlupara):
            textValue = ''
            text = nlupara.split("\n")
            text[0] = re.sub('([.,!?()])', r' \1 ', text[0])  
            text[0] = re.sub('\s{2,}', ' ', text[0])    
            textValue= text[0].split(" ")
            while "" in textValue: #remove anything that is just ""
                textValue.remove("")
            for i in range(0, len(textValue)):
                #feature1 : check if 1st character is capital
                if(textValue[i].istitle()):
                    trainfeatureDict['Capital'].append(1)
                else:
                    trainfeatureDict['Capital'].append(0)
                
                #feature2: check if numeric
                if(textValue[i].isnumeric()):
                    trainfeatureDict['Numeric'].append(1)
                else:
                    trainfeatureDict['Numeric'].append(0)
                #feature3: Length of Token
                trainfeatureDict['Length'].append(len(textValue[i]))
                #feature4: check if uupercase
                if(textValue[i].isupper):
                    trainfeatureDict['UpperCase'].append(1)
                else:
                    trainfeatureDict['UpperCase'].append(0)
                #feature 5: token value
                trainfeatureDict['Value'].append(textValue[i])
                #feature 6: if length of left word < 3    
                if(len(textValue[i])<3):
                    trainfeatureDict['LengthBelowThree'].append(1)
                else:
                    trainfeatureDict['LengthBelowThree'].append(0)
                # feature 7: checks if Right word contains any "."
                if('.' in textValue[i]):
                    trainfeatureDict['Period_in_token'].append(1) 
                else:
                    trainfeatureDict['Period_in_token'].append(0)
                #feature 8: if token starts with vowel
                if textValue[i] in 'aeiou':
                    trainfeatureDict['Vowel'].append(1) 
                else:
                    trainfeatureDict['Vowel'].append(0)

            if text[1].startswith("<class"):
                start = "<class"
                end = ">"
                data = centerdata(start, end, nlupara)
                data=data.split("\n")
                for i in range(0,len(data)):
                    if("id=" in data[i]):
                        idValue = data[i].split("=")[1]
                    if("name=" in data[i]):
                        nameValue = data[i].split("=")[1]
                        nameList = nameValue.split(" ")
            #traintextList = textValue.split(" ")
            for i in range(0, len(textValue)):
                for j in range(0,len(nameList)):
                    if(textValue[i] == nameList[j]):
                        if(j==0):
                            textValue[i]=nameList[j]+("/B")
                        elif(j>0):
                            textValue[i]=nameList[j]+("/I")
                if(textValue[i] == idValue):
                    textValue[i]=textValue[i]+("/B")
                
                else:
                    textValue[i]=textValue[i]+("/O")
                trainfeatureDict['IOB'].append(textValue[i])
            

    


    for nlupara in wordsList2.split("\n\n"):
        if("<class" in nlupara):
            testtextValue = ''
            testtext = nlupara.split("\n")
            testtext[0] = re.sub('([.,!?()])', r' \1 ', testtext[0])  
            testtext[0] = re.sub('\s{2,}', ' ', testtext[0])    
            testtextValue= testtext[0].split(" ")
            while "" in testtextValue: #remove anything that is just " "
                testtextValue.remove("")
            for i in range(0, len(testtextValue)):
                #feature1 : check if 1st character is capital
                
                if(testtextValue[i].istitle()):
                    testfeatureDict['Capital'].append(1)
                else:
                    testfeatureDict['Capital'].append(0)
                
                #feature2: check if numeric
                if(testtextValue[i].isnumeric()):
                    testfeatureDict['Numeric'].append(1)
                else:
                    testfeatureDict['Numeric'].append(0)
                #feature3: Lengt
                # h of Token
                testfeatureDict['Length'].append(len(testtextValue[i]))
                #feature4: check if uupercase
                if(testtextValue[i].isupper):
                    testfeatureDict['UpperCase'].append(1)
                else:
                    testfeatureDict['UpperCase'].append(0)
                #feature 5: token value
                testfeatureDict['Value'].append(testtextValue[i])
                #feature 6: if length of word < 3    
                if(len(testtextValue[i])<3):
                    testfeatureDict['LengthBelowThree'].append(1)
                else:
                    testfeatureDict['LengthBelowThree'].append(0)
                # feature 7: checks if  word contains any "."
                if('.' in testtextValue[i]):
                    testfeatureDict['Period_in_token'].append(1) 
                else:
                    testfeatureDict['Period_in_token'].append(0)
                #feature 8: if token starts with vowel
                if testtextValue[i] in 'aeiou':
                    testfeatureDict['Vowel'].append(1) 
                else:
                    testfeatureDict['Vowel'].append(0)

            if testtext[1].startswith("<class"):
                start = "<class"
                end = ">"
                data = centerdata(start, end, nlupara)
                data=data.split("\n")
                for i in range(0,len(data)):
                    if("id=" in data[i]):
                        idValue = data[i].split("=")[1]
                    if("name=" in data[i]):
                        nameValue = data[i].split("=")[1]
                        nameList = nameValue.split(" ")
            
            for i in range(0, len(testtextValue)):
                for j in range(0,len(nameList)):
                    if(testtextValue[i] == nameList[j]):
                        if(j==0):
                            testtextValue[i]=nameList[j]+("/B")
                        elif(j>0):
                            testtextValue[i]=nameList[j]+("/I")
                if(testtextValue[i] == idValue):
                    testtextValue[i]=testtextValue[i]+("/O")
                
                else:
                    testtextValue[i]=testtextValue[i]+("/O")
                testfeatureDict['IOB'].append(testtextValue[i])
    
        
    trainData =p.DataFrame.from_dict(trainfeatureDict)
    testData =p.DataFrame.from_dict(testfeatureDict)
    
    
    trainData['Value'] = trainData.index
    testData['Value'] = testData.index
    trainData['IOB'] = trainData['IOB'].map(replaceIOBWithNum)
    
    testData['IOB'] = testData['IOB'].map(replaceIOBWithNum )
    
    
    features = ['Value','Capital','Numeric','Length','UpperCase','LengthBelowThree','Period_in_token','Vowel']
    X = trainData[features] 
    y = trainData['IOB']
    test_data_X = testData[features]
    test_data_y = testData['IOB']
    X_train,  X_test, y_train, y_test = train_test_split(X, y )

    # fit a model
    '''lm = linear_model.LinearRegression()
    model = lm.fit(X_train, y_train)
    predictions = lm.predict(X_test)
    print(predictions[0:5])
    Accuracy = model.score(X_test, y_test) * 100'''

    # Create Decision Tree classifer object
    clf = DecisionTreeClassifier()

# Build Decision Tree Classifer from train set X and Y
    
    clf = clf.fit(X_train,y_train)
#predict class for X 
   
  
    test_pred=clf.predict(test_data_X)
    
    testaccuracy=accuracy_score(test_data_y,test_pred)
    print("Accuracy:",testaccuracy*100)
    f = open('NLU.test.out','w')
    for i in range(0,len(test_pred)):
        f.write('\n')
        if(test_pred[i] == 0):
            f.write(str(testfeatureDict['Value'][i]))
            f.write('/O')
        elif(test_pred[i] == 1):
            f.write(str(testfeatureDict['Value'][i]))
            f.write('/I')
        else:
            f.write(str(testfeatureDict['Value'][i]))
            f.write('/B')
                
    f.close()
    
def centerdata(start, end, between):
	centertext = ""
	if between.find(start):
		startword = between[between.find(start):between.rfind(end)]
		centertext = startword[len(start):]
		return centertext   
    
def main():
       
    input1 = sys.argv[1]
    input2 = sys.argv[2]
    trainfile = open(input1)
    testfile = open(input2)
    namely(trainfile,testfile)
    
    
    
    
    
        
if __name__ == "__main__":
    main()      
