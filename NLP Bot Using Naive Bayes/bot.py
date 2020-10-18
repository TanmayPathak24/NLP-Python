import json
from nltk.corpus import stopwords
import re
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import copy
from nltk.stem.porter import PorterStemmer
import random
# import nltk
# nltk.download('punkt')

class Dexter:
    repositoryJsonPath = 'repository.json'
    probabilityJsonPath = 'probability.json'
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()
    probabilityJsonFile = None
    
    # Private Method
    def __openJsonRepository(self):
        # Open JSON file
        # Opening JSON in read mode
        try:
            file = open(self.repositoryJsonPath,)
            return file
        except FileNotFoundError:
            print("Repository Not Found At : {}".format(self.repositoryJsonPath))
            return None
        except:
            print("Something Went Wrong")
            return None

    def __parseJson(self, file):
        JsonObject = json.load(file)

        # close the file object
        file.close()

        return JsonObject

    def __generateRepositoryProbability(self, jsonObject):
        probabilityResult = {}

        # adding class list
        probabilityResult['class'] = {}

        total_input_sentences = self.__totalInputSentenceSample(jsonObject)
        classList = jsonObject['data']
        for classObject in classList:
            classType = classObject['class']
            inputSentence = classObject['input']

            inputSentenceToken = self.__FilterSentences(inputSentence)

            classProbability = self.__probability(len(inputSentenceToken), total_input_sentences)
            probabilityResult['class'][classType] = classProbability
            probabilityResult[classType] = self.__wordProbabilityInClass(inputSentenceToken)
        
        # dumping the result
        self.__dumpProbabilityResult(probabilityResult)
        
    
    def __wordProbabilityInClass(self, tokenList):
        tokenFrequency = {}
        totalToken = 0
        for sentenceToken in tokenList:
            for token in sentenceToken:
                if token in tokenFrequency:
                    # update the count
                    tokenFrequency[token] += 1
                else:
                    # make new entry
                    tokenFrequency[token] = 1
                    totalToken += 1

        tokenProb = {}
        for token in tokenFrequency.keys():
            tokenProb[token] = (tokenFrequency[token] / totalToken)
        
        return tokenProb



    def __dumpProbabilityResult(self, result = {}):
        processedJson = json.dumps(result)
        file = open("probability.json", "w+")
        file.write(processedJson)
        file.flush()
        file.close()


            
    def __totalInputSentenceSample(self, jsonObject):
        total = 0
        for x in jsonObject['data']:
            total += len(x['input'])
        return total

    def __probability(self, numerator, denomenator = 1):
        return (numerator / denomenator)

    def __FilterSentences(self, sentenceList):
        result = []
        for sentence in sentenceList:
            result.append(self.__filterSentence(sentence))
        return result
            
    def __filterSentence(self, sentence):
        sentence = copy.copy(sentence)

        sentence = sentence.lower()

        # remove whitespaces
        sentence = re.sub(r' +',' ', sentence)

        # remove numbers from sentence
        sentence = re.sub(r'\d+', '', sentence)

        # remove punctuations
        sentence = re.sub(r'[^\w\s]', '', sentence)

        # tokenization 
        sentenceToken = word_tokenize(sentence)

        # Stemming
        # process of converting word to its root word
        sentenceToken = [self.stemmer.stem(word) for word in sentenceToken]

        # Lemmatization of sentence
        sentenceToken = [self.lemmatizer.lemmatize(word) for word in sentenceToken]

        return sentenceToken

    def loadRepository(self):
        file = self.__openJsonRepository()
        jsonObject = self.__parseJson(file)
        self.__generateRepositoryProbability(jsonObject)
        self.probabilityJsonFile = open(self.probabilityJsonPath,)

    
    # predict the response message
    def chat(self, message):
        if self.probabilityJsonFile  is None:
            self.loadRepository()
        
        messageToken = self.__filterSentence(message)

        probJson = json.load(self.probabilityJsonFile)
        messageClassProb = {}
        maxProbability = 0.0
        maxProbabilityClass = None

        for classType in probJson['class'].keys():
            classProb = probJson['class'][classType]
            classWrodProb = probJson[classType]
            totalClassToken = len(probJson[classType].keys())
            tokenTotalProbability = 1
            for token in messageToken:
                if token in classWrodProb.keys():
                    tokenTotalProbability *= classWrodProb[token]
                else:
                    tokenTotalProbability *= self.__probability(1, totalClassToken)

            # message to be a class type
            totalProbability = classProb * tokenTotalProbability

            messageClassProb[classType] = totalProbability

            if totalProbability > maxProbability:
                maxProbability = totalProbability
                maxProbabilityClass = classType
        
        print("Message Class Prob : {}".format(messageClassProb))
        print("Message Class Type : {}".format(maxProbabilityClass))

        messageRepositoryFile = open('repository.json',)
        messageRepoJson = json.load(messageRepositoryFile)
        for classJson in messageRepoJson['data']:
            if classJson['class'] == maxProbabilityClass:
                classOutputMessages = classJson['output']
                index = random.randint(0, len(classOutputMessages)-1)
                return classOutputMessages[index]

        


bot = Dexter()
bot.loadRepository()
while(True):
    inputMessage = input("USER : ")
    output = bot.chat(inputMessage)
    print("BOT : {}".format(output))
    if output == 'exit':
        break
    
