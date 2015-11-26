# coding=utf-8
import numpy
import random
import matplotlib.pyplot as plt
from hmm import HMMTrainer
from hmm import HMM
from random import shuffle

#**********************
# Lecture des donnees
# Retourne deux listes X et Y avec:
# X la liste des listes de symboles
# Y la liste des listes d'états
#**********************
def lireData(sourceFile):
    source = open(sourceFile, "r")
    toutesleslignes = source.readlines()
    l=len(toutesleslignes)
    X=[]
    Y=[]
    sequenceX = []
    sequenceY = []
    for i in range(l):
        r = toutesleslignes[i].rstrip('\n\r')
        #print (r)
        if r=='':       
            X.append(sequenceX)
            sequenceX =[]
            Y.append(sequenceY)
            sequenceY =[]
        else:
            r = r.split(' ')
            sequenceX.append(int(r[0]))
            sequenceY.append(int(r[1]))
    return(X,Y)

#test la précision du training
def evalAlgo(m, sequenceTestX,sequenceTestY):
	goodWordNumber,badWordNumber, wordWrong, goodSentenceNumber, badSentenceNumber = m.accuracyEval(sequenceTestX,sequenceTestY)
	accuracyWord=goodWordNumber/float(goodWordNumber+badWordNumber)
	accuracySentence=goodSentenceNumber/float(goodSentenceNumber+badSentenceNumber)
	return accuracyWord,accuracySentence, wordWrong

#choisie une portion aléatoire de taille corpusLength du corpus d'apprentissage pour le training.
def selectTrainRandom(Xtrain,Ytrain,corpusLength):
	#Randomization du trainset.
	Xtaken=[]
	Ytaken=[]
	shuffler= list(range(0, len(Xtrain))) # listes des indices de X et Y.
	random.shuffle(shuffler) # shuffle pour prendre des élements aléatoires. On ne shuffle pas X et Y pour garder la correspondance X(i) correspond à Y(i)
	for k in range(corpusLength):
		Xtaken.append(Xtrain[shuffler[k]])
		Ytaken.append(Ytrain[shuffler[k]])
	return(Xtaken,Ytaken)	

def runCorpus(corpusSizeMin, corpusSizeMax, corpusStep, runNumber, isTrainedOnPerceptron, isTestOnTrainingSet, epsilon=None, discriminantIteration=1, perceptronMoyenne=False):
	#fonction principale, lance une simulation avec les parametres suivants:
#Xtrain: liste de liste de symboles pour le training
#Ytrain: liste de liste de catégorie étiquettant Xtrain
#sequenceTestX: liste de liste de symbole pour tester le modèle
#sequenceTestY: liste de liste de catégorie étiquettant sequenceTestX
#corpusSizeMin: taille minimale du corpus d'apprentissage (i.e. quel pourcentage de Xtrain est pris pour train le HMM au minimum). En pourcentage.
#corpusSizeMax: cf corpusSizeMax en remplaçant min par max
#corpusStep: le pas (en pourcent) entre chaque apprentissage. Exemple: corpusSizeMin= 5, corpusSizeMax=100, corpusStep=5 <=> training d'un HMM sur Xtrain en prenant 5%, puis 10%...puis 100% de Xtrain
#runNumber: combien de run faire pour chaque pallier de taille du corpus.
#trainModel: 0 pour modèle génératif, 1 pour modèle discriminant
#testCorpus: 0 pour tester sur le train set
#discriminantIteration: combien d'iteration pour le training dans le cas du modèle discriminant
	if (isTrainedOnPerceptron):
		if (perceptronMoyenne):
			print "training suivant le modèle discriminant avec perceptron moyenne"
		else:
			print "training suivant le modèle discriminant avec perceptron NON moyenne"
	else:
		print "training suivant le modèle génératif"
	if (isTestOnTrainingSet):
		print "tests sur le corpus d'entrainement"
		nbSymbolesTest=0
		nbStatesTest=0
	else:
		print "test sur un set de test"
		sequenceTestX,sequenceTestY=lireData("Data/ftb.dev.encode")
		nbSymbolesTest= numpy.max([numpy.max(U) for U in sequenceTestX]) + 1
		nbStatesTest= numpy.max([numpy.max(U) for U in sequenceTestY]) + 1

	if (corpusSizeMax<corpusSizeMin):
		print "please select an appropriate size"
		return

	Xtrain,Ytrain=lireData("Data/ftb.train.encode")

	nbSymbolesTrain= numpy.max([numpy.max(U) for U in Xtrain]) + 1
	nbStatesTrain = numpy.max([numpy.max(U) for U in Ytrain]) + 1 
	nbSymboles=max(nbSymbolesTrain,nbSymbolesTest)
	nbStates=max(nbStatesTrain,nbStatesTest)
	trainer = HMMTrainer(range(nbStates), range(nbSymboles))
	corpusRange= corpusSizeMax - corpusSizeMin
	stepMax =corpusRange /corpusStep +1
	accuracyWord=numpy.zeros((stepMax,runNumber),float)
	accuracySentence=numpy.zeros((stepMax,runNumber),float)
	wordWrong=numpy.zeros(nbSymboles, float)
	wordWrongTemp=numpy.zeros(nbSymboles, float)

	for j in range(0,stepMax):
		if(corpusSizeMin+j*corpusStep>0):
			message="on passe à des corpus de " + str(corpusSizeMin + j*corpusStep) +" pourcents du corpus total."
			print(message)
			corpusLength=int((corpusSizeMin + corpusStep*j)/100.0*len(Xtrain))
			print(corpusLength)
			for i in range (0,runNumber): #nbr de run par tranche de corpus
				print("lancement du run ", i+1, "avec j=",j)
				Xtaken,Ytaken = selectTrainRandom(Xtrain,Ytrain,corpusLength)

				if(isTrainedOnPerceptron):
					m= trainer.modeleDiscriminant(Xtaken,Ytaken,discriminantIteration,nbStates,nbSymboles,epsilon, perceptronMoyenne)
					# m=HMM(0,0,0,0,0,1)
					# m.load("hmm.pkl")
					accuracyW="accuracyW-itModel"+str(discriminantIteration)+"-epsilon"+str(epsilon)+".csv"
					accuracyS="accuracyS-itModel"+str(discriminantIteration)+"-epsilon"+str(epsilon)+".csv"
				else:
					m= trainer.modeleGeneratif(Xtaken, Ytaken)
					accuracyW="accuracyWGen.csv"
					accuracyS="accuracySWGen.csv"
				if (isTestOnTrainingSet):
					sequenceTestX=Xtaken
					sequenceTestY=Ytaken
				aW, aS, wordWrongTemp=evalAlgo(m,sequenceTestX,sequenceTestY)
				wordWrong= wordWrong + wordWrongTemp
				accuracyWord[j,i]=aW
				accuracySentence[j,i]=aS
				numpy.savetxt(accuracyW,accuracyWord,delimiter=",")
				numpy.savetxt(accuracyS,accuracySentence,delimiter=",")
	numpy.savetxt("wrongWordFrequencies.csv",wordWrong,delimiter=",")
	m.save("hmm.pkl")


#teste un entrainement avec comme parametre:
#training sur (Xtrain, Ytrain)
#training sur des corpus d'entrainement allant de 2% à 5% par saut de 2%
# 4 tests par pallier de taille de corpus
# training en modele generatif
# test sur le trainingset
corpusSizeMin=100
corpusSizeMax=100
corpusStep=100
runNumber=5
isTrainedOnPerceptron= True
isTestOnTrainingSet= False
epsilon=0.1 # valeur d'epsilon par defaut
perceptronMoyenne= False # Fonctionne uniquement si isTrainedOnPerceptron est à True

# for epsilon in [0.01,0.03,0.1,0.3,1,3]: # pas du gradient pour le modèle discriminant, on cherche à voir si la valeur du pas de gradient permet d'obtenir de meilleurs résultats ou non
# for iterationModel in [13,13]:#,3,5,8,13,21,34,53]: # suite de Fibonacci pour balayer un nombre de valeurs assez écartées
iterationModel=13
runCorpus(corpusSizeMin, corpusSizeMax, corpusStep, runNumber, isTrainedOnPerceptron, isTestOnTrainingSet, epsilon, iterationModel, perceptronMoyenne) # Rajouter epsilon et iterationModel pour mon cas