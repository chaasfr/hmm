# coding=utf-8
import numpy
import random
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
	goodWord,badWord,goodSentence, badSentence = m.accuracyEval(sequenceTestX,sequenceTestY)
	accuracyWord=goodWord/float(goodWord+badWord)
	accuracySentence=goodSentence/float(goodSentence+badSentence)
	# print("words correctly estimated: ", goodWord)
	# print("words not correctly estimated: ", badWord)
	# print("accuracy on words: ", accuracyWord)
	# print("accuracy on sentence:")
	# print("sentences correctly estimated: " ,goodSentence)
	# print("sentences not correctly estimated: ", badSentence)
	# print("accuracy on sentences: ", accuracySentence)
	return accuracyWord,accuracySentence

#choisie une portion aléatoire de taille corpusLength du corpus d'apprentissage pour le training.
def selectTrainRandom(Xtrain,YTrain,shuffler,corpusLength):
	#Randomization du trainset.
	Xtaken=[]
	Ytaken=[]
	random.shuffle(shuffler) # shuffle pour prendre des élements aléatoires. On ne shuffle pas X et Y pour garder la correspondance X(i) correspond à Y(i)
	for k in range(corpusLength):
		Xtaken.append(Xtrain[shuffler[k]])
		Ytaken.append(Ytrain[shuffler[k]])
	return(Xtaken,Ytaken)	

#train suivant le modèle génératif
def runGeneratif(Xtaken, Ytaken):
	nb_symboles= numpy.max([numpy.max(U) for U in Xtaken])
	nb_states = numpy.max([numpy.max(U) for U in Ytaken])
	trainer = HMMTrainer(range(nb_states), range(nb_symboles))
	m= trainer.modeleGeneratif(Xtaken, Ytaken)
	return m

#train suivant le modèle discriminant
def runDiscriminant(Xtaken,Ytaken,discriminantIteration):
	nb_symboles= 27143
	nb_states = 15
	trainer = HMMTrainer(range(nb_states), range(nb_symboles))
	m= trainer.modeleDiscriminant(Xtaken,Ytaken,discriminantIteration,nb_states,nb_symboles,1)
	return m

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
def runCorpus(Xtrain, YTrain, sequenceTestX, sequenceTestY, corpusSizeMin, corpusSizeMax, corpusStep, runNumber, trainModel, testCorpus, discriminantIteration=None):
	if (trainModel==0):
		print "training suivant le modèle génératif"
	elif (trainModel==1):
		print "training suivant le modèle discriminant"
	else:
		print "please select trainModel 0 for generative or 1 for discriminant"
		return
	if (testCorpus==0):
		print "tests sur le corpus d'entrainement"
	else:
		print "test sur un set de test"

	if (corpusSizeMax<corpusSizeMin):
		print "please select an appropriate size"
		return

	corpusRange= corpusSizeMax - corpusSizeMin
	stepMax =corpusRange /corpusStep +1
	accuracyWord=numpy.zeros((stepMax,runNumber),float)
	accuracySentence=numpy.zeros((stepMax,runNumber),float)
	if discriminantIteration==None:
		discriminantIteration=1;

	for j in range(1,stepMax +1):
		if(j*corpusStep<=corpusSizeMax):
			message="on passe à des corpus de " + str(j*corpusStep) +" pourcents du corpus total."
			print(message)
			corpusLength=int(round(corpusStep*j*len(Xtrain)/100))
			shuffler= list(range(0, len(Xtrain))) # listes des indices de X et Y.
			for i in range (0,runNumber): #nbr de run par tranche de corpus
				print("lancement du run ", i+1, "avec j=",j)
				Xtaken,Ytaken = selectTrainRandom(Xtrain,YTrain,shuffler,corpusLength)
				if(trainModel==0):
					m=runGeneratif(Xtaken, Ytaken)
				elif (trainModel==1):
					m=runDiscriminant(Xtaken,Ytaken,discriminantIteration)

				if (testCorpus==0):
					sequenceTestX=Xtaken
					sequenceTestY=Ytaken
				aW,aS=evalAlgo(m,sequenceTestX,sequenceTestY)
				accuracyWord[j-1,i]=aW
				accuracySentence[j-1,i]=aS
				numpy.savetxt("raccuracyWord.csv",accuracyWord,delimiter=",")
				numpy.savetxt("accuracySentence.csv",accuracySentence,delimiter=",")

Xtrain,Ytrain=lireData("Data/ftb.train.encode")
Xdev,Ydev=lireData("Data/ftb.dev.encode")

#teste un entrainement avec comme parametre:
#training sur (Xtrain, Ytrain)
#training sur des corpus d'entrainement allant de 2% à 5% par saut de 2%
# 4 tests par pallier de taille de corpus
# training en modele generatif
# test sur le trainingset
corpusSizeMin=2
corpusSizeMax=5
corpusStep=2
runNumber=4
runCorpus(Xtrain,Ytrain, Xdev, Ydev, corpusSizeMin, corpusSizeMax, corpusStep, runNumber, 1, 0)