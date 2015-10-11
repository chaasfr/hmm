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

Xtrain,Ytrain=lireData("Data/ftb.train.encode")
Xdev,Ydev=lireData("Data/ftb.dev.encode")


# Xtrain=[[1, 2, 1, 1, 3, 5, 6, 7, 1, 1, 2], [2, 1, 0, 3, 1, 4, 7, 6, 2, 0, 1] , [0, 1, 1, 2, 5, 7, 7, 7, 1, 1, 1]]
# Ytrain=[[0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0], [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0],  [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0]]
# nb_states = 2
# nb_symboles = 8
# Xdev=Xtrain
# Ydev=Ytrain
goodWordTrain=numpy.zeros((20,50),float)
goodSentenceTrain=numpy.zeros((20,50),float)
goodWordDev=numpy.zeros((20,50),float)
goodSentenceDev=numpy.zeros((20,50),float)
badWordTrain=numpy.zeros((20,50),float)
badSentenceTrain=numpy.zeros((20,50),float)
badWordDev=numpy.zeros((20,50),float)
badSentenceDev=numpy.zeros((20,50),float)

for j in range(1,21):
	message="on passe à des corpus de " + str(j*5) +" pourcents du corpus total."
	print(message)
	corpus_length=int(round(0.05*j*len(Xtrain)))
	shuffler= list(range(0, len(Xtrain))) # listes des indices de X et Y.
	for i in range (1,51): #nbr de run par tranche de corpus
		print("lancement du run ", i, "avec j=",j)
		#Randomization du trainset. On train sur des tranches de j*5% du corpus
		Xtaken=[]
		Ytaken=[]
		random.shuffle(shuffler) # shuffle pour prendre des élements aléatoires. On ne shuffle pas X et Y pour garder la correspondance X(i) correspond à Y(i)
		for k in range(corpus_length):
			Xtaken.append(Xtrain[shuffler[k]])
			Ytaken.append(Ytrain[shuffler[k]])
		nb_symboles= numpy.max([numpy.max(U) for U in Xtaken])
		nb_states = numpy.max([numpy.max(U) for U in Ytaken])
		trainer = HMMTrainer(range(nb_states), range(nb_symboles))
		m= trainer.modeleGeneratif(Xtaken, Ytaken)
		#évaluation de l'algo
		# (goodWordTrain[j-1,i],badWordTrain[j-1,i-1],goodSentenceTrain[j-1,i-1],badSentenceTrain[j-1,i-1]) = m.accuracyEval(Xtaken,Ytaken)
		# print("TRAIN SET")
		# print(	"words correctly estimated: ", goodWordTrain[j-1,i-1])
		# print(	"words not correctly estimated: ", badWordTrain[j-1,i-1])
		# print(	"accuracy on words: ", goodWordTrain[j-1,i-1]/float(goodWordTrain[j-1,i-1]+badWordTrain[j-1,i-1]))
		# print("accuracy on sentence:")
		# print(	"sentences correctly estimated: ", goodSentenceTrain[j-1,i-1])
		# print(	"sentences not correctly estimated: ", badSentenceTrain[j-1,i-1])
		# print(	"accuracy on sentences: ", goodSentenceTrain[j-1,i-1]/float(goodSentenceTrain[j-1,i-1]+badSentenceTrain[j-1,i-1]))
		(goodWordDev[j-1,i-1],badWordDev[j-1,i-1],goodSentenceDev[j-1,i-1], badSentenceDev[j-1,i-1]) = m.accuracyEval(Xdev,Ydev)
		print("DEV SET")
		print(	"words correctly estimated: ", goodWordDev[j-1,i-1])
		print(	"words not correctly estimated: ", badWordDev[j-1,i-1])
		print(	"accuracy on words: ", goodWordDev[j-1,i-1]/float(goodWordDev[j-1,i-1]+badWordDev[j-1,i-1]))
		print("accuracy on sentence:")
		print(	"sentences correctly estimated: " ,goodSentenceDev[j-1,i-1])
		print(	"sentences not correctly estimated: ", badSentenceDev[j-1,i-1])
		print(	"accuracy on sentences: ", goodSentenceDev[j-1,i-1]/float(goodSentenceDev[j-1,i-1]+badSentenceDev[j-1,i-1]))
		# numpy.savetxt("goodWordTrain.csv", goodWordTrain, delimiter=",")
		# numpy.savetxt("goodSentenceTrain.csv", goodSentenceTrain, delimiter=",")
		# numpy.savetxt("badSentenceTrain", badSentenceTrain, delimiter=",")
		# numpy.savetxt("badWordTrain", badWordTrain, delimiter=",")
		numpy.savetxt("goodWordDev.csv", goodWordDev, delimiter=",")
		numpy.savetxt("goodSentenceDev.csv", goodSentenceDev, delimiter=",")
		numpy.savetxt("badSentenceDev", badSentenceDev, delimiter=",")
		numpy.savetxt("badWordDev", badWordDev, delimiter=",")


# m=HMM(0,0,[],[],[],0)
# m.load("hmm.pkl")

# l = m.viterbi(Xtrain[601])
# print('Xtrain[600]=',Xtrain[601])
# print('l=',l)
# print('Ytrain[600]=',Ytrain[601])

# l = m.viterbi(Xtrain[1])
# print('Xtrain[600]=',Xtrain[1])
# print('l=',l)
# print('Ytrain[600]=',Ytrain[1])

print("done. saving.")
#évaluation de l'algo
# (goodWordTrain,badWordTrain,goodSentenceTrain,badSentenceTrain) = m.accuracyEval(Xtrain,Ytrain)
# (goodWordDev,badWordDev,goodSentenceDev, badSentenceDev) = m.accuracyEval(Xdev,Ydev)

# print("TRAIN SET")
# print(	"words correctly estimated: ", goodWordTrain)
# print(	"words not correctly estimated: ", badWordTrain)
# print(	"accuracy on words: ", goodWordTrain/float(goodWordTrain+badWordTrain))
# print("accuracy on sentence:")
# print(	"sentences correctly estimated: ", goodSentenceTrain)
# print(	"sentences not correctly estimated: ", badSentenceTrain)
# print(	"accuracy on sentences: ", goodSentenceTrain/float(goodSentenceTrain+badSentenceTrain))

# print("DEV SET")
# print(	"words correctly estimated: ", goodWordDev)
# print(	"words not correctly estimated: ", badWordDev)
# print(	"accuracy on words: ", goodWordDev/float(goodWordDev+badWordDev))
# print("accuracy on sentence:")
# print(	"sentences correctly estimated: " ,goodSentenceDev)
# print(	"sentences not correctly estimated: ", badSentenceDev)
# print(	"accuracy on sentences: ", goodSentenceDev/float(goodSentenceDev+badSentenceDev))
# m.save("hmm.pkl")