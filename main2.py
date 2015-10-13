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
nb_symboles= numpy.max([numpy.max(U) for U in Xtrain])
nb_states = numpy.max([numpy.max(U) for U in Ytrain])
print nb_states

goodWordTrain=numpy.zeros((20,15),float)
goodSentenceTrain=numpy.zeros((20,15),float)
badSentenceTrain=numpy.zeros((20,15),float)
badWordTrain=numpy.zeros((20,15),float)
accuracyWordTrain=numpy.zeros((20,15),float)
accuracySentenceTrain=numpy.zeros((20,15),float)

# goodWordDev=numpy.zeros((20,15),float)
# goodSentenceDev=numpy.zeros((20,15),float)
# badWordDev=numpy.zeros((20,15),float)
# badSentenceDev=numpy.zeros((20,15),float)
# accuracyWordDev=numpy.zeros((20,15),float)
# accuracySentenceDev=numpy.zeros((20,15),float)

iteration = 10

for j in range(8,21):
	message="on passe à des corpus de " + str(j*5) +" pourcents du corpus total."
	print(message)
	corpus_length=int(round(0.05*j*len(Xtrain)))
	shuffler= list(range(0, len(Xtrain))) # listes des indices de X et Y.
	for i in range (1,6): #nbr de run par tranche de corpus
		print("lancement du run ", i, "avec j=",j)
		#Randomization du trainset. On train sur des tranches de j*5% du corpus
		Xtaken=[]
		Ytaken=[]
		random.shuffle(shuffler) # shuffle pour prendre des élements aléatoires. On ne shuffle pas X et Y pour garder la correspondance X(i) correspond à Y(i)
		for k in range(corpus_length):
			Xtaken.append(Xtrain[shuffler[k]])
			Ytaken.append(Ytrain[shuffler[k]])

		#modèle génératif:
		nb_symboles= numpy.max([numpy.max(U) for U in Xtaken])
		nb_states = numpy.max([numpy.max(U) for U in Ytaken])
		print nb_states
		trainer = HMMTrainer(range(nb_states), range(nb_symboles))
		m= trainer.modeleGeneratif(Xtaken, Ytaken)

		#modèle discriminant:
		# nb_symboles= 27143
		# nb_states = 15
		# trainer = HMMTrainer(range(nb_states), range(nb_symboles))
		# m= trainer.modeleDiscriminant(Xtaken,Ytaken,iteration,nb_states,nb_symboles,1)

		#évaluation de l'algo
		#Train set
		(goodWordTrain[j-1,i-1],badWordTrain[j-1,i-1],goodSentenceTrain[j-1,i-1],badSentenceTrain[j-1,i-1]) = m.accuracyEval(Xtaken,Ytaken)
		accuracyWordTrain[j-1,i-1]=goodWordTrain[j-1,i-1]/float(goodWordTrain[j-1,i-1]+badWordTrain[j-1,i-1])
		accuracySentenceTrain[j-1,i-1]=goodSentenceTrain[j-1,i-1]/float(goodSentenceTrain[j-1,i-1]+badSentenceTrain[j-1,i-1])
		print("TRAIN SET")
		print(	"words correctly estimated: ", goodWordTrain[j-1,i-1])
		print(	"words not correctly estimated: ", badWordTrain[j-1,i-1])
		print(	"accuracy on words: ", accuracyWordTrain[j-1,i-1])
		print("accuracy on sentence:")
		print(	"sentences correctly estimated: ", goodSentenceTrain[j-1,i-1])
		print(	"sentences not correctly estimated: ", badSentenceTrain[j-1,i-1])
		print(	"accuracy on sentences: ", accuracySentenceTrain[j-1,i-1])
		numpy.savetxt("goodWordTrain.csv", goodWordTrain, delimiter=",")
		numpy.savetxt("goodSentenceTrain.csv", goodSentenceTrain, delimiter=",")
		numpy.savetxt("badSentenceTrain", badSentenceTrain, delimiter=",")
		numpy.savetxt("badWordTrain", badWordTrain, delimiter=",")
		numpy.savetxt("accuracyWordTrain",accuracyWordTrain,delimiter=",")
		numpy.savetxt("accuracySentenceTrain",accuracySentenceTrain,delimiter=",")

		#Dev set
		# (goodWordDev[j-1,i-1],badWordDev[j-1,i-1],goodSentenceDev[j-1,i-1], badSentenceDev[j-1,i-1]) = m.accuracyEval(Xdev,Ydev)
		# accuracyWordDev[j-1,i-1]=goodWordDev[j-1,i-1]/float(goodWordDev[j-1,i-1]+badWordDev[j-1,i-1])
		# accuracySentenceDev[j-1,i-1]=goodSentenceDev[j-1,i-1]/float(goodSentenceDev[j-1,i-1]+badSentenceDev[j-1,i-1])
		# print("DEV SET")
		# print(	"words correctly estimated: ", goodWordDev[j-1,i-1])
		# print(	"words not correctly estimated: ", badWordDev[j-1,i-1])
		# print(	"accuracy on words: ", accuracyWordDev[j-1,i-1])
		# print("accuracy on sentence:")
		# print(	"sentences correctly estimated: " ,goodSentenceDev[j-1,i-1])
		# print(	"sentences not correctly estimated: ", badSentenceDev[j-1,i-1])
		# print(	"accuracy on sentences: ", accuracySentenceDev[j-1,i-1])
		# numpy.savetxt("goodWordDev.csv", goodWordDev, delimiter=",")
		# numpy.savetxt("goodSentenceDev.csv", goodSentenceDev, delimiter=",")
		# numpy.savetxt("badSentenceDev", badSentenceDev, delimiter=",")
		# numpy.savetxt("badWordDev", badWordDev, delimiter=",")
		# numpy.savetxt("accuracyWordDev",accuracyWordDev,delimiter=",")
		# numpy.savetxt("accuracySentenceDev",accuracySentenceDev,delimiter=",")

print("done. saving.")