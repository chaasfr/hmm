# coding=utf-8
import numpy
from hmm import HMMTrainer
from hmm import HMM
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

# Xtrain=[[1, 2, 1, 1, 3, 5, 6, 7, 1, 1, 2], [2, 1, 0, 3, 1, 4, 7, 6, 2, 0, 1] , [0, 1, 1, 2, 5, 7, 7, 7, 1, 1, 1]]
# Ytrain=[[0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0], [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0],  [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0]]
# nb_states = 2
# nb_symboles = 8

trainer = HMMTrainer(range(nb_states), range(nb_symboles))
m= trainer.modeleGeneratif(Xtrain, Ytrain)


# l = m.viterbi(Xtrain[601])
# print('Xtrain[600]=',Xtrain[601])
# print('l=',l)
# print('Ytrain[600]=',Ytrain[601])

# l = m.viterbi(Xtrain[1])
# print('Xtrain[600]=',Xtrain[1])
# print('l=',l)
# print('Ytrain[600]=',Ytrain[1])

#évaluation de l'algo
good=0
bad=0
for i in range(len(Xtrain)):
	l=m.viterbi(Xtrain[i])
	for j in range(len(Ytrain[i])):
		if Ytrain[i][j] == l[j]:
			good +=1
		else:
			bad +=1
print('TRAINING SET')
print ('good=',good)
print('bad=',bad)
print('accuracy=',good/float(good+bad))

good=0
bad=0
for i in range(len(Xdev)):
	l=m.viterbi(Xdev[i])
	for j in range(len(Ydev[i])):
		if Ydev[i][j] == l[j]:
			good +=1
		else:
			bad +=1
print('TEST SET')
print ('good=',good)
print('bad=',bad)
print('accuracy=',good/float(good+bad))