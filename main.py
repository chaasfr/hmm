# coding=utf-8
import hmm
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