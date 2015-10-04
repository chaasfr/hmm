
#**********************
# Lecture des donnees
#**********************
source = open("Data/ftb.dev.encode", "r")

toutesleslignes = source.readlines()
l=len(toutesleslignes)

X=[]
Y=[]

sequenceX = []
sequenceY = []
for i in range(l):
    r = toutesleslignes[i].rstrip('\n\r')
    print (r)
    if r=='':       
        X.append(sequenceX)
        sequenceX =[]
        Y.append(sequenceY)
        sequenceY =[]
    else:
        r = r.split(' ')
        sequenceX.append(int(r[0]))
        sequenceY.append(int(r[1]))

print len(X)
print type(X[0])
print (sequenceX)

#print X
#************************************
# Sauvegarde et Chargement d'un modele 
# ou de tout autre objet
#***************************************

model = 5 #ou n importe quel objet, un hmm par exemple

from sklearn.externals import joblib

joblib.dump(model, 'hmm.pkl')

hmm = joblib.load('hmm.pkl')


import numpy as np
from nltk.probability import *


#********************************
# Exemple de code python pour 
# utiliser un HMM
# ********************************

from hmm_mit_simple import HiddenMarkovModel
from hmm_mit_simple import HiddenMarkovModelTrainer


# Cas 1 : donnees simulees

SeqX=[[1, 2, 1, 1, 3, 5, 6, 7, 1, 1, 2], [2, 1, 0, 3, 1, 4, 7, 6, 2, 0, 1] , [0, 1, 1, 2, 5, 7, 7, 7, 1, 1, 1]]
SeqY=[[0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0], [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0],  [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0]]

nb_states = 2
nb_symboles = 8

trainer = HiddenMarkovModelTrainer(range(nb_states), range(nb_symboles))
m= trainer.train_supervised(SeqX, SeqY)

l = m.best_path_simple(SeqX[1])

print('l=',l)
# Cas 2 : avec les vraies donnees

Nb_seq_train = len(X)

SeqX = X[0:Nb_seq_train]
SeqY= Y[0:Nb_seq_train]
nb_symboles= np.max([np.max(U) for U in SeqX])
nb_states = np.max([np.max(U) for U in SeqY])

trainer = HiddenMarkovModelTrainer(range(nb_states), range(nb_symboles))
m= trainer.train_supervised(SeqX, SeqY)
print(X[600])
l = m.best_path_simple(X[600])

p = m.log_probability(X[600])

print("La sequence d etats la plus probabe est :", l, " avec la probablite :",p)
