# coding=utf-8
import nltk.probability
import numpy

class HMM(object):

    def __init__(self, symbols, states, T, E, pi):
        self._states = states #liste d'états possibles
        self._symbols = symbols #liste des symboles d'observations possibles
        self._T = T #matrice des scores de transition (i.e. t(i,j) la proba d'aller en j en étant en i)
        self._E = E #matrice des scores d'émission (i.e. E(i,j) la proba d'observer j en étant en i)
        self._pi = pi #liste des scores initiaux (i.e. pi(Mi) est la probabilité que Mi débute la phrase)

    #**********************
    # Forme générale du de l'algorithme de Viterbi
    # Retourne la séq d'obs maximisant le score associé à la séquence d'entrée
    # Entrées:
    # Seq: la sequence de mots étudiée
    #**********************
    def Viterbi(self,seq):
        N=len(seq) #longueur de la séquence étudiée
        S=len(T[1,:]) #nombre d'états
        V = zeros((T, N), float64) #TODO
        B = {} #résultat temporaire


class HMMTrainer(object):

    def __init__(self, states=None, symbols=None):
        #peuvent être nuls car ils sont updated dans le training
        self._states = states
        self._symbols = symbols

#**********************
# Entraine un HMM suivant le modèle génératif
# Pi la liste des scores initiaux (i.e. pi(Mi) est la probabilité que Mi débute la phrase)
# T la liste des scores de transition
# E la liste des scores d'émission
# compte le nombre d'apparition en premier mot(pi), le nombre de bigramme(transition) et d'observation (emission)
#**********************
    def ModeleGeneratif(self,sequenceX,sequenceY):
        # update de la liste des états et des symboles du HMM
        for i in range(len(sequenceX)):
            xs=sequenceX[i]
            ys=sequenceY[i]
            for j in range(xs):
                state= ys[j]
                symbol= xs[j]
                if state not in self._states:
                    self._states.append(state)
                if symbol not in self._symbols:
                    self._symbols.append(symbol)
        #Compte les occurences, les transitions, les emissions et les starters de chaque phrase
        first = [0]*len(self._states)
        T = zeros(len(self._states),len(self._states))
        E = zeros(len(self._states),len(self._symbols))
        occurence=[0]*len(self._states)
        for i in range(len(sequenceX)):
            lasts = -1
            xs = sequenceX[i]
            ys = sequenceY[i]
            for j in range(len(xs)):
                state = ys[j]
                symbol = xs[j]
                occurence[state] +=1; #compte le nombre d'occurence de state
                if lasts == -1: #si le mot débute la phrase
                    starting[state] +=1 #compte le nombre de fois que state commence une phrase
                else:
                    T[lasts][state] +=1 #compte le nombre de fois qu'on passe de lasts à state
                E[state][symbol] +=1 # compte le nombre de fois qu'on étiquette symbol par state
                lasts = state
        #calcul les probabilités
        pi = first/len(sequenceX) #retourne pi cf énoncé
        for i in range(len(self._states)):
            for j in range(len(self._states)):
                T[i,j] = T[i,j]/occurence[i]
            for j in range(len(self._symbols)):
                E[i,j]=E[i,j]/occurence[i]
        #Crée le HMM avec les paramêtres appris
        return HMM(self._symbols,self._states,T,E,pi)

    #def ModeleDiscriminant(self,sequenceX,sequenceY):
        #TODO