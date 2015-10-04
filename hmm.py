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
    def Viterbi(self,seq): #inspiré de hmm_mit_simple
        T=len(seq)
        N=len(self._states)
        Score = zeros((T, N), float64) #Matrice de calcul des scores temporaires. V[t,Sj]= score d'arriver en Sj au temps t
        B = zeros(T,N) #Matrice des paths temp. B[t,Sj]=Si <=> si on est à l'état Sj au temps t, l'état le plus probable suivant est Si

        # Les probabilités de débuter de chaque état
        symbol = seq[0]
        for i, state in enumerate(self._states):
            V[0, i] = self._pi(state) + self._E(state, symbol)
            B[0, state] = None

        # Cherche l'état avec la plus grande proba au temps t
        for t in range(1, T):
            symbol = seq[t]
            for j in range(N):
                sj = self._states[j]
                best = None
                for i in range(N):
                    si = self._states[i]
                    va = V[t-1, i] + self._T[j,i]
                    if not best or va > best[0]:
                        best = (va, si)
                V[t, j] = best[0] + self._E(sj, symbol)
                B[t, sj] = best[1]

        # Cherche l'état final
        best = None
        for i in range(N):
            val = V[T-1, i]
            if not best or val > best[0]:
                best = (val, self._states[i])

        # Parcourt B (dans le sens inverse) pour déterminer le chemin le plus probable
        current = best[1]
        sequence = [current]
        for t in range(T-1, 0, -1):
            last = B[t, current]
            sequence.append(last)
            current = last

        # retourne la chemin obtenu pour le remettre dans le bon sens
        sequence.reverse()
        return sequence

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