# coding=utf-8
from nltk.probability import *
from numpy import *

class HMM(object):

    def __init__(self, symbols, states, T, E, pi):
        self._states = states #liste d'états possibles
        self._symbols = symbols #liste des symboles d'observations possibles
        self._T = T #matrice des scores de transition (i.e. t(i,j) la proba d'aller en j en étant en i)
        self._E = E #matrice des scores d'émission (i.e. E(i,j) la proba d'observer j en étant en i)
        self._pi = pi #liste des scores initiaux (i.e. pi(Mi) est la probabilité que Mi débute la phrase)


    def save(self,fileHMM):
        fo= open(fileHMM, "wb")
        fo.write( \
            "#nb etats \n" + \
            str(len(self._states)) + "\n" +\
            "#nb observables \n" + \
            str(len(self._symbols)) + "\n" +\
            "#parametres initiaux \n")
        for i in range(len(self._states)):
            fo.write("%.6f" % self._pi[i] + " # I(" + str(i) + ")\n")
        fo.write("#parametres de transition \n")
        for i in range(len(self._states)):
            for j in range(len(self._states)):
                fo.write("%.6f" %self._T[i][j] + " # T(" + str(i) + "," + str(j) + ")\n")
        fo.write("#parametres d\'émission \n")
        for i in range(len(self._states)):
            for j in range(len(self._symbols)):
                fo.write("%.6f" %self._E[i][j] + " # E(" + str(i) + "," + str(j) + ")\n")
        fo.close

    def load(self,fileHMM):
        fo=open(fileHMM,"r")
        currentLine= 0
        lines=fo.readlines()

        nb_states=int(lines[1])
        self._states=[]
        for i in range(nb_states):
            self._states.append(i)

        nb_symbols=int(lines[3])
        self._symbols=[]
        for i in range(nb_symbols):
            self._symbols.append(i)

        self._pi=FreqDist()
        currentLine=5
        for i in range(len(self._states)):
            self._pi[i]=float(lines[i+currentLine][0:7])

        self._T=ConditionalFreqDist()
        currentLine = currentLine + len(self._states) + 1
        for i in range(len(self._states)):
            for j in range(len(self._states)):
                self._T[i][j]=float(lines[i*len(self._states)+j+currentLine][0:7])

        self._E=ConditionalFreqDist()
        currentLine = currentLine + len(self._states) * len(self._states) + 1
        for i in range(len(self._states)):
            for j in range(len(self._symbols)):
                self._E[i][j]=float(lines[i*len(self._symbols)+j+currentLine][0:7])


        fo.close

    #**********************
    # Forme générale de l'algorithme de Viterbi
    # Retourne la séq d'obs maximisant le score associé à la séquence d'entrée
    # Entrées:
    # Seq: la sequence de mots étudiée
    #**********************
    def viterbi(self,seq): #inspiré de hmm_mit_simple.best_path_simple
        T=len(seq)
        N=len(self._states)
        V = zeros((T, N), float64) #Matrice de calcul des scores temporaires. V[t,Sj]= score d'arriver en Sj au temps t
        B = {} #Matrice des paths temp. B[t,Sj]=Si <=> si on est à l'état Sj au temps t, l'état le plus probable suivant est Si

        # Les probabilités de débuter pour chaque état
        symbol = seq[0]
        for i, state in enumerate(self._states):
            V[0, i] = self._pi[state] + self._E[state][symbol]
            B[0, state] = None

        # Cherche l'état avec la plus grande proba au temps t à partir de t-1
        for t in range(1, T):
            symbol = seq[t]
            for j in range(N):
                sj = self._states[j]
                best = None
                for i in range(N):
                    si = self._states[i]
                    va = V[t-1, i] + self._T[j][i]
                    if not best or va > best[0]:
                        best = (va, si)
                V[t, j] = best[0] + self._E[sj][symbol]
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
        if states:
            self._states = states
        else:
            self._states = []
        if symbols:
            self._symbols = symbols
        else:
            self._symbols = []

#**********************
# Entraine un HMM suivant le modèle génératif
# Pi la liste des scores initiaux (i.e. pi(Mi) est la probabilité que Mi débute la phrase)
# T la liste des scores de transition
# E la liste des scores d'émission
# compte le nombre d'apparition en premier mot(pi), le nombre de bigramme(transition) et d'observation (emission)
#**********************
    def modeleGeneratif(self,sequenceX,sequenceY):
        # update de la liste des états et des symboles du HMM. A optimiser
        # for i in range(len(sequenceX)):
        #     xs=sequenceX[i]
        #     ys=sequenceY[i]
        #     for j in range(len(xs)):
        #         state= ys[j]
        #         symbol= xs[j]
        #         if state not in self._states:
        #             self._states.append(state)
        #         if symbol not in self._symbols:
        #             self._symbols.append(symbol)

        #Compte les occurences, les transitions, les emissions et les starters de chaque phrase
        pi = FreqDist()
        T = ConditionalFreqDist()
        E = ConditionalFreqDist()
        occurence = FreqDist()
        for i in range(len(sequenceX)):
            lasts = -1
            xs = sequenceX[i]
            ys = sequenceY[i]
            for j in range(len(xs)):
                state = ys[j]
                symbol = xs[j]
                occurence[state] +=1
                if lasts == -1: #si le mot débute la phrase
                    pi[state] +=1 #compte le nombre de fois que state commence une phrase
                else:
                    T[lasts][state] +=1 #compte le nombre de fois qu'on passe de lasts à state
                E[state][symbol] +=1 # compte le nombre de fois qu'on étiquette symbol par state
                lasts = state

                # update de la liste des états et des symboles du HMM. A optimiser
                if state not in self._states:
                    self._states.append(state)
                if symbol not in self._symbols:
                    self._symbols.append(symbol)

        #calcul les probabilités~fréquences relatives
        for state in pi:
            pi[state]=pi[state]/float(len(sequenceX)) #float car on est sur 0<x<1
        for i in range(len(self._states)):
            for j in range(len(self._states)):
                T[i][j]=T[i][j] / float(occurence[i])
            for j in range(len(self._symbols)):
                E[i][j]=E[i][j] / float(occurence[i])
        #Crée le HMM avec les paramêtres appris
        return HMM(self._symbols,self._states,T,E,pi)

    #def ModeleDiscriminant(self,sequenceX,sequenceY):
        #TODO