# coding=utf-8
from nltk.probability import *
from numpy import *
from random import shuffle
_NINF = float('-1e300')

class HMM(object):

    def __init__(self, symbols, states, T, E, pi,modele):
        if states != None:
            self._states = states #liste d'états possibles
        else:
            self.states = []

        if symbols != None:
            self._symbols = symbols #liste des symboles d'observations possibles
        else:
            self._symbols = []

        if T != None:
            self._T = T #matrice des scores de transition (i.e. t(i,j) la proba d'aller en j en étant en i)
        else:
            self._T=ConditionalFreqDist()

        if E != None:
            self._E = E #matrice des scores d'émission (i.e. E(i,j) la proba d'observer j en étant en i)
        else:
            self._E= ConditionalFreqDist()

        if pi != None:
            self._pi = pi #liste des scores initiaux (i.e. pi(Mi) est la probabilité que Mi débute la phrase)
        else:
            pi = FreqDist()

        if modele != None:
            self._modele=modele #savoir si on a train suivant un modèle gen(0) ou un modèle discriminant(1). 
        else:
            print("no training selected ! Select 0 for gen or 1 for dis")
            self._modele =0

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

    def accuracyEval(self,sequenceX,sequenceY):
        goodW=0
        badW=0
        goodS=0
        badS=0
        for i in range(len(sequenceX)):
            l=self.viterbi(sequenceX[i])
            foundBad=0
            for j in range(len(sequenceY[i])):
                if sequenceY[i][j] == l[j]:
                    goodW +=1
                else:
                    badW +=1
                    foundBad +=1
            if foundBad == 0:
                goodS +=1
            else:
                badS +=1
        return(goodW,badW,goodS,badS)

    def _problog(self,p):
        #sert à calculer le log des probabilités dans Viterbi
        return (math.log(p, 2) if p != 0 else _NINF)

    def viterbi(self,seq): #inspiré de hmm_mit_simple.best_path_simple
        #**********************
        # Forme générale de l'algorithme de Viterbi
        # Retourne la séq d'obs maximisant le score associé à la séquence d'entrée
        # Entrées:
        # Seq: la sequence de mots étudiée
        #**********************

        T=len(seq)
        N=len(self._states)
        V = zeros((T, N), float64) #Matrice de calcul des scores temporaires. V[t,Sj]= score d'arriver en Sj au temps t
        B = {} #Matrice des paths temp. B[t,Sj]=Si <=> si on est à l'état Sj au temps t, l'état le plus probable suivant est Si

        #On discerne deux cas suivant si l'on a train en modele_gen ou en modele discr
        #Si l'on a train en modele_gen, il faut prendre le log des probs pour éviter des pbs
        #de calcul. En revanche, en modele disc, on n'en a pas besoin.
        if self._modele==0:
            # Les probabilités de débuter pour chaque état
            symbol = seq[0]
            for i, state in enumerate(self._states):
                V[0, i] = self._problog(self._pi[state])+ self._problog(self._E[state][symbol])
                B[0, state] = None
            # Cherche l'état avec la plus grande proba au temps t à partir de t-1
            for t in range(1, T):
                symbol = seq[t]
                for j in range(N):
                    sj = self._states[j]
                    best = None
                    for i in range(N):
                        si = self._states[i]
                        va = V[t-1, i] + self._problog(self._T[j][i])
                        if not best or va > best[0]:
                            best = (va, si)
                    V[t, j] = best[0] + self._problog(self._E[sj][symbol])
                    B[t, sj] = best[1]

        elif self._modele==1:
            # Les probabilités de débuter pour chaque état
            symbol = seq[0]
            for i, state in enumerate(self._states):
                V[0, i] = self._pi[state]+ self._E[state][symbol]
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

        # Reconstruit le chemin final:
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
    def modeleGeneratif(self,sequenceX,sequenceY, taille):
        #Compte les occurences, les transitions, les emissions et les starters de chaque phrase
        pi = FreqDist()
        T = ConditionalFreqDist()
        E = ConditionalFreqDist()
        occurence = FreqDist()
        indice= list(range(0, len(sequenceX))) # listes des indices de X et Y.
        random.shuffle(indice) # shuffle pour prendre des élements aléatoires. On ne shuffle pas X et Y pour garder la correspondace X(i) correspond à Y(i)

        for i in range(taille):
            lasts = -1
            xs = sequenceX[indice[i]]
            ys = sequenceY[indice[i]]
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

                # update de la liste des états et des symboles du HMM.
                if state not in self._states:
                    self._states.append(state)
                if symbol not in self._symbols:
                    self._symbols.append(symbol)

        #calcul les probabilités~fréquences relatives
        for state in pi:
            pi[state]=pi[state]/float(taille) #float car on est sur 0<x<1
        for i in self._states:
            for j in self._states:
                if (occurence[i]!=0): T[i][j]=T[i][j] / float(occurence[i])
            for j in self._symbols:
                if (occurence[i]!=0): E[i][j]=E[i][j] / float(occurence[i])
        #Crée le HMM avec les paramêtres appris
        return HMM(self._symbols,self._states,T,E,pi,0)

    #def ModeleDiscriminant(self,sequenceX,sequenceY):
        #TODO