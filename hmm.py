# coding=utf-8
from nltk.probability import *
from numpy import *
from random import shuffle
import linecache
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

        self._states = states
        self._T = T
        self._symbols = symbols
        self._E = E
        self._pi = pi

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

        nbStates=int(lines[1])
        self._states=range(nbStates)

        nbSymbols=int(lines[3])
        self._symbols=range(nbSymbols)

        self._pi=zeros(nbStates,float)
        currentLine=5
        for i in range(len(self._states)):
            self._pi[i]=float(lines[i+currentLine][0:7])

        self._T=zeros((nbStates,nbStates),float)
        currentLine = currentLine + len(self._states) + 1
        for i in range(len(self._states)):
            for j in range(len(self._states)):
                self._T[i][j]=float(lines[i*len(self._states)+j+currentLine][0:7])

        self._E=zeros((nbStates,nbSymbols),float)
        currentLine = currentLine + len(self._states) * len(self._states) + 1
        for i in range(len(self._states)):
            for j in range(len(self._symbols)):
                self._E[i][j]=float(lines[i*len(self._symbols)+j+currentLine][0:7])
        fo.close

    def accuracyEval(self,sequenceX,sequenceY):
        goodWordNumber=0
        badWordNumber=0
        goodSentenceNumber=0
        badSentenceNumber=0
        wordWrong=zeros(len(self._symbols),float)
        wordCount=zeros(len(self._symbols),float)
        for i in range(len(sequenceX)):
            l=self.viterbi(sequenceX[i], True)
            foundBad=0
            for j in range(len(sequenceY[i])):
                wordCount[sequenceX[i][j]] += 1
                if sequenceY[i][j] == l[j]:
                    goodWordNumber +=1
                else:
                    badWordNumber +=1
                    foundBad +=1
                    wordWrong[sequenceX[i][j]]+=1
            if foundBad == 0:
                goodSentenceNumber +=1
            else:
                badSentenceNumber +=1
        return(goodWordNumber, badWordNumber, wordWrong,wordCount, goodSentenceNumber, badSentenceNumber)
        # on retourne toutes les stats et non seulement le pourcentage d'erreur au cas où
        # on souhaite analyser plus de données que juste la precision

    def isWordUnknown(self, wordTested):
        wordIsUnknown= True
        for i in self._states:
            if(self._E[i,wordTested] !=0):
                wordIsUnknown = False
        return(wordIsUnknown)

    def indexToWord(self, wordIndex):
        return(linecache.getline("Data/vocab",wordIndex))

    def isANoun(self,word):
        return(len(word)>1 and word[0].isupper() and word[1].islower())

    def probability(self, sequence):
        return exp(self.log_probability(sequence))

    def _output_logprob(self, state, symbol):
        return self._E[state].logprob(symbol)

    def log_probability(self, unlabeled_sequence):
        T = len(unlabeled_sequence)
        N = len(self._states)
        alpha = zeros((T, N), float64)

        symbol = unlabeled_sequence[0]
        for i, state in enumerate(self._states):
            alpha[0, i] = self._pi.logprob(state) + \
                          self._E[state].logprob(symbol)
        for t in range(1, T):
            symbol = unlabeled_sequence[t]
            for i, si in enumerate(self._states):
                alpha[t, i] = _NINF
                for j, sj in enumerate(self._states):
                    alpha[t, i] = _log_add(alpha[t, i], alpha[t-1, j] +
                                           self._T[sj].logprob(si))
                alpha[t, i] += self._E[si].logprob(symbol)

        p = _log_add(*alpha[T-1, :])
        return p

    def suffixAnalyse(self,word,symbol):
        fo=open("Data/vocab","r")
        lines=fo.readlines()
        nbWords = 0
        for j in range(len(lines)):
            if(not self.isWordUnknown(j)):
                i=0
                sizeLongestSameSuffixe=0
                sameSuffixe= True
                while(i<len(word) and i<len(lines[j]) and sameSuffixe):
                        if(word[-i] != lines[j][-i]):
                            sameSuffixe = False
                        else:
                            i +=1
                if(i==sizeLongestSameSuffixe):
                    self._E[:,symbol] += self._E[:,j]
                    nbWords += 1
                elif (i> sizeLongestSameSuffixe):
                    self._E[:,symbol] = zeros(len(self._states),float)
                    self._E[:,symbol] += self._E[:,j]
                    nbWords = 1
        self._E[:,symbol] = self._E[:,symbol]/ nbWords
        return()

    def viterbi(self,seq, isEvaling=False): #inspiré de hmm_mit_simple.best_path_simple
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
            # find the starting log probabilities for each state
            symbol = seq[0]
            for i, state in enumerate(self._states):
                V[0, i] = self._pi.logprob(state) + \
                          self._output_logprob(state, symbol)
                B[0, state] = None

            # find the maximum log probabilities for reaching each state at time t
            for t in range(1, T):
                symbol = seq[t]
                for j in range(N):
                    sj = self._states[j]
                    best = None
                    for i in range(N):
                        si = self._states[i]
                        va = V[t-1, i] + self._T[si].logprob(sj)
                        if not best or va > best[0]:
                            best = (va, si)
                    V[t, j] = best[0] + self._output_logprob(sj, symbol)
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
                #detect les mots inconnus
                if (isEvaling and self.isWordUnknown(symbol)):
                    unknownWord=self.indexToWord(symbol)
                    if(self.isANoun(unknownWord) and len(self.indexToWord(seq[t-1])) > 3):
                        self._E[7,symbol]=1
                    else:
                        self.suffixAnalyse(unknownWord,symbol)
                for j in range(N):
                    sj = self._states[j]
                    best = None
                    for i in range(N):
                        si = self._states[i]
                        va = V[t-1, i] + self._T[si][sj]
                        if not best or va > best[0]:
                            best = (va, si)
                    V[t, j] = best[0] + self._E[sj,symbol]
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

    def modeleGeneratif(self, labelled_sequences_X, labelled_sequences_Y, **kwargs):
        #**********************
        # Entraine un HMM suivant le modèle génératif
        # Pi la liste des scores initiaux (i.e. pi(Mi) est la probabilité que Mi débute la phrase)
        # T la liste des scores de transition
        # E la liste des scores d'émission
        # compte le nombre d'apparition en premier mot(pi), le nombre de bigramme(transition) et d'observation (emission)
        #**********************
        # default to the MLE estimate
        estimator = kwargs.get('estimator')
        if estimator == None:
            estimator = lambda fdist, bins: MLEProbDist(fdist)

        # count occurences of starting states, T out of each state
        # and output symbols observed in each state
        starting = FreqDist()
        T = ConditionalFreqDist()
        E = ConditionalFreqDist()
        for i in range(len(labelled_sequences_X)):
            lasts = -1
            xs = labelled_sequences_X[i]
            ys = labelled_sequences_Y[i]
            for j in range(len(xs)):
                state = ys[j]
                symbol = xs[j]
                if lasts == -1:
                    starting[state] +=1
                else:
                    T[lasts][state] +=1 
                E[state][symbol] +=1
                lasts = state

                # update the state and symbol lists
                if state not in self._states:
                    self._states.append(state)
                if symbol not in self._symbols:
                    self._symbols.append(symbol)

        # create probability distributions (with smoothing)
        N = len(self._states)
        pi = estimator(starting, N)
        A = ConditionalProbDist(T, ELEProbDist, N)
        B = ConditionalProbDist(E, ELEProbDist, len(self._symbols))
                               
        return HMM(self._symbols, self._states, A, B, pi,0)

    def modeleDiscriminant(self,sequenceX,sequenceY,iteration,nbStates,nbSymbols,epsilon, perceptronMoyenne=False):
        #**********************
        # Entraine un HMM suivant le modèle discriminant
        # Pi la liste des scores initiaux (i.e. pi(Mi) est la probabilité que Mi débute la phrase)
        # T la liste des scores de transition
        # E la liste des scores d'émission
        # compte le nombre d'apparition en premier mot(pi), le nombre de bigramme(transition) et d'observation (emission)
        #**********************
        pi = zeros(nbStates,float)
        T = zeros((nbStates,nbStates),float)
        E = zeros((nbStates,nbSymbols),float)

        symbols = range(nbSymbols)
        states = range(nbStates)
        modelHMM = HMM(symbols,states,T,E,pi,1)

        # Iteration
        for iterationNumber in range(iteration):
            print(iterationNumber)
            for sentenceNumber in range(len(sequenceY)):
                # Initialisation du gradient
                phiT = zeros((nbStates,nbStates))
                phiTViterbi = zeros((nbStates,nbStates))
                phiE = zeros((nbStates,nbSymbols))
                phiEViterbi = zeros((nbStates,nbSymbols))
                phiPi = zeros(nbStates)
                phiPiViterbi = zeros(nbStates)

                modelHMM = HMM(symbols,states,T,E,pi,1)
                # Initialisation
                m = sequenceX[sentenceNumber] # charge la phrase voulue

                # Calcul via Viterbi à l'aide du modèle discriminant ayant fait iterationNumber - 1 itérations
                cViterbi = modelHMM.viterbi(m) # Calcul la meilleure séquence de catégorie via l'algorithme de Viterbi
                c = sequenceY[sentenceNumber] # Charge les catégories voulues

                phiPi[c[0]] = epsilon
                phiPi[cViterbi[0]] = epsilon


                for k in range(len(m)-1): # Boucle sur la longueur de la phrase
                    phiT[c[k],c[k+1]] = phiT[c[k],c[k+1]] + epsilon
                    phiTViterbi[cViterbi[k],cViterbi[k+1]] = phiTViterbi[cViterbi[k],cViterbi[k+1]] + epsilon
                for k2 in range(len(m)):
                    for k in range(len(m)):
                        for categorie in range(nbStates):
                            if c[k] == categorie and m[k]==m[k2]:
                                phiE[categorie,m[k]] = phiE[categorie,m[k]] + epsilon
                            if cViterbi[k] == categorie and m[k]==m[k2]:
                                phiEViterbi[categorie,m[k]] = phiEViterbi[categorie,m[k]] + epsilon

                # Mise à jour des poids du modèle pour la prochaine itération
                if(perceptronMoyenne):
                    T = (T * (iterationNumber) + phiT - phiTViterbi) / (iterationNumber+1)
                    E = (E * (iterationNumber) + phiE - phiEViterbi) / (iterationNumber+1)
                    pi = (pi * (iterationNumber) + phiPi - phiPiViterbi) / (iterationNumber+1)
                else:
                    T = T + phiT - phiTViterbi
                    E = E + phiE - phiEViterbi
                    pi = pi + phiPi - phiPiViterbi

        return HMM(symbols,states,T,E,pi,1)