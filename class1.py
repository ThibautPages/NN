import numpy as np
import random
import time




## contient les informations sur le comment le réseau doit gérer le cout
## cout entropri croisée, réduit le problème de faible variation lorsque l'erreur est grande
class Cout_Entropie_Croisee():

    @staticmethod
    def fn(entree, sortieAttendue):
        return np.sum(np.nan_to_num(-y*np.log(entree)-(1-sortieAttendue)*np.log(1-entree)))

    @staticmethod
    def delta(z, entree, sortieAttendue):
        return (entree-sortieAttendue)


class Cout_Quadratique():

    @staticmethod
    def fn(entree, sortieAttendue):
        return 0.5*np.linalg.norm(entree-sortieAttendue)**2

    @staticmethod
    def delta(z, entree, sortieAttendue):
        return (entree-sortieAttendue) * sigmoide_prime(z)






class Reseau_Neurones:

    ## Initialisation du reseaux de neurones
    ## Entree : tableau contenant le nombre de neurones par couche
    def __init__(self,taille,cout=Cout_Entropie_Croisee):
        self.nbrCouche = len(taille)
        self.taille = taille

        self.init_poids_grand()

        self.cout = cout


    ## initialisation des poids du reseau
    ## initialisation avec une loi Normal/Gaussienne avec une espérance de 0 et u écart type de 1
    ## pour les poids ont divise par la racine du nbr de connection entrante
    def init_poids_defaut(self):
        ## crée un tableau contenant les biais de chaque couche sauf l'entree
        ## les biais sont un vecteur colonne de la taille de la couche
        self.biais = [np.random.randn(y, 1) for y in self.taille[1:]]

        ## crée un tableau contenant les poids entre chaque couche
        ## les poids sont des matrice de taille coucheSuivante x couchePrecedente
        self.poids = [np.random.randn(y, x)/np.sqrt(x) for x, y in zip(self.taille[:-1], self.taille[1:])]


    def init_poids_grand(self):
        self.biais = [np.random.randn(y, 1) for y in self.taille[1:]]
        self.poids = [np.random.randn(y, x) for x, y in zip(self.taille[:-1], self.taille[1:])]

    ## retourne la valeur de sortie en fonction d'une entrée a
    def propagation(self, a):
        for b, p in zip(self.biais, self.poids):
            a = sigmoide(np.dot(p, a)+b)
        return a
    

    ## entraine le reseau de neurone
    def descente_gradient(self, donneesEntrainement, nbrTours, tailleDecoupes, rythmeApprentissage, lmbda = 0, donneesTest = None):
        if donneesTest:
            nTest = len(donneesTest)
        n = len(donneesEntrainement)

        for j in range(nbrTours):
            ## mélange l'ordre des données pour que les découpages soit différents à chaque tour
            random.shuffle(donneesEntrainement)

            ## crée les découpes
            miniDecoupes = [donneesEntrainement[k:k+tailleDecoupes] for k in range(0,n,tailleDecoupes)]

            ## pour chaque découpe met à jour tout le réseau ( rétropropagation )
            for miniDecoupe in miniDecoupes:
                self.maj_decoupe(miniDecoupe, rythmeApprentissage, lmbda, n)

            ## permet de contrôler l'evolution du réseau
            if donneesTest:
                print(f"Tour {j}: {(self.evaluer(donneesTest)*100/nTest).__round__(3):<3}%")
            else:
                print(f"Tour {j} complet")


    ## évolution du réseau !
    def maj_decoupe(self, miniDecoupe, rythmeApprentissage, lmbda, n):
        nablaB = [np.zeros(b.shape) for b in self.biais]
        nablaP = [np.zeros(p.shape) for p in self.poids]

        for entree, sortieAttendue in miniDecoupe:
            ## calcul du gradient de descente pour 1 entrée
            deltaNablaB, deltaNablaP = self.retropropagation(entree, sortieAttendue)

            ## somme des gradients
            nablaB = [nb+dnb for nb, dnb in zip(nablaB, deltaNablaB)]
            naplaP = [np+dnp for np, dnp in zip(nablaP, deltaNablaP)]

        ## L2 regularisation (1-rythmeApprentissage*(lmbda/n))
        self.biais = [b - ((rythmeApprentissage/len(miniDecoupe)) * nb) for b, nb in zip(self.biais, nablaB)]
        self.poids = [(1-rythmeApprentissage*(lmbda/n))*p - ((rythmeApprentissage/len(miniDecoupe)) * np) for p, np in zip(self.poids, nablaP)]


    ## retropropagation du gradient
    def retropropagation(self, entree, sortieAttendue):
        nablaB = [np.zeros(b.shape) for b in self.biais]
        nablaP = [np.zeros(p.shape) for p in self.poids]

        ## descente du gradient
        ## la méthode déjà programé n'est as utilisé car nous enregistrons les différentes activations
        ## ainsi que les vecteurs z, valeur de l'activation avant l'application de la sigmoide (zs)
        ## qui sont utilisées dans le calcul des gradients retropropagés
        activation = entree
        activations = [entree]
        zs = []

        for b, p in zip(self.biais, self.poids):
            z = np.dot(p, activation)+b
            zs.append(z)
            
            activation = sigmoide(z)
            activations.append(activation)
        
        ## retropropagation
        ## calcul de la dernière couche
        delta = (self.cout).delta(zs[-1],activations[-1], sortieAttendue)## EQ 1
        nablaB[-1] = delta ## EQ 3
        nablaP[-1] = np.dot(delta, activations[-2].transpose()) ## EQ 4

        ## pour toute les autres couches
        for l in range(2, self.nbrCouche):
            z = zs[-l]
            sp = sigmoide_prime(z)

            delta = np.dot(self.poids[-l+1].transpose(), delta) * sp ## EQ 2
            nablaB[-l] = delta ## EQ 3
            nablaP[-l] = np.dot(delta, activations[-l-1].transpose()) ## EQ 4

        return (nablaB, nablaP)
            

    ## donne le nombre de données qui sont correctement évaluées
    def evaluer(self, donneesTest):
        resultatsTest = [(np.argmax(self.propagation(x)),np.argmax(y)) for (x, y) in donneesTest]

        return sum(1 for (x, y) in resultatsTest if int(x) == int(y))



def sigmoide(z):
    return 1.0/(1.0+np.exp(-z))


def sigmoide_prime(z):
    return sigmoide(z)*(1-sigmoide(z))
