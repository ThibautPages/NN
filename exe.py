import numpy as np
from class1 import *
import time

from numpy import array

import BDD_Entrainement
import BDD_Verification

r = Reseau_Neurones([9,1000,6])

r.descente_gradient(BDD_Entrainement.donneeEntrainement,10000,50,0.001, 0,
                    BDD_Verification.donneeEntrainement)

time.sleep(36000)
