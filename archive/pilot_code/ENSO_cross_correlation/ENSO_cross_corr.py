# correlate el nino and my metrics

import pandas as pd
import numpy as np

nino = pd.read_csv("el_nino_3p4.csv", header=None, delim_whitespace=True)
nino = np.asarray(nino)
nino = nino[:,1:-1]

