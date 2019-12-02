"""
 plot_roc: plots ROC curve (previously serialized using pkl)

 Author: Keith Kenemer
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc
import os,sys
import pickle

# process command line
if len(sys.argv) < 2:
   print("\nUsage: plot_roc <roc.pkl>"   )
   print("roc.pkl: pickled (tpr,fpr,thr) tuple output from sklearn roc_curve()" )
   print("\n")
   exit()
else:
  roc_path = sys.argv[1]

# load saved roc data
with open(roc_path,"rb") as f:
   roc = pickle.load(f, encoding = 'latin1')
fpr = roc[0]
tpr = roc[1]
thr = roc[2]
#print("fpr:",fpr)
#print("tpr:",tpr)

# AUC
roc_auc = auc(fpr, tpr)
idx1 = np.argmin(np.abs(fpr - 0.01))  # 1% fpr
idx2 = np.argmin(np.abs(fpr - 0.001)) # 0.1% fpr

# plot results
plt.plot(fpr,tpr)
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.xscale('log')
plt.title('ROC curve for: ' + roc_path)
plt.text(0.1, 0.8, 'AUC: '+  "%.4f" % roc_auc)
plt.text(0.1, 0.7, 'TPR @ 1% FPR: '+  "%.4f" % tpr[idx1])
plt.text(0.1, 0.6, 'TPR @ 0.1% FPR: '+  "%.4f" % tpr[idx2])
plt.grid()
plt.show()


