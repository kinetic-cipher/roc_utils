"""
 plot_roc_multi:
   Plots multiple ROC curves on the same graph (useful for comparison).
   Note: for more detailed information on a single ROC see 'plot_roc'.

 Author:
   Keith Kenemer   
"""

import os,sys
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import auc

# process command line
if len(sys.argv) < 2:
   print("\nUsage: plot_roc_multi <roc1.pkl>  <roc2.pkl> ... "   )
   print("roc<x>.pkl: pickled (tpr,fpr,thr) tuple output from sklearn roc_curve()"   )
   print("\n")
   exit()

# setup plot
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.xscale('log')
plt.title('ROC curve comparison')

# load & plot saved roc data
colors = ['b', 'g', 'r','m','c']
for k in range(1,len(sys.argv)  ):
   with open(sys.argv[k],"rb") as f:
      roc = pickle.load(f, encoding = 'latin1')
      fpr = roc[0]
      tpr = roc[1]
      plt.plot(fpr,tpr, color = colors[k%len(colors)], linewidth = 1, label = sys.argv[k] )

# show completed plot
plt.grid()
plt.legend(loc='lower right')
plt.show()


