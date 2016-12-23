# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 01:59:06 2016

@author: Andy
"""

from sklearn import datasets
from sklearn import svm
import matplotlib.pyplot as plt

iris = datasets.load_iris()
digits = datasets.load_digits()

#print(digits.data)

clf = svm.SVC(gamma = 0.001, C = 100.)

clf.fit(digits.data[:-1], digits.target[:-1])
  
im = clf.predict(digits.data[-1:])

plt.plot(im)
plt.show()

print(im)