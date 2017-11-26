import numpy as np
from sklearn import svm
import pandas as pd


tmp = np.loadtxt("traindata.csv", dtype=np.str, delimiter=",")
data = tmp.astype(np.float)
tmp1 = np.loadtxt("trainlabel.csv", dtype=np.str, delimiter=",")
label = tmp1.astype(np.float)
tmp2 =  np.loadtxt("testdata.csv", dtype=np.str, delimiter=",")
tdata = tmp2.astype(np.float)

trainingdata = data[0:2415,:]  # To train the model
traininglabel = label[0:2415]
testdata = data[2415:3220, :] # To verify
testlabel = label[2415:3220]

clf = svm.SVC(kernel='linear', C=0.1).fit(trainingdata, traininglabel)


#scoresvm = clf.score(testdata, testlabel)
#print(scoresvm)
result = clf.predict(tdata)
np.reshape(result,(1380,1))
print(result)

np.savetxt('project1.csv',result, delimiter = ',')





