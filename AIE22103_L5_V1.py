from sklearn.metrics import confusion_matrix, precision_score,recall_score,f1_score,mean_squared_error, mean_absolute_error, r2_score
import random
from sklearn.model_selection import train_test_split 
import openpyxl
import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV


def q1(y_train,yp_train,y_test,yp_test):
    #confusion matrix for train data
    ctrain = confusion_matrix(y_train,yp_train)
    #confusion matrix for test data
    ctest = confusion_matrix(y_test,yp_test)

    #precision score, recall score and f1 score for train
    ptrain=precision_score(y_train,yp_train)
    rtrain=recall_score(y_train,yp_train)
    f1train=f1_score(y_train,yp_train)

    #precision score, recall score and f1 score for test
    ptest=precision_score(y_test,yp_test)
    rtest=recall_score(y_test,yp_test)
    f1test=f1_score(y_test,yp_test)
    p=ptrain-ptest
    r=rtrain-rtest
    f1=f1train-f1test
    if p>0.05 or r>0.05 or f1>0.05:
        fit="overfitted"
    elif (-1*p)>0.05 or (-1*r)>0.05 or (-1*f1)>0.05:
        fit="underfitted"
    else:
        fit="well-fitted"
    return ptrain,rtrain,f1train,ptest,rtest,f1test,fit
def q2(prices,predict):
    a_p=np.array(prices)
    p_p=np.array(predict)
    mse=mean_squared_error(a_p,p_p)
    rmse=np.sqrt(mse)
    mape=np.mean(np.abs((a_p-p_p)/a_p))*100
    r2=r2_score(a_p,p_p)
    return mse,rmse,mape,r2
def q3(x0,y0,x1,y1):
    X0=np.array(x0)
    Y0=np.array(y0)
    X1=np.array(x1)
    Y1=np.array(y1)
    plt.scatter(X0,Y0,color='blue',label='class 0')
    plt.scatter(X1,Y1,color='red',label='class 1')
    plt.show()
def q4(k):
    x=np.arange(0,10.1,0.1)
    y=np.arange(0,10.1,0.1)
    X,Y=np.meshgrid(x,y)
    test=np.c_[X.ravel(),Y.ravel()]
    Xtrain=np.random.rand(100,2)*10
    ytrain=np.random.randint(2,size=100)
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(Xtrain,ytrain)
    p=knn.predict(test)
    plt.scatter(test[:,0],test[:,1],c=p,cmap='coolwarm',s=5)
    t="KNN = "+str(k)
    plt.title(t)
    plt.show()
def q7(x,y):
    X=np.array(x)
    Y=np.array(y)
    knn=KNeighborsClassifier()
    param_grid={'n_neighbors':range(1,51)}
    grid_search = GridSearchCV(knn, param_grid, cv=5)
    grid_search.fit(X,Y)
    bestk = grid_search.best_params_['n_neighbors']
    return bestk
wb=openpyxl.load_workbook("lab5.xlsx")#lab 5 contains cleaned data taken from styles.csv
data=wb["Sheet1"]
g=[]#l will contain the gender
for i in data['A']:
    g+=[i.value]
fid=[]#fashion id
for i in data['B']:
    fid+=[i.value]
yr=[]#year
for i in data['C']:
    yr+=[i.value]
label=[]
X=[]
X0=[]
Y0=[]
X1=[]
Y1=[]

for i in range(0,len(g)):
    if (fid[i] is not None) and (yr[i] is not None):
        if g[i]=="Men":
            label+=[0]
            X0+=[fid[i]]
            Y0+=[yr[i]]
            X+=[[fid[i],yr[i]]]
        if g[i]=="Women":
            label+=[1]
            X1+=[fid[i]]
            Y1+=[yr[i]]
            X+=[[fid[i],yr[i]]]
        
X_train, X_test, y_train,y_test=train_test_split(X,label,test_size=0.3)#splitting data into train n test
clf=LogisticRegression()
clf.fit(X_train,y_train)#training the classifier
yp_train=clf.predict(X_train)
yp_test=clf.predict(X_test)
q1_ans=q1(y_train,yp_train,y_test,yp_test)
print("Q1~")
print(" Training Data metrics~")
print(" Precision =",q1_ans[0],"\n Recall =",q1_ans[1],"\n F1 score =",q1_ans[2])
print(" Testing Data metrics~")
print(" Precision =",q1_ans[3],"\n Recall =",q1_ans[4],"\n F1 score =",q1_ans[5])
print(" Model is",q1_ans[6])

print("Q2~")
w=openpyxl.load_workbook("lab3.xlsx")
d=w["Sheet1"]
prices=[]
l=[]
for i in d['F']:
    l+=[i.value]
for i in d['E']:
    prices+=[i.value]
prices.pop(0)
#will be using random for this question since i will be using only 1 feature
p=[]
for i in range(0,len(prices)):
    if l[i]==0:
        p+=[random.randint(200,400)]
    else:
        p+=[random.randint(100,200)]
q2ans=q2(prices,p)
print(" Mean Squared Error=",q2ans[0])
print(" Root Mean Squared Error=",q2ans[1])
print(" Mean Absolute Percentage Error=",q2ans[2])
print(" R2=",q2ans[3])


print("Q3~")
x=[random.randint(1,11) for i in range(0,20)]
y=[random.randint(1,11) for i in range(0,20)]
x0=[]
y0=[]
x1=[]
y1=[]

for i in range(0,20):
    if (x[i]+y[i])%2==0:
        x0+=[x[i]]
        y0+=[y[i]]
    else:
        x1+=[x[i]]
        y1+=[y[i]]
q3(x0,y0,x1,y1)
print("Q4~")
q4(3)
print("Q5~")
for k in [1]:#,3,5,7,9]:
    q4(k)
print("Q6~")
q3(X0,Y0,X1,Y1)
print("Q7~")
bestk=q7(X,label)
print("Best value of k is ",bestk)
