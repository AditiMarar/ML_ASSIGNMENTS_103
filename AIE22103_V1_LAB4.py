#in the excel sheet sheet 1 is the cleaned data and sheet 2 is the dataset downloaded
import openpyxl
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as KNC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
def get_list(data):
    sepal_l=[]#sepal length in cm
    sepal_w=[]#sepal width in cm
    petal_l=[]#petal length
    petal_w=[]#petal width
    label=[]#which species it belongs to
    species=["Iris-setosa","Iris-versicolor","Iris-virginica"]
    #putting all information of the dataset in the lists
    for i in data['A']:
        sepal_l+=[i.value]
    for i in data['B']:
        sepal_w+=[i.value]
    for i in data['C']:
        petal_l+=[i.value]
    for i in data['D']:
        petal_w+=[i.value]
    for i in data['E']:
        label+=[i.value]
    return sepal_l,sepal_w,petal_l,petal_w,label,species
def q1(a,b,c,d):
    l=[]
    l2=[]
    for i in range(0,len(a)):
        l+=[[a[i],b[i]]]
        l2+=[[c[i],d[i]]]
    c1_data=np.array(l)#class 1 data (sepal)
    c2_data=np.array(l2)# class 2 data (petal)
    centeroid1=np.mean(c1_data, axis=0)
    centeroid2=np.mean(c2_data, axis=0)
    spread1=np.std(c1_data)
    spread2=np.std(c2_data)
    interclass_dist=np.linalg.norm(centeroid1-centeroid2)
    return c1_data,c2_data,centeroid1,centeroid2,spread1,spread2,interclass_dist
def q2(d):# is the data
    data=np.array(d)
    n_bins=10
    h_range=(3,9)#histogram range
    h_value,bin_edges=np.histogram(data,bins=n_bins, range=h_range)
    mean=np.mean(data)
    var=np.var(data)
    plt.hist(data,bins=n_bins,range=h_range, edgecolor="white")#edgecolor to to separate the bins
    plt.title("Histogram for sepals")
    plt.grid(True)
    plt.show()
    return mean,var
def q3(a,b):
    l1=np.array(a)
    l2=np.array(b)
    r=len(a)+1# r is range
    m_d=[]#minkowski distance
    for i in range(1,r):
        var=np.linalg.norm(l1-l2,ord=i)
        m_d.append(var)
    plt.plot(range(1,r),m_d)
    plt.title("Minkowski distance vs r")
    plt.grid(True)
    plt.show()
def q4(a,b,c):
    x=[]
    for i in range(0,len(a)):
        x+=[[a[i],b[i]]]
    X=np.array(x)
    Y=np.array(c)
    return train_test_split(X,Y,test_size=0.9)
def q567(a,b,c):
    x=[]
    for i in range(0,len(a)):
        x+=[[a[i],b[i]]]
    X=np.array(x)
    Y=np.array(c)
    xtest=np.array([[5,3],[5,2],[6,3]])
    ytest=np.array([0,1,2])
    neigh=KNC(n_neighbors=3)
    neigh.fit(X,Y)
    data_pt=np.array([[5,3]])
    p_label=neigh.predict(data_pt)
    a=neigh.score(xtest,ytest)
    t_v=np.array([[5,2]])#test vector
    predicted_class=neigh.predict(t_v)
    return a,p_label,predicted_class
def q8(a,b,c):
    x=[]
    for i in range(0,len(a)):
        x+=[[a[i],b[i]]]
    X=np.array(x)
    Y=np.array(c)
    xtest=np.array([[5,3],[5,2],[6,3]])
    ytest=np.array([0,1,2])
    a_knn=[]#accuracy score for knn and nn
    a_nn=[]
    for k in range(1,151):#vary from k 1 to 11
        neigh=KNC(n_neighbors=k)
        neigh.fit(X,Y)
        y_pred_knn = neigh.predict(X)
        aknn = accuracy_score(ytest, y_pred_knn)
        a_knn.append(aknn)
        nnc=KNC(n_neighbors=1)
        nnc.fit(X,Y)
        y_pred_nn = nnc.predict(X)
        ann = accuracy_score(ytest, y_pred_nn)
        a_nn.append(ann)
    kval=range(1,11)
    plt.plot(kval,a_knn)
    plt.plot(kval,a_nn)
    plt.title("Accuracy of knn and nn")
    plt.xticks(kval)
    plt.legend()
    plt.show()
workbook=openpyxl.load_workbook("iris.xlsx")
data=workbook["Sheet1"]
info=get_list(data)#contains all the information about the dataset
sepal_l=info[0]
sepal_w=info[1]
petal_l=info[2]
petal_w=info[3]
label=info[4]
species=info[5]
print("\n\t\t\tML LAB-4 AIE22103 ADITI.A.M\n")
print("\nQ1~\n")
q1_ans=q1(sepal_l,sepal_w,petal_l,petal_w)
c1=q1_ans[0]
c2=q1_ans[1]
centeroid1=q1_ans[2]
centeroid2=q1_ans[3]
spread1=q1_ans[4]
spread2=q1_ans[5]
interclass_dist=q1_ans[6]

def q9()

    x=[]
    for i in range(0,len(a)):
        x+=[[a[i],b[i]]]
    X=np.array(x)
    Y=np.array(c)
    xtest=np.array([[5,3],[5,2],[6,3]])
    ytest=np.array([0,1,2])  
    neigh = KNC(n_neighbors=3)
    neigh.fit(X, Y)
    y_train_pred = neigh.predict(X)
    y_test_pred = neigh.predict(xtest)
    conf_matrix_train = confusion_matrix(Y, y_train_pred)
    conf_matrix_test = confusion_matrix(ytest, y_test_pred)
    precision_train = precision_score(Y, y_train_pred)
    precision_test = precision_score(ytest, y_test_pred)
    recall_train = recall_score(Y, y_train_pred)
    recall_test = recall_score(ytest, y_test_pred)
    f1_score_train = f1_score(Y, y_train_pred)
    f1_score_test = f1_score(ytest, y_test_pred)
    print("Confusion Matrix for Training Data:")
    print(conf_matrix_train)
    print("\nConfusion Matrix for Test Data:")
    print(conf_matrix_test)
    print("\nPerformance Metrics for Training Data:")
    print("Precision:", precision_train)
    print("Recall:", recall_train)
    print("F1-Score:", f1_score_train)
    print("\nPerformance Metrics for Test Data:")
    print("Precision:", precision_test)
    print("Recall:", recall_test)
    print("F1-Score:", f1_score_test)

print(" mean/centeroid of class 1 aka sepals~ ",centeroid1)
print(" mean/centeroid of class 2 aka petals~ ",centeroid2)
print(" Spread of class 1~ ",spread1)
print(" Spread of class 2~ ",spread2)
print(" Interclass distance~ ",interclass_dist)
print("\nQ2~\n")
q2_ans=q2(sepal_l)
print(" Mean of sepals length=",q2_ans[0])
print(" Variance of sepals length=",q2_ans[1])
print("\nQ3~\n")
q3(sepal_l,sepal_w)
print("\nQ4~\n")
x_train,x_test,y_train,y_test=q4(sepal_l,sepal_w,label)
print(" X_train = \n",x_train)
print(" X_test =\n",x_test)
print(" Y_train=\n",y_train)
print(" Y_test =\n",y_test)
print("\nQ5~\n")
a=q567(sepal_l,sepal_w,label)
print(" Predicted label of [5,3] is= ",a[1])

print("\nQ6~\n")

print(" Accuracy=",a[0])
print("\nQ7~\n")
print(" Predicted class=",a[2])
print("\nQ8~\n")
q8(sepal_l,sepal_w,label)
print("\nQ9~\n")
q9(sepal_,sepal_w,label)
