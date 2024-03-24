import numpy as np
from tabulate import tabulate
import openpyxl
from sklearn.model_selection import train_test_split,RandomizedSearchCV
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from scipy.stats import uniform, randint

#for sigmoid activation function
def sigmoid(x):
    return 1/(1+np.exp(-x))
def sigmoid_derivative(x):
    return x*(1-x)
#initializing weights
def weights(inp_size,op_size,hid_size):#sizez of input, hidden and output
    np.random.seed(1)
    weights_inp_hid=np.random.rand(inp_size,hid_size)
    bias_hid=np.random.rand(hid_size)
    weights_hid_op=np.random.rand(hid_size,op_size)
    bias_op=np.random.rand(op_size)
    return weights_inp_hid,bias_hid,weights_hid_op,bias_op
def train_nn(inp,targets,iterations,lr):
    weights_inp_hid,bias_hid,weights_hid_op,bias_op=weights(inp.shape[1],targets.shape[1],2)
    for i in range(iterations):
        h_inp=np.dot(inp,weights_inp_hid)+bias_hid
        h_op=sigmoid(h_inp)
        op_inp=np.dot(h_op,weights_hid_op)+bias_op
        op=sigmoid(op_inp)
        err=targets-op
        if np.mean(np.abs(err))<=0.002:
            print("Converged at iteration:",i)
            break
        d_op=err*sigmoid_derivative(op)
        err_hid=np.dot(d_op,weights_hid_op.T)
        d_hid=err_hid*sigmoid_derivative(h_op)
        weights_hid_op+=np.dot(h_op.T,d_op)*lr
        bias_op+=np.sum(d_op)*lr
        weights_inp_hid+=np.dot(inp.T,d_hid)*lr
        bias_hid+=np.sum(d_hid,axis=0)*lr
    return weights_inp_hid,bias_hid,weights_hid_op,bias_op
def predict(inp,w_inp_h,bias_h,w_h_op,bias_op):
    h_inp=np.dot(inp,w_inp_h)+bias_h
    h_op=sigmoid(h_inp)
    op_inp=np.dot(h_op,w_h_op)+bias_op
    op=sigmoid(op_inp)
    return op
def q2(X,y):
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
    perceptron_param_dist={'alpha':uniform(0.0001,0.1),'penalty':['12','11','elasticnet'],'max_iter':randint(500,2000),'tol':uniform(0.0001,0.01)}
    mlp_param_dist = {
        'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50, 25)],
        'activation': ['relu', 'tanh', 'logistic'],
        'solver': ['sgd', 'adam'],
        'alpha': uniform(0.0001, 0.1),
        'learning_rate': ['constant', 'adaptive'],
        'max_iter': randint(500, 2000),
        'tol': uniform(0.0001, 0.01)}
    percep=Perceptron()
    mlp=MLPClassifier()
    p_r_search=RandomizedSearchCV(percep,perceptron_param_dist,n_iter=1000,cv=5,verbose=2,random_state=42,n_jobs=-1)
    p_r_search.fit(X_train,y_train)
    mlp_r_search=RandomizedSearchCV(mlp,mlp_param_dist,n_iter=1000,cv=5,verbose=2,random_state=42,n_jobs=-1)
    mlp_r_search.fit(X_train,y_train)
    return perceptron_random_search.best_params_,perceptron_random_search.best_score_,mlp_random_search.best_params_,mlp_random_search.best_score_
inp_and=np.array([[0,0],[0,1],[1,0],[1,1]])
op_and=np.array([[0],[0],[0],[1]])
w_i_h,b_h,w_h_o,b_o=train_nn(inp_and,op_and,10000,0.02)
print("Prediction for AND gate:")
final=[]
for i in range(0,2):
    for j in range(0,2):
        output=predict(np.array([[i,j]]),w_i_h,b_h,w_h_o,b_o)
        final+=[[i,j,output[0][0],round(output[0][0])]]
head=["A","B","OUTPUT","ROUNDED OUTPUT"]
print(tabulate(final,headers=head,tablefmt="grid"))


wb=openpyxl.load_workbook('d1.xlsx')
sheet=wb.active
X=[]
y=[]
for row in sheet.iter_rows(values_only=True):
    X.append(row[:-1])
    y.append(row[-1])
a,b,c,d=q2(X,y)
print("Best parameters for perceptron")
print(a)
print("Best score",b)
print("Best parameters for MLP")
print(c)
print("Best score",d)
