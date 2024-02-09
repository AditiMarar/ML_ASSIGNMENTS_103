import math
from tabulate import tabulate
def manhattan(x,y):
    m_l=[]# the manhattan list with all the variables i.e. the difference
    for i in range(len(x)):
        var=x[i]-y[i]#variable that will be added to the list
        if var<0:
            var*=-1
        m_l.append(var)#absolute variable required
    m_val=0#the manhattan value
    for i in m_l:
        m_val+=i#adding all the values to get the manhattan value
    return(m_val,m_l)#returning the manhattan value and the list containing variables that will be used in euclidean
def euclidean(x,y):
    m=manhattan(x,y)#obtaining the differences from manhattan
    e_val=0
    for i in m[1]:
        e_val+=i**2#adding the squared values of the differences
    ans=math.sqrt(e_val)
    return ans

def sortknn(d,val,x,y):#this is for sortin so that the minimum values are takes first
    n=len(d)
    for i in range(n-1):
        for j in range(0,n-i-1):
            if d[j]>d[j+1]:
                d[j], d[j+1] = d[j+1], d[j]
                x[j], x[j+1] = x[j+1], x[j]
                y[j], y[j+1] = y[j+1], y[j]
                val[j], val[j+1] = val[j+1], val[j]
    ans=[[x[i], y[i], d[i], val[i]] for i in range(n)]
    return ans
def freq(l):#freq is to determine which type has the highest frequency
    f={}
    for i in l:
        if i not in f:
            f[i]=1
        else:
            f[i]+=1
    m=0
    v=''
    for i in f:
        if f[i]>m:
            m=f[i]
            v=i
    return (v,m)
def knn(X,Y,x,y,k,val):
    dist=[]#distace that is subtracted and eucliidean is performed
    for i in range(len(x)):
        dist+=[math.sqrt(((x[i]-X)**2)+((y[i]-Y)**2))]
    if k>len(x):
        return -1
    else:
        o=sortknn(dist,val,x,y)
        ans=[]
        for i in range(k):
            ans+=[o[i][3]]
        final=freq(ans)
        return [final[0],o]

def label_encoding(label):
    d=[]
    for i in label:
        if i not in d:
            d+=[i]#saving all the types of labels in d
    decoded=[d.index(i) for i in label]#getting the decoded data
    return (decoded,d)
def one_hot(label):
    l=label_encoding(label)#l will store the output recieved when doing label encoding
    inp=l[0]#decoded data from label encoding, inp if for input for the one hot decoded data
    d=l[1]#types of labels
    decoded=[]
    for i in range(len(inp)):
        o=[0 for i in range(len(d))]#o stands for output
        o[inp[i]]=1#1 is added to the index of the decoded data of label encoding
        #example there are 1 types of labels therefore in label encoding it will be 0,1
        # but for one hot it will be [1,0],[0,1], notice when at the index at label encoding is equl to 1
        
        decoded.append(o)
    return decoded
print("\t\tLAB 2\n\nQ1\nManhattan and Euclidean")
x1=[]
y1=[]
d=int(input("Enter the dimension of the vector: "))
for i in range(d):
    x1+=[int(input(f" Enter X{i} "))]
    y1+=[int(input(f" Enter Y{i} "))]
print("Manhattan value of the vectors is ",manhattan(x1,y1)[0])
print("Euclidean value of the vectors is ",euclidean(x1,y1))


print("\n\nQ2\nKNN CLASSIFIER")
x2=[]
y2=[]
val=[]
n_e=int(input("Enter total no. of entries: "))
for i in range(n_e):
    x2+=[int(input(f"Enter value of X{i}: "))]
    y2+=[int(input(f"Enter value of Y{i}: "))]
    val+=[int(input(f"Enter type class at {i}: "))]
print("To find the class at a particular point")
X=int(input("Enter X: "))
Y=int(input("Enter Y: "))
k=int(input("Enter the value of k for knn classifier: "))
o2=knn(X,Y,x2,y2,k,val)
if o2==-1:
    print("Invalid no. for k")
else:
    final_2=o2[1]
    head2=["X","Y","Distance","Type"]
    print(tabulate(final_2,headers=head2, tablefmt="grid"))
    print("Answer~")
    print(tabulate([[X,Y,0,o2[0]]],headers=head2, tablefmt="grid"))



print("\n\nQ3&4\nLabel encoding and One hot encoding")
head=["DECODED DATA","ENCODED DATA"]
label=input("Enter labels with a comma between each element:\n")
label=label.split(",")

print("\n\nLABEL ENCODING\n")
output3=label_encoding(label)
final_output_3=[[output3[0][i], label[i]] for i in range(len(label))]
print(tabulate(final_output_3,headers=head, tablefmt="grid"))



print("\n\nONE HOT ENCODING\n")
output4=one_hot(label)
final_output_4=[[output4[i],label[i]] for i in range(len(label))]
print(tabulate(final_output_4, headers=head, tablefmt="grid"))


