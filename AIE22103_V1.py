'''
MACHINE LEARNING ~ 22AIE213
LAB 1
SET A
ADITI AJAY MARAR
BL.EN.U4AIE22103
'''

# ALL NECESSARY FUNCTIONS ARE DEFINED BELOW

def pair_10(list_of_no):# the argument is an inputed list of integers from the user
    list_containing_pairs=[]#it will be a nested list that contains a pairs that sum up to 10
    for variable in list_of_no:
        pair_ten=10-variable#finding the complement of 10 wrt the variable
        if pair_ten in list_of_no:
            if([pair_ten,variable] not in list_containing_pairs):#this is to make sure there are no duplcates for example if for anf 6 are ther in the list then if the condition is not there the both [4,6] and [6,4] will be there in the resulted list

                list_containing_pairs.append([variable,pair_ten])
    if list_of_no.count(5)==1:
        list_containing_pairs.remove([5,5])#if 10-5 = 5 therefore 5 will also be there in the list eve though there is only 1 5
    result=[len(list_containing_pairs),list_containing_pairs]#1st index contains the count 
    return result
def max_min_range(list_of_no):#for q2 to find range of the inputed list
    if len(list_of_no)<3:#there is a condition in q2 that the range should be more that 1
        result = "ERROR!!\nMore than 2 elements neede for range determination"
    else:
        maximum=list_of_no[0]
        minimum=list_of_no[0]
        for index in range(1,len(list_of_no)):#ind is for index, since we have already started and initialized both the variables with the first no. in the list we will start with the second no.
            if list_of_no[index]<minimum:
                minimum=list_of_no[index]#if the value is less than the minimum then minimum=value
            if list_of_no[index]>maximum:
                maximum=list_of_no[index]#if the value is more than the maximum then maximum=value
        result="Range of the given list "+str(list_of_no)+" is ("+str(maximum)+","+str(minimum)+")"
    return result
def highest_char(string):
    if string.isalpha():# a condition for q4, the string should only contain alphabets
        dict_for_characters_n_count={}#an empty dictionary that will soon contain frequency of each character
        for character in string:#each value in the string
            if character not in dict_for_characters_n_count:
                dict_for_characters_n_count[character]=1#adding new fornd alphabets to the dictionary containing frequency
            else:
                dict_for_characters_n_count[character]+=1#udating the frequency of the character count in the string
        maximum=0
        max_char=''#maximum will contain count of the character with highest count,the character will be stored in max_char
        for temp in dict_for_characters_n_count:
            if dict_for_characters_n_count[temp]>maximum:
                maximum=dict_for_characters_n_count[temp]
                max_char=temp#if the count of the charaacter exceeds the maximum then maximum is equal to the count and the max_char=temp, temp is for temporary variable
        result=[maximum,max_char]
        return result
    else:
        return [-1,""]#this is so that the main function will know there is an error in the inputed string (since count can't be -1)
def matrix_multiply(a,b):#a and b are 2 matrices that are going to be multiplied
    result_matrix=[]
    length=len(a)
    #note we don't neen to keep changing the range in the below for loops as we are using square matrices, therefore no need to change it
    for i in range(length):
        row=[]
        for j in range(length):
            variable=0
            for k in range(length):
                variable+=a[i][k]*b[k][j]#adding the value to the result matrix, at the moment, adding an element to the row i of the matrix
            row.append(variable)#adding the element to the row of the result multiplied matrix
        result_matrix.append(row)#adding the row to the resulting matrix
    return result_matrix
def matrix_power(const,matrix):
    if const>0:#the power should the a +ve no., acondition in q3
        result_matrix=matrix#initializing it to the matrix incase the power is only 1
        for i in range(1, const):#since to the power 1 is already done
            result_matrix=matrix_multiply(result_matrix,matrix)#doind repeated multiplication to get the power of the matrix
        return result_matrix
    else:
        return -1#to indicate error message incase the power is a -ve no.
print("\n\t\t\tANSWERS TO SET A\n")

#QUESTION 1

print("QUESTION 1~\nCONSIDER THE GIVEN LIST [2,7,4,1,,3,6]\nWAP TO COUNT PAIRS OF ELEMENTS WITH SUM = 10\n")
input_str=input(" Enter a set of integers in this manner~ 1,2,3,4 : ")
input_list=input_str.split(",")#converting the string to a list
#the next loop is to convert the elements in the list that are a strin into integers
for index_q1 in range(len(input_list)):
    input_list[index_q1]=int(input_list[index_q1])
result_of_q1=pair_10(input_list)#calling the function that will return the no. of pairs that sum up to 10 and the pairs itself
print("ANSWER~")
if result_of_q1[0]==0:#incase there are no pairs found
    print(" No pairs found in the list that sum up to 10")
else:
    print(" Total count of pairs of elements that sum up to 10 is ",result_of_q1[0],"\n The pairs are~")
    for pairs in result_of_q1[1]:
        print(" ",pairs[0],",",pairs[1])

#QUESTION 2

print("\n\nQUESTION 2~\nWAP THAT TAKES A LIST OF REAL NO.S AS INPUT AND RETURNS RANGE\n")
input_str2=input(" Enter a set of integers in this manner~ 1,2,3,4 : ")
input_list2=input_str2.split(",")#convert str to int
for index_q2 in range(len(input_list2)):
    input_list2[index_q2]=int(input_list2[index_q2])#at the moment each element is a string, now converting to integer
result_of_q2="ANSWER~\n "+max_min_range(input_list2)#will return the final result of the answer to the q2
print(result_of_q2)

#QUESTION 3

print("\n\nQUESTION 3~\nWAP THAT ACCEPTS A SQUARE MATRIX A AND A +VE INTEGER NO. M AND RETURNS A TO THE POWER M\n")
r_c=int(input("Enter total no. of rows/columns "))#no. of rows and columns will be same for square matrix
k=int(input("Enter a +ve no. "))#the power
input_matrix_q3=[]
for i in range(0,r_c):
    row=[]
    for j in range(0,r_c):
        var=int(input(str("Enter variable at position "+str(i)+","+str(j)+": ")))
        row.append(var)#adding variables of thesquare matrix to the row i
    input_matrix_q3.append(row)#adding the rows to the matrix
print("ANSWER\n")
ans_3=matrix=matrix_power(k,input_matrix_q3)#calling function that returns power of the matrix
if -1==ans_3:
    print(" ERROR!! INPUTED CONSTANT SHOULD BE +VE")
else:#if the power is +ve printing the matrix in a format
    for i in ans_3:
        output_str_q3=""
        for  j in i:
            output_str_q3+=" "+str(j)
        print(output_str_q3)# the row i of the matrix is printed

#QUESTION 4
    
print("\n\nQUESTION 4~\nWAP TO COUNT THE HIGHEST OCCURING CHARACTERS IN A STRING\n")
inp_str=input(" Enter a word: ")
result_of_q4=highest_char(inp_str)
if result_of_q4[0]==-1:\
   print(" INVALID INPUT")
else:
    print("ANSWER\n The highest occuring character in '",inp_str,"' is ",result_of_q4[1]," with occurrence of ",str(result_of_q4[0]))

    
