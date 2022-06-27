# Python Program for data calibration based on Newton-Raphson method algorithm

# External libraries for coding calibration program
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from tabulate import tabulate

# Definition of basic parameters of mode function F=A(x^b)
# A : Scale factor
# x : Digital output (raw data)
# b : constant, determine slope of model function
# n : The number of sensels that have the same digital output
# j : number of equations
# i : Range of digital output value

# Ture force of given raw data(known)
F1=3610
F2=25270

# Read raw dataA,B(Filename extension : .xlsx, Matrix shape of raw data ; 48x48)
dir1 = 'C:\\Users\\A\\Desktop\\SFC1900CXR2-22.06.07-20220607T113023_220624(2번가압) (1)\\bias_test\\1bar\\22.02.28.39'
dir2 = 'C:\\Users\\A\\Desktop\\SFC1900CXR2-22.06.07-20220607T113023_220624(2번가압) (1)\\bias_test\\7bar\\22.02.28.39'
excelFile1 = os.path.join(dir1, '22.02.28.39-2d.xlsx')
data1 = pd.read_excel(excelFile1)
dataA = np.array(data1,dtype='int32')
excelFile2 = os.path.join(dir2, '22.02.28.39-2d.xlsx')
data2 = pd.read_excel(excelFile2)
dataB = np.array(data2,dtype='int32')

# Definition of using variable
N=256 # 8bit
data_mag = dataA.shape # Shape of matrix of raw data
data=np.zeros((2,data_mag[0],data_mag[1]),dtype='int32') # Define variables for storing two 48x48 matrices in a 3d array structure
ni=np.zeros((2,256),dtype='int32') # Define variables for counting the number of nodes that have same digital output value
Total_sum=np.zeros((2)) # Define variables for calculating total sum of digital output value
Total_sumb=np.zeros((2)) # Define variables for calculating total sum of digital output value multiplied by exponent b
Total_sumc=np.zeros((2)) # Define variables for calculating total force based on calculated model function

# Process for storing two 48x48 matrices in a 3d array structure and calculating total sum of digital output value
for n in range(0,2):
    if n==0:
        for j in range(0, data_mag[0]):
            for k in range(0, data_mag[1]):
                data[n][j][k] = dataA[j][k]
                Total_sum[0]=Total_sum[0]+dataA[j][k]
    elif n==1:
        for j in range(0, data_mag[0]):
            for k in range(0, data_mag[1]):
                data[n][j][k] = dataB[j][k]
                Total_sum[1] = Total_sum[1] + dataB[j][k]

# Process for counting the number of nodes that have same digital output value
for n in range(0,2):
    for i in range(0, 256):
        for j in range(0, data_mag[0]):
            for k in range(0, data_mag[1]):
                if data[n][j][k] == i:
                    ni[n][i] = ni[n][i] + 1

# Process to determine exponent b of model function based on First order Newton-Rapshon method
# Process to apply first order Newton-Rapshon method : definition of object function (function that have b as dependent variable)
def f(b):
    sum = np.zeros(2)
    term = np.zeros(2)
    for n in range(0,2):
        for i in range(1, N):
            term[n] = ni[n][i] * (i ** b) * 0.001
            sum[n] = sum[n] + term[n]
    return (sum[0]/sum[1])-(F1/F2)
# Process to apply first order Newton-Rapshon method : First derivative of object function
def df(b):
    sum=np.zeros(4)
    term=np.zeros(4)
    for n in range(0,2):
        for i in range(1, N):
            term[2*n] = ni[n][i] * (i ** b) * 0.001
            sum[2*n] = sum[2*n] + term[2*n]
            term[2*n+1] = ni[n][i] * np.log(i) * (i ** b) * 0.001
            sum[2*n+1] = sum[2*n+1] + term[2*n+1]
    return ((sum[1]*sum[2])-(sum[0]*sum[3]))/(sum[2]**2)

# Process to apply first order Newton-Rapshon method : Calculate exponent b and the number of iteration
def NR1(f, df):
    i=0
    x0=1
    while True:
        i=i+1
        x = x0-(f(x0)/df(x0))
        if f(x) > 10**(-10):
            x0 = x
        else:
            break
    return x,i
b1 = NR1(f, df)[0]

# Process to determine coefficient A of model function based on result of calculated b
# Process for calculating total sum of digital output value multiplied by exponent b
for n in range(0,2):
    if n==0:
        for j in range(0, data_mag[0]):
            for k in range(0, data_mag[1]):
                Total_sumb[0] = Total_sumb[0] + (data[n][j][k]) ** b1
    elif n==1:
        for j in range(0, data_mag[0]):
            for k in range(0, data_mag[1]):
                Total_sumb[1]= Total_sumb[1] + (data[n][j][k]) ** b1
a1=1000*(F1-F2)/(Total_sumb[0]-Total_sumb[1])

print("Result of using first order NR method")
print("The exponent of model function is", b1)
print("The coefficient model function is", a1)
print("The number of iteration is", NR1(f,df)[1])

# Algorithm validation program
# Process for calculating total force based on calculated model function
for n in range(0,2):
    if n==0:
        for j in range(0, data_mag[0]):
            for k in range(0, data_mag[1]):
                Total_sumc[0] = Total_sumc[0] + a1*(data[n][j][k]) ** b1

    elif n==1:
        for j in range(0, data_mag[0]):
            for k in range(0, data_mag[1]):
                Total_sumc[1]= Total_sumc[1] + a1*(data[n][j][k]) ** b1
print('The true force of DataA[N]',F1,'The true force of DataB[N]', F2)
print('The calibrated force of DataA[N] :',Total_sumc[0]/1000, 'The calibrated force of DataB[N] :',Total_sumc[1]/1000)

# Read raw data C,D(not participating in two point calibration)
dir3='C:\\Users\\A\\Desktop\\SFC1900CXR2-22.06.07-20220607T113023_220624(2번가압) (1)\\bias_valid_test\\1bar\\22.02.28.39'
dir4='C:\\Users\\A\\Desktop\\SFC1900CXR2-22.06.07-20220607T113023_220624(2번가압) (1)\\bias_valid_test\\7bar\\22.02.28.39'
excelFile3 = os.path.join(dir3, '22.02.28.39-2d.xlsx')
data3 = pd.read_excel(excelFile3)
dataC = np.array(data3,dtype='int32')
excelFile4 = os.path.join(dir4, '22.02.28.39-2d.xlsx')
data4 = pd.read_excel(excelFile4)
dataD = np.array(data4,dtype='int32')
dataT=np.zeros((2,data_mag[0],data_mag[1]),dtype='int32') # Define variables for storing two 48x48 matrices in a 3d array structure
Total_sumcT=np.zeros((2)) # Define variables for calculating total force based on calculated model function
for n in range(0,2):
    if n==0:
        for j in range(0, data_mag[0]):
            for k in range(0, data_mag[1]):
                dataT[n][j][k] = dataC[j][k]
    elif n==1:
        for j in range(0, data_mag[0]):
            for k in range(0, data_mag[1]):
                dataT[n][j][k] = dataD[j][k]
for n in range(0,2):
    if n==0:
        for j in range(0, data_mag[0]):
            for k in range(0, data_mag[1]):
                Total_sumcT[0] = Total_sumcT[0] + a1*(dataT[n][j][k]) ** b1

    elif n==1:
        for j in range(0, data_mag[0]):
            for k in range(0, data_mag[1]):
                Total_sumcT[1]= Total_sumcT[1] + a1*(dataT[n][j][k]) ** b1

print('The calibrated force of DataC[N] :',Total_sumcT[0]/1000, 'The calibrated force of DataD[N] :',Total_sumcT[1]/1000)

# Percent relative error
F1PRE1= abs(F1-Total_sumc[0]/1000)/F1*100
F2PRE1=abs(F2-Total_sumc[1]/1000)/F2*100
F1PRE11= abs(F1-Total_sumcT[0]/1000)/F1*100
F2PRE11=abs(F2-Total_sumcT[1]/1000)/F2*100
print('------------------------------------------------------------------Result of calibration-----------------------------------------------------------------')
d=[['1bar(Data A)', F1/9.8, Total_sumc[0]/9800, round(Total_sumc[0]/9800,3), F1PRE1],
   ['7bar(Data B)', F2/9.8, Total_sumc[1]/9800, round(Total_sumc[1]/9800,3), F2PRE1],
   ['1bar(Data C)', F1/9.8, Total_sumcT[0]/9800, round(Total_sumcT[0]/9800,3), F1PRE11],
   ['7bar(Data D)', F2/9.8, Total_sumcT[1]/9800, round(Total_sumcT[1]/9800,3), F2PRE11]]
print(tabulate(d,headers=["True stress","True force[kgf]","Calibrated force[kgf]","Calibration force rounded to the third decimal place[kgf]","True percent relative error[%]"]))

# Plotting
xx=np.linspace(0,255,500)
yy=a1*(xx**b1)/9.8
plt.plot(xx,yy)
plt.xlabel('DO(Digital output)')
plt.ylabel('Total force[kgf]')
plt.title('Result of Curve fitting')
plt.grid()
plt.show()