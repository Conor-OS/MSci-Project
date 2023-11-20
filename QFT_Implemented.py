
import numpy as np
import matplotlib.pyplot as plt


'''
=====================================
n = 2 QUBIT QUANTUM FOURIER TRANSFORM
=====================================
'''


### Initialize variables ###

# Create qubit states |0> and |1>
qubit_state_0 = np.array([[1], [0]]) 
qubit_state_1 = np.array([[0], [1]])

# Create outer products |0><0| and |1><1|
operator_00 = qubit_state_0.transpose() * qubit_state_0
#print(operator_00)
operator_11 = qubit_state_1.transpose() * qubit_state_1
#print(operator_11)

# Create Hadamard gate
H = (1/np.sqrt(2))*np.array([[1, 1], 
                             [1, -1]])
# Create Rotation gate as a function of k
def R(k):
    R = np.array([[1, 0],
                  [0, np.exp(2j*np.pi/(2**k))]])
    return R


# Create Identity gate
I = np.array([[1, 0], [0, 1]])

# Create Swap gate
SWAP = np.array([[1, 0, 0, 0],
                 [0, 0, 1, 0],
                 [0, 1, 0, 0],
                 [0, 0, 0, 1]])

### CONSTRUCTING QFT ###

# Tensor products of H and I gates for each end of n=2 QFT
HxI = np.kron(H, I)
#print(HxI)
IxH = np.kron(I, H)
#print(IxH)

# Creating conditional-rotation gate
op_00xI = np.kron(operator_00, I)
op_11xS = np.kron(operator_11, R(2))
#print(op_00xI)
#print(op_11xS)
CR = op_00xI + op_11xS
#print(np.matmul(CR, IxH))

# Creating Pre-swapped QFT
QFT_tilda = np.matmul(HxI, np.matmul(CR, IxH))
#print(QFT_tilda)

# Creating Post-swapped QFT
QFT_final = np.matmul(SWAP, QFT_tilda)

print(QFT_final)



'''
=====================================
n = 3 QUBIT QUANTUM FOURIER TRANSFORM
=====================================
'''


M2 = np.kron(H, np.kron(I, I))

#N1 = np.kron(R(2), np.kron(I, I))
#N1 = np.kron(I, np.kron(R(2), I))
N1 = np.kron(operator_11, np.kron(R(2), I)) + np.kron(operator_00, np.kron(I, I))
#print(N1)

M1 = np.kron(I, np.kron(H, I))


Nn1 = np.kron(operator_11, np.kron(I, R(2))) + np.kron(operator_00, np.kron(I, I))
Nn2 = np.kron(operator_11, np.kron(I, R(3))) + np.kron(operator_00, np.kron(I, I))
#N0 = np.matmul(np.kron(I, np.kron(I, R(2))), np.kron(I, np.kron(I, R(3))))
N0 = np.matmul(Nn2, Nn1)
#N0 = np.matmul(np.kron(I, np.kron(R(2), I)), np.kron(R(3), np.kron(I, I)))

M0 = np.kron(I, np.kron(I, H))

'''
DIFFERENT CONTRACTION ORDERS
I believe only `preswap2` is correct

By varying contraction order, circuit can be more efficiently simulable

'''

#preswap = np.matmul(M2, np.matmul(N1, np.matmul(M1, np.matmul(N0, M0))))

preswap2 = np.matmul(M0, np.matmul(N0, np.matmul(M1, np.matmul(N1, M2))))

#print(preswap*2**(3/2))

print(preswap2*2**(3/2))


SWAP3 = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 1, 0, 0, 0],
                  [0, 0, 1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 1, 0],
                  [0, 1, 0, 0, 0 ,0, 0, 0],
                  [0, 0, 0, 0, 0, 1, 0, 0],
                  [0, 0, 0, 1, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 1]])

print("===================")

### THIS IS CORRECT ###
print(np.matmul(SWAP3, preswap2*2**(3/2)))







qubit_state_00 = np.kron(qubit_state_0, qubit_state_0)
qubit_state_01 = np.kron(qubit_state_0, qubit_state_1)
qubit_state_10 = np.kron(qubit_state_1, qubit_state_0)
qubit_state_11 = np.kron(qubit_state_1, qubit_state_1)


operator_0000 = qubit_state_00 * qubit_state_00.transpose()
operator_0101 = qubit_state_01 * qubit_state_01.transpose()
operator_1010 = qubit_state_10 * qubit_state_10.transpose()
operator_1111 = qubit_state_11 * qubit_state_11.transpose()

#print(np.kron(operator_0000, I))

#print(np.kron(operator_0101, I))

#print(np.kron(operator_1010, I))

#print(np.kron(operator_1111, R(2)))

CR2 = np.kron(operator_0000, I) + np.kron(operator_0101, I) + np.kron(operator_1010, I) +np.kron(operator_1111, R(2))

#print(CR2)
#print("=========================")

#print(np.kron(operator_11, np.kron(R(2), I)) + np.kron(operator_00, np.kron(I, I)))
#print(np.kron(operator_00, np.kron(I, I)))

#print(np.kron(operator_11, np.kron(I, R(2))))




'''
=====================================
Testing QFT via signal processing

Unsurprisingly, doesnt work for l = 4 (n=2 qubits) very accurately
=====================================
'''

a=2
l = 2**a
x = np.linspace(0,20,l)
freqs = [0.5, 1.2, 4.3 ,2.7 ,1.9, np.pi] # add frequency components to this list to make arbitrarily complicated data
y = np.zeros(l)

for i in freqs:
    y = y + np.cos(x*i*2*np.pi) 

y = y #+ 1*np.random.rand(l) # uncomment for noise

#plt.plot(x, y)

def Plots(y, l):
    y_npfft = np.fft.fft(y)
    
    y_qft = np.matmul(QFT_final, y)

    
    freq = np.fft.fftfreq(l, max(x)/l) # generate the corresponding frequency list
    
    # the frequency components you added in will show up in the Fourier transform
    
    fig, ax = plt.subplots()
    ax.plot(freq, y_npfft, label='NumPy FFT')
    #ax.plot(freq, y_qft, label='QFT')
    
    ax.legend()
    #ax.set_xlim([-6,6])

#Plots(y, l)










