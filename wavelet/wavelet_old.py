import numpy as np
from matplotlib import pyplot as plt
from matplotlib import image as img

import matlab.engine
eng = matlab.engine.start_matlab()
print(eng)

## UTILIZZO DI MATPLOTLIB PER APRIRE LE IMMAGINI ##
#image = img.imread('0004s1_3_0.pgm_4.png')
#print(image)
#plt.figure('immagine aperta da matplotlib')
#plt.imshow(image, cmap='gray')
#plt.show()

name = '0088t1_1_1_1.pgm_4.png'

I = eng.imread(name)
#print(I)
I = np.asarray(I)
#print(I)
#plt.figure('immagine aperta da matlab')
#plt.imshow(I, cmap='gray')
#plt.show()

wave = 'sym3'
N = 3
c, s = eng.wavedec2(I, N, wave, nargout=2)

#print(s)
s = np.asarray(s)
#print(s)

#print(c)
c = np.asarray(c)
#print(c)

A1 = eng.appcoef2(c,s,wave,1, nargout=1)
H1,V1,D1 = eng.detcoef2('all',c,s,1, nargout=3)

A2 = eng.appcoef2(c,s,wave,2, nargout=1)
H2,V2,D2 = eng.detcoef2('all',c,s,2, nargout=3)

A3 = eng.appcoef2(c,s,wave,3, nargout=1)
H3,V3,D3 = eng.detcoef2('all',c,s,3, nargout=3)

size_CM = eng.prod(s,2, nargout=1)

#print(size_CM[0])
size_CM = np.asarray(size_CM)
#print(size_CM)

c_labels= eng.zeros(1,size_CM[0])

#print(c_labels)
c_labels = np.asarray(c_labels)
#print(c_labels)
#print(eng.size(c_labels))
#print(c_labels[0])
c_labels = c_labels[0]
#print(c_labels)
#print(c_labels[1])
#print(c_labels)

for il in range(1, N+1):
    #print(il)
    ones = eng.ones(1,3*np.double(size_CM[il]))
    ones = np.asarray(ones)
    ones = ones[0]
    c_labels = np.concatenate((c_labels, np.double(N+1-il)*ones))
    #print(c_labels)
    #print(len(c_labels))

#print(eng.size(c))         # DEVONO ESSERE 
#print(eng.size(c_labels))  # UGUALI !!!

#print(c)
#print(np.shape(c))
c = c[0]
#print(c)
#print(c[0])

#print(c_labels)

#print(s)
#print(s[3][1])

#print(c)
#print(c[c_labels==0])
#print(len(c[c_labels==0]))
#print(s[1,:])

## SALTO LA PARTE DI Approx_3 e Details_1 perché non mi funziona reshape sugli arrray di numpy

#print(c[c_labels==1])
#print(c[c_labels==2])
#print(c[c_labels==3])
'''
plt.figure('1')
plt.hist(c[c_labels==1])
std1=eng.std(c[c_labels==1], nargout=1)
std1 = np.asarray(std1)
print(std1)
plt.show()

plt.figure('2')
plt.hist(c[c_labels==2])
std2=eng.std(c[c_labels==2], nargout=1)
std2 = np.asarray(std2)
print(std2)
plt.show()

plt.figure('3')
plt.hist(c[c_labels==3])
std3=eng.std(c[c_labels==3], nargout=1)
std3 = np.asarray(std3)
print(std3)
plt.show()
'''
std1=np.double(eng.std(c[c_labels==1], nargout=1))
std2=np.double(eng.std(c[c_labels==2], nargout=1))
std3=np.double(eng.std(c[c_labels==3], nargout=1))

#print(std1)
#print(std2)
#print(std3)

c_mod = c.copy()
c_mod.setflags(write=1)

c_mod[c_labels==0]=0

c_mod[(c_labels==1)&(abs(c)<1*std1)]=0
c_mod[(c_labels==2)&(abs(c)<1*std2)]=0
c_mod[(c_labels==3)&(abs(c)<1*std3)]=0

I_rec = eng.waverec2(c_mod,s,wave, nargout=1)
I_rec = np.asarray(I_rec)

#plt.figure('rec_wavelet')
#plt.imshow(I_rec, cmap='gray')
plt.imsave(f'./DATA/{name}.png', I_rec, cmap='gray', format='png')
#plt.show()






























