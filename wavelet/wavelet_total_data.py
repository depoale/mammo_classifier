import numpy as np
from matplotlib import pyplot as plt
from matplotlib import image as img
import matlab.engine
import os
from PIL import Image

eng = matlab.engine.start_matlab()
print(eng)

Q = 2 #quante dev std considero nel filtraggio

dataset_path_0 = './DATA/0'
dataset_path_1='./DATA/1'

names_0 = os.listdir(dataset_path_0)
names_1 = os.listdir(dataset_path_1)


for name in names_1:
    I = eng.imread(f'./DATA/1/{name}')
    I = np.asarray(I)

    wave = 'sym3'
    N = 3
    c, s = eng.wavedec2(I, N, wave, nargout=2)
    s = np.asarray(s)
    c = np.asarray(c)
    c = c[0]

    A1 = eng.appcoef2(c,s,wave,1, nargout=1)
    H1,V1,D1 = eng.detcoef2('all',c,s,1, nargout=3)

    A2 = eng.appcoef2(c,s,wave,2, nargout=1)
    H2,V2,D2 = eng.detcoef2('all',c,s,2, nargout=3)

    A3 = eng.appcoef2(c,s,wave,3, nargout=1)
    H3,V3,D3 = eng.detcoef2('all',c,s,3, nargout=3)

    size_CM = eng.prod(s,2, nargout=1)
    size_CM = np.asarray(size_CM)

    c_labels= eng.zeros(1,size_CM[0])
    c_labels = np.asarray(c_labels)
    c_labels = c_labels[0]

    for il in range(1, N+1):
        ones = eng.ones(1,3*np.double(size_CM[il]))
        ones = np.asarray(ones)
        ones = ones[0]
        c_labels = np.concatenate((c_labels, np.double(N+1-il)*ones))

    std1=np.double(eng.std(c[c_labels==1], nargout=1))
    std2=np.double(eng.std(c[c_labels==2], nargout=1))
    std3=np.double(eng.std(c[c_labels==3], nargout=1))

    c_mod = c.copy()
    c_mod.setflags(write=1)
    c_mod[c_labels==0]=0

    c_mod[(c_labels==1)&(abs(c)<Q*std1)]=0
    c_mod[(c_labels==2)&(abs(c)<Q*std2)]=0
    c_mod[(c_labels==3)&(abs(c)<Q*std3)]=0

    I_rec = eng.waverec2(c_mod,s,wave, nargout=1)
    I_rec = np.asarray(I_rec)

    #plt.figure('rec_wavelet')
    #plt.imshow(I_rec, cmap='gray')
    plt.imsave(f'./DATA_WAVELET/1/{name}.png', I_rec, cmap='gray', format='png')
    Image.open(f'./DATA_WAVELET/1/{name}.png').convert('L').save(f'./DATA_WAVELET/1/{name}.png')
    #plt.show()


for name in names_0:
    I = eng.imread(f'./DATA/0/{name}')
    I = np.asarray(I)

    wave = 'sym3'
    N = 3
    c, s = eng.wavedec2(I, N, wave, nargout=2)
    s = np.asarray(s)
    c = np.asarray(c)
    c = c[0]

    A1 = eng.appcoef2(c,s,wave,1, nargout=1)
    H1,V1,D1 = eng.detcoef2('all',c,s,1, nargout=3)

    A2 = eng.appcoef2(c,s,wave,2, nargout=1)
    H2,V2,D2 = eng.detcoef2('all',c,s,2, nargout=3)

    A3 = eng.appcoef2(c,s,wave,3, nargout=1)
    H3,V3,D3 = eng.detcoef2('all',c,s,3, nargout=3)

    size_CM = eng.prod(s,2, nargout=1)
    size_CM = np.asarray(size_CM)

    c_labels= eng.zeros(1,size_CM[0])
    c_labels = np.asarray(c_labels)
    c_labels = c_labels[0]

    for il in range(1, N+1):
        ones = eng.ones(1,3*np.double(size_CM[il]))
        ones = np.asarray(ones)
        ones = ones[0]
        c_labels = np.concatenate((c_labels, np.double(N+1-il)*ones))

    std1=np.double(eng.std(c[c_labels==1], nargout=1))
    std2=np.double(eng.std(c[c_labels==2], nargout=1))
    std3=np.double(eng.std(c[c_labels==3], nargout=1))

    c_mod = c.copy()
    c_mod.setflags(write=1)
    c_mod[c_labels==0]=0

    c_mod[(c_labels==1)&(abs(c)<Q*std1)]=0
    c_mod[(c_labels==2)&(abs(c)<Q*std2)]=0
    c_mod[(c_labels==3)&(abs(c)<Q*std3)]=0

    I_rec = eng.waverec2(c_mod,s,wave, nargout=1)
    I_rec = np.asarray(I_rec)

    #plt.figure('rec_wavelet')
    #plt.imshow(I_rec, cmap='gray')
    plt.imsave(f'./DATA_WAVELET/0/{name}.png', I_rec, cmap='gray', format='png')
    Image.open(f'./DATA_WAVELET/0/{name}.png').convert('L').save(f'./DATA_WAVELET/0/{name}.png')
    #plt.show()
