import scipy.io

# Se lee el archivo .mat
mat = scipy.io.loadmat('jaccard.mat')

# Se guarda en variables la segmentación y la segmentación perfecta
real = mat['GroundTruth']
seg = mat['Segmentation']

intersection = 0
union = 0

# Se calculan los elementos en la intersección y en la unión
for i in range(len(seg)):
    for j in range(len(seg[i])):
        for k in range(len(seg[i,j])):
            if seg[i,j,k]==1 and real[i,j,k]==1:
                intersection+=1
            if seg[i,j,k]==1 or real[i,j,k]==1:
                union+=1

# Calculo indice de jaccard
print('Indice de jaccard'+intersection/union)
