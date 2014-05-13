import numpy
from matplotlib import pylab as plt

def get_sincos_mat(npt=100,order=10):
    #get an x vector
    xvec=numpy.arange(0,npt)/(0.0+npt)*2*numpy.pi
    #OK, now put the matrix together.  Note that the k=0 mode is strictly real
    #so the total number of vectors is odd
    mat=numpy.zeros((npt,2*order-1))
    mat[:,0]=1.0
    for j in range(1,order):
        mat[:,2*j-1]=numpy.cos(j*xvec)
        mat[:,2*j]=-numpy.sin(j*xvec)
    return numpy.matrix(mat)

if __name__=='__main__':
    n=100
    order=5
    data=numpy.random.randn(n)
    mat=get_sincos_mat(n,order)
    data=numpy.random.randn(n)
    #we take the fft before the transpose, as numpy otherwise does an element-by-element transpose
    datft=numpy.fft.fft(data)
    
    data=numpy.matrix(data).transpose()
    fitp=numpy.linalg.inv(mat.transpose()*mat)*(mat.transpose()*data)
    #recall that there is a normalization ambiguity in FFTs.  Numpy appears to multiply 
    #by n/2 in the forward direction.
    delta3=numpy.complex(fitp[5],fitp[6])-datft[3]/(0.5*n)
    print 'error in 3rd term is ' + repr(delta3)
    

