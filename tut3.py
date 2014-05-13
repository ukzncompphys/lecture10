import numpy
from matplotlib import pylab as plt
def simulate_gaussian(t,sig=0.5,amp=1,cent=0):
    dat=numpy.exp(-0.5*(t-cent)**2/sig**2)*amp
    dat+=numpy.random.randn(t.size)
    return dat

#get a trial step. Take gaussian random numbers and scale by an input vector
def get_trial_offset(sigs):
    return sigs*numpy.random.randn(sigs.size)

class Gaussian:
    def __init__(self,t,sig=0.5,amp=1.0,cent=0,offset=0):
        self.t=t
        self.y=simulate_gaussian(t,sig,amp,cent)+offset
        self.err=numpy.ones(t.size)
        self.sig=sig
        self.amp=amp
        self.cent=cent
        self.offset=offset

    def get_chisq(self,vec):
        sig=vec[0]
        amp=vec[1]
        cent=vec[2]
        off=vec[3]

        pred=off+amp*numpy.exp(-0.5*(self.t-cent)**2/sig**2)
        chisq=numpy.sum(  (self.y-pred)**2/self.err**2)
        return chisq

def run_mcmc(data,start_pos,nstep,scale=None):
    nparam=start_pos.size
    params=numpy.zeros([nstep,nparam+1])
    params[0,0:-1]=start_pos
    cur_chisq=data.get_chisq(start_pos)
    cur_pos=start_pos.copy()
    if scale==None:
        scale=numpy.ones(nparam)
    #to get the accept fraction, let's keep track of it while running the chain.
    tot_accept=0.0
    tot_reject=0.0
    for i in range(1,nstep):
        new_pos=cur_pos+get_trial_offset(scale)
        new_chisq=data.get_chisq(new_pos)
        if new_chisq<cur_chisq:
            accept=True
        else:
            delt=new_chisq-cur_chisq
            prob=numpy.exp(-0.5*delt)
            if numpy.random.rand()<prob:
                accept=True
            else:
                accept=False
        if accept:
            tot_accept=tot_accept+1
            cur_pos=new_pos
            cur_chisq=new_chisq
        else:
            tot_reject=tot_reject+1
        params[i,0:-1]=cur_pos
        params[i,-1]=cur_chisq
        accept_frac=tot_accept/(tot_accept+tot_reject)
    return params,accept_frac


if __name__=='__main__':
    
    #get a realization of a gaussian, with noise added
    t=numpy.arange(-5,5,0.01)
    dat=Gaussian(t,amp=2.5)

    #pick a random starting position, and guess some errors
    guess=numpy.array([0.3,1.2,0.3,-0.2])
    scale=numpy.array([0.1,0.1,0.1,0.1])

    #first do a short chain
    nstep=1000
    chain,accept=run_mcmc(dat,guess,nstep,scale)
    nn=numpy.round(0.2*nstep)
    chain=chain[nn:,:]
    

    #here's our standard deviation of the parameters
    scale=numpy.std(chain[:,0:-1],0)
    
    #now run a chain using the scalings we found.
    #we can just use the final step in the previous chain for our starting value
    #since this is a legitimate chain sample, there's no burn-in
    nstep=30000
    chain,accept2=run_mcmc(dat,chain[-1,0:-1],nstep,scale)
    print "old accept was: " + repr(accept)
    print "new accept is: " + repr(accept2)
    
    #pull true values out, compare to what we got
    param_true=numpy.array([dat.sig,dat.amp,dat.cent,dat.offset])
    for i in range(0,param_true.size):
        val=numpy.mean(chain[:,i])
        scat=numpy.std(chain[:,i])
        print [param_true[i],val,scat]


    
