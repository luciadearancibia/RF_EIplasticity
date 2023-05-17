import numpy as np
import matplotlib.pyplot as plt

def OU_input_signal(decay=0.05, c=6, mean=1, sigma=0.1, dur=100., dt=1):
    """Generates input signals from independent Ornstein-Uhlenbeck processes"""
    
    # parameters
    x0 = sigma*np.random.standard_normal(size=1)     # initial condition of x at time 0

    # initialize variables
    t = np.arange(0., dur, dt)
    x = np.zeros_like(t)
    x[0] = x0

    # Step through in time
    for k in range(len(t)-1):
        x[k+1] = x[k] - decay * (x[k] - c) + sigma * np.random.standard_normal(size=1)

    # set negative values to 0
    #x = x-c
    x[x<0] = 0

    # rescale to have mean of 1
    x = x/np.mean(x) #x = x + (1- np.mean(x))

    if np.any(x<0):
        raise ValueError('Negative values on input signal')
    
    # x[x<0] = 0

    return x #, sparsity(x)



def sparsity(s):
    """Compute sparsity of the signal"""

    return np.mean(s,axis=1)**2/np.mean(s**2,axis=1)


def input_filters(signals, sigma=3, n_signals=None, n_filters=None, norm=True, plot_gauss=False):
    """Compute the output of the E/I populations that filter the input signal"""
    # inputs
    if n_signals is None: n_signals = signals.shape[0]
    if n_filters is None: n_filters = n_signals

    # n_signals = signal.shape[2]
    #print('Number of signals %i, number of filters %i, signal duration %i' 
    #      % (n_signals, n_filters, signals.shape[1]))
    j = np.arange(n_signals)
    filter_out = np.zeros((n_filters, signals.shape[1]))

    # loop over filters
    for i in range(n_filters): 
        ij = (i*np.ones_like(j) - j)
        ij[np.abs(ij)>5] = 11 - np.abs(ij[np.abs(ij)>5])  # circulat tuning
        T = np.exp(-(ij**2) / (2*sigma**2))

        if norm: T = T/np.sum(T)
        #print(-(i*np.ones_like(j) - j)**2)
        #print(T)
        
        # row-wise multiplication, then sum over signal
        filter_out[i,:] = np.sum(T.reshape([-1,1]) * signals, axis=0)

    if np.any(filter_out<0):
        raise ValueError('Negative values on population activities')
    
    # if norm: 
    #     filter_out = filter_out/np.max(filter_out, axis=1).reshape(-1,1) # normalize signals (so max=1)

    if plot_gauss:
        t = np.exp(-(np.arange(-n_signals, n_signals+1))**2 / (2*sigma**2))
        plt.plot(np.arange(-n_signals, n_signals+1), t) #/np.sum(t))
        plt.title('Gaussian filters')
        plt.xlabel('input signal')
        plt.gca().spines['left'].set_visible(False)
        plt.yticks([])
        plt.xlim(-n_signals-.5,n_signals+.5)

    return filter_out



def output_rate(E, I, W_E, W_I, **other):
    ''' subthreshold activity of the output neuron (rate) over time
        **other : other populations (dishinibitory or whatever)
                  dicctionary with keys:
                  - pops    : list of population activity
                  - weights : list of popultion weights
    '''

    #if len(W_E.shape)==0: rate = (W_E.T @ E) - (W_I.T @ I)
    # else: 
    rate = (W_E.T @ E) - (W_I @ I)
    if other:
        for idx, pop in enumerate(other['pops']):
            rate = rate - (other['weights'][idx] @ pop)

    if type(rate)==np.ndarray: rate[rate<0] = 0.
    elif rate<0: rate = 0

    return rate


def Hebbian_plasticity(W_0, presyn, rate, lr, target_rate=0., norm_type='subt'):
    ''' W_(t+1) = W_t + n*E*(R-target_rate)
        lr :  learning rate
        target_rate:  0 for classic Hebbian plasticity
    '''
    deltaW = lr * presyn * (rate - target_rate)
    # print('deltaW: ', deltaW)
    W = W_0 + deltaW #np.cumsum(np.concatenate(W_0, deltaW))
    
    W = norm(W,type=norm_type)

    return W


def antiHebbian_plasticity(W_0, presyn, rate, lr, target_rate=0., norm_type='subt'):
    ''' W_(t+1) = W_t - n*E*(R-target_rate)
        lr :  learning rate
        target_rate:  0 for classic Hebbian plasticity
    '''
    deltaW = -lr * presyn * (rate - target_rate)
    #print('deltaW: ', deltaW)
    W = W_0 + deltaW #np.cumsum(np.concatenate(W_0, deltaW))
    
    W = norm(W,type=norm_type)

    return W


def Hebbian_plasticity_saturation(W_0, presyn, rate, lr, target_rate=0., w_max=1):
    ''' Hebbian plasticity with saturation has w reaches the maximum value w_max
        W_(t+1) = W_t + n*E*(R-target_rate)
        lr :  learning rate
        target_rate:  0 for classic Hebbian plasticity
    '''
    deltaW = lr * (w_max - W_0) * presyn * (rate - target_rate)
    #print('deltaW: ', deltaW)
    W = W_0 + deltaW #np.cumsum(np.concatenate(W_0, deltaW))
    
    return W


def Ojas_plasticity(W_0, presyn, rate, lr, target_rate=0.):
    ''' Δwi = n * (x_i y - y^2 w_i),    i=1,...,n
        lr :  learning rate
        target_rate:  0 for classic Hebbian plasticity
    '''
    deltaW = lr * (presyn * (rate - target_rate) - (rate**2 * W_0))
    #print('deltaW: ', deltaW)
    W = W_0 + deltaW #np.cumsum(np.concatenate(W_0, deltaW))
    
    return W


def norm(W, type):
    if  type=='mult':
        W_norm = W / np.sqrt(np.sum(W**2))
    elif  type=='subt': 
        W_norm = np.copy(W) - np.mean(W) + 1
    else:
        W_norm = np.copy(W)    
    W_norm = np.clip(W_norm,0,np.inf) #W[W<0] = 0
    if np.any(W_norm<0): raise ValueError('Negative values after normalization')

    return W_norm


def simulate(W_E_0, W_I_0, E_input, I_input,
             E_plasticity=Hebbian_plasticity, I_plasticity=Hebbian_plasticity, Other_plasticity=Hebbian_plasticity,
             paramsE={}, paramsI={}, paramsOther={}):
    '''  *params: 
            for Hebbain_plasticity: norm_type
            paramsOther : params for other populations in the network
    '''
    # Initialize output rate
    rate = np.zeros(E_input.shape[1])
    rate[0] = output_rate(E_input[:,0], I_input[:,0], W_E_0, W_I_0)

    # initialize weights
    W_E = np.zeros_like(E_input)
    W_E[:,0] = W_E_0
    W_I = np.zeros_like(I_input)
    W_I[:,0] = W_I_0

    W_O = []
    if paramsOther:
        for idx, input in enumerate(paramsOther['inputs']):
            wO = np.zeros_like(input)
            wO[:,0] = paramsOther['W0'][idx]
            W_O.append(wO)
        
    # Step through in time
    for t in range(E_input.shape[1]-1):
        #print('t = ',t)
        w_e = W_E[:,t]
        w_i = W_I[:,t]

        W_E[:,t+1] = E_plasticity(w_e, E_input[:,t], rate[t], **paramsE)
        W_I[:,t+1] = I_plasticity(w_i, I_input[:,t], rate[t], **paramsI)

        if paramsOther:
            for idx, input in enumerate(paramsOther['inputs']):
                w_o = W_O[idx][:,t]
                plast = paramsOther['plasticity'][idx]
                plastParams = paramsOther['plasticityParms'][idx]
                W_O[idx][:,t+1] = plast(w_o, input[:,t], rate[t], target_rate=paramsOther['target_rate'][idx], norm_type=paramsOther['norm_type'][idx])

            rate[t+1] = output_rate(E_input[:,t+1], I_input[:,t+1], W_E[:,t+1], W_I[:,t+1], pops=paramsOther['inputs'], weights=W_O)

        else:
            rate[t+1] = output_rate(E_input[:,t+1], I_input[:,t+1], W_E[:,t+1], W_I[:,t+1])

    return rate, W_E, W_I