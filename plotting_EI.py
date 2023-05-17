import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_dynamics(x, t, lam, c, mean, xinfty=0):
    """ Plot the dynamics """
    fig = plt.figure()
    plt.title('$\lambda=%0.1f, c=%0.1f, mean=%0.1f$' % (lam, c, mean), fontsize=16)
    x0 = x[0]
    plt.plot(t, x0 * lam**t, 'r', label='analytic solution')
    plt.plot(t, x, 'k', label='simulation')          # simulated data pts
    #plt.ylim(0, x0+1)

    plt.xlabel('time (ms)')
    plt.ylabel('x')
    plt.legend()
    plt.show()

    return None


def plot_simulation(signals, E_filters, I_filters, output_neuron, time=None, a=(6,14)):

    if time is None: 
        time = np.arange(0,signals.shape[1],1)

    n_signals = signals.shape[0]
    n_E_filters = E_filters.shape[0]
    n_I_filters = I_filters.shape[0]

    scale_s = np.ceil(np.max(signals))
    scale_E = np.ceil(np.max(E_filters))
    scale_I = np.ceil(np.max(I_filters))
    r = int(np.floor(n_E_filters/(n_E_filters+n_I_filters))*20)+2

    fig, axs = plt.subplots(ncols=3, nrows=21, figsize=(15,6))
    plt.tight_layout()
    gs = axs[1, 2].get_gridspec()

    sns.set_palette('Dark2', n_signals)
    # first column: single figure all rows  
    # remove the underlying axes
    for ax in np.concatenate(axs):
        ax.remove()  
    plt.subplot(gs[:, 0])
    plt.title('input signals')
    plt.plot(time, signals.T + np.arange(0,n_signals*scale_s,scale_s),color='silver',linewidth=1.)  
    plt.yticks(np.arange(0,n_signals*scale_s,scale_s), labels=np.arange(1,n_signals+1))     
    #plt.ylim(-5., (n_signals+1)*scale_s)
    plt.xlabel('time (ms)',fontsize=16)  

    # second column: two figures (E and I) with varying distribution  
    sns.set_palette('OrRd', n_E_filters)
    plt.subplot(gs[:9, 1])
    plt.plot(time, E_filters.T + np.arange(0,n_E_filters*scale_E,scale_E),linewidth=1.)         #label='$E_{%i}$' % i, 
    plt.yticks(np.arange(0,n_E_filters*scale_E,scale_E), labels=np.arange(1,n_E_filters+1))     
    # plt.title('$\sigma_E=%1.1f$' % sigma_E, fontsize=14)
    #plt.ylim(-5., (n_E_filters+1)*scale_E)
    plt.ylabel('E filter #',fontsize=16)
    plt.xticks([])
    plt.gca().spines['bottom'].set_visible(False)

    sns.set_palette('Blues', n_I_filters)
    plt.subplot(gs[11:, 1])
    plt.plot(time, I_filters.T + np.arange(0,n_I_filters*scale_I,scale_I),linewidth=1.)         #label='$E_{%i}$' % i, 
    plt.ylabel('I filter #',fontsize=16)  
    #plt.legend(bbox_to_anchor=(.9, 1.05))
    # plt.title('$\sigma_I=%1.1f$' % sigma_I, fontsize=14)
    #plt.ylim(-2., 25)
    plt.xlabel('time (ms)',fontsize=16)  
    plt.yticks(np.arange(0,n_I_filters*scale_I,scale_I), labels=np.arange(1,n_I_filters+1))     

    plt.subplot(gs[a[0]:a[1], 2])  # rows, columns
    plt.plot(time, output_neuron.T, linewidth=1., color='#2166ac')         #label='$E_{%i}$' % i, 
    plt.title('output rate')
    # plt.yticks([])
    #plt.show()
    plt.xlabel('time (ms)',fontsize=16)  

    return fig, gs


def plot_weights(W_E, W_I, fig, extent=None):

    W = np.concatenate((W_E, W_I), axis=0)
    W_ee = np.copy(W)
    W_ee[W_E.shape[0]:, :] = np.nan #W_ee[-1, :] = np.nan #
    data1 = np.ma.masked_where(np.isnan(W_ee), W)

    W_ii = np.copy(W)
    W_ii[:W_E.shape[0], :] = np.nan #W_ii[:-1, :] = np.nan
    data2 = np.ma.masked_where(np.isnan(W_ii), W)

    if extent is None: extent = (0, W_E.shape[1], 0, W.shape[0])

    im1 = plt.imshow(data1, cmap=sns.color_palette("Reds", as_cmap=True), aspect=100, vmin=0, extent=extent)
    im2 = plt.imshow(data2, cmap=sns.color_palette("Blues", as_cmap=True), aspect=100, vmin=0, extent=extent)

    plt.gca().spines['bottom'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.xlabel('Time (ms)')
    plt.ylabel('input neuron #')
    # im1 = ax.imshow(W[:W_E.shape[0]-1, :], cmap=sns.color_palette("Reds_r", as_cmap=True), aspect=20, vmin=np.min(W), vmax=np.max(W))
    #im2 = ax.imshow(W[W_E.shape[0]:, :], cmap=sns.color_palette("Blues_r", as_cmap=True), aspect=200, vmin=np.min(W), vmax=np.max(W))

    cb1 = plt.colorbar(im1, shrink=0.8)
    cb1.outline.set_visible(False)
    cb2 = plt.colorbar(im2, shrink=0.8)
    cb2.outline.set_visible(False)


def plot_RF(t, W_E, W_I, n_signals=None, en_idx=None, plot_net=False):
    if n_signals is None: n_signals = n_E.shape[0]

    n_E = W_E.shape[0]
    n_I = W_I.shape[0]

    dur = np.copy(t[-1])

    sns.set_palette('Reds',n_E)
    fig, ax = plt.subplots(1,2,figsize=(12,4))
    for i in range(n_E): 
        ax[0].plot(t, W_E[i,:], label='$E_{%i}$' % i, linewidth=1.5)
    plt.hlines(0,-100,dur,'k')
    # ax[0].legend(bbox_to_anchor=(1.,1))
    # ax.set_title(r'$\sigma_E = %1.1f$, weight normalization: %s' % (sigma_E, normtype), fontsize=12)
    # ax[0].plot(t, -W_I.T/10, color='#2b7bba', linewidth=2)
    #ax[0].set_yticklabels([])
    bl  = sns.color_palette('Blues',n_I).as_hex()
    for i in range(n_I): 
        ax[0].plot(t, -W_I[i,:], label='$I_{%i}$' % i, linewidth=1.5, color = bl[i])
    ax[0].set_xlabel('time (a.u.)')
    ax[0].set_ylabel('syn. weight')
    ax[0].set_xlim(-50,dur+100)

    if en_idx is None:
        en_idx = np.where(t==980.)[0][0]
        print(en_idx)
    W_Ee = W_E[:,en_idx] #/n_signals
    W_Ie = W_I[:,en_idx]

    ax[1].plot(np.arange(1,n_signals+1), W_Ee, 'o-w', markersize=6)
    if n_I<=1: 
        x_idx = (n_signals+1)/2.
        ax[1].plot(x_idx, -W_Ie, 'o--w', markersize=6, markerfacecolor='none', markeredgewidth=1.5)
        ax[1].hlines(-W_Ie, 1,n_signals, linestyles={'dotted'})
    else: ax[1].plot(np.arange(1,n_signals+1), -W_Ie, 'o--w', markersize=6., markerfacecolor='none')
    ax[1].set_xlim(0,n_signals+1)
    ax[1].set_xlabel('signal #')
    ax[1].set_xticks(np.arange(1,n_signals+1))
    ax[1].set_ylabel('syn. current')

    if plot_net:
        ax[1].plot(np.arange(1,n_signals+1), W_Ee-W_Ie, 'o:', color='grey', markersize=4., markerfacecolor='none')

    return ax
