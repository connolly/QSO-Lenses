import pylab as plt
import numpy as np

plt.style.use("https://gist.githubusercontent.com/rhiever/a4fb39bfab4b33af0018/raw/9c857ed71cd8361b0b0da2c36404fa7245f3de5f/tableau20.mplstyle") 

def selectBounds(datax):
    scale = (datax.max() - datax.min())*0.1
    x_min, x_max = datax.min() - scale, datax.max() + scale
    return x_min, x_max

def scatter_contour_color(x, y,
                    levels=5,
                    threshold=200,
                    log_counts=False,
                    histogram2d_args=None,
                    plot_args=None,
                    contour_args=None,
                    filled_contour=False,
                    ax=None):
    """Scatter plot with contour over dense regions
    Parameters
    ----------
    x, y : arrays
        x and y data for the contour plot
    levels : integer or array (optional, default=10)
        number of contour levels, or array of contour levels
    threshold : float (default=100)
        number of points per 2D bin at which to begin drawing contours
    log_counts :boolean (optional)
        if True, contour levels are the base-10 logarithm of bin counts.
    histogram2d_args : dict
        keyword arguments passed to numpy.histogram2d
        see doc string of numpy.histogram2d for more information
    plot_args : dict
        keyword arguments passed to plt.plot.  By default it will use
        dict(marker='.', linestyle='none').
        see doc string of pylab.plot for more information
    contour_args : dict
        keyword arguments passed to plt.contourf or plt.contour
        see doc string of pylab.contourf for more information
    filled_contour : bool
        If True (default) use filled contours. Otherwise, use contour outlines.
    ax : pylab.Axes instance
        the axes on which to plot.  If not specified, the current
        axes will be used
    Returns
    -------
    points, contours :
       points is the return value of ax.plot()
       contours is the return value of ax.contour or ax.contourf
    """
    x = np.asarray(x)
    y = np.asarray(y)

    default_contour_args = dict(zorder=2, alpha=0.5, linewidths=1)
    default_plot_args = dict(marker='.', zorder=1)


    if plot_args is not None:
        default_plot_args.update(plot_args)
    plot_args = default_plot_args

    if contour_args is not None:
        default_contour_args.update(contour_args)
    contour_args = default_contour_args

    if histogram2d_args is None:
        histogram2d_args = {}

    if contour_args is None:
        contour_args = {}

    if ax is None:
        # Import here so that testing with Agg will work
        from matplotlib import pyplot as plt
        ax = plt.gca()

    H, xbins, ybins = np.histogram2d(x, y, **histogram2d_args)

    if log_counts:
        H = np.log10(1 + H)
        threshold = np.log10(1 + threshold)

    levels = np.asarray(levels)

    if levels.size == 1:
        levels = np.linspace(threshold, H.max(), levels)

    extent = [xbins[0], xbins[-1], ybins[0], ybins[-1]]

    i_min = np.argmin(levels)

    # draw a zero-width line: this gives us the outer polygon to
    # reduce the number of points we draw
    # somewhat hackish... we could probably get the same info from
    # the full contour plot below.
    outline = ax.contour(H.T, levels[i_min:i_min + 1],
                         linewidths=0, extent=extent,
                         alpha=0)

    if filled_contour:
        contours = ax.contourf(H.T, levels, extent=extent, **contour_args)
    else:
        contours = ax.contour(H.T, levels, extent=extent, **contour_args)

    X = np.hstack([x[:, None], y[:, None]])

    if len(outline.allsegs[0]) > 0:
        outer_poly = outline.allsegs[0][0]
        try:
            # this works in newer matplotlib versions
            from matplotlib.path import Path
            points_inside = Path(outer_poly).contains_points(X)
        except:
            # this works in older matplotlib versions
            import matplotlib.nxutils as nx
            points_inside = nx.points_inside_poly(X, outer_poly)

        Xplot = X[~points_inside]
    else:
        Xplot = X
        
    points = ax.scatter(Xplot[:, 0], Xplot[:, 1],  **plot_args)

    return points, contours

def plotPairwise(data, fig, labels=None, mixtures=None, limits=None, 
                 plot_args=None, contour_args=None, filled_contour=False,
                 **kwargs):
    '''Plot a set of pairwise correlations using scatter_contour_color'''
    
    npts, nattr = data.shape
    if labels is None:
        labels = ['var%d'%i for i in range(nattr)]

    default_plot_args = dict(marker='.', zorder=1)
    if plot_args is not None:
        default_plot_args.update(plot_args)
    plot_args = default_plot_args


    default_contour_args = dict(zorder=2, alpha=0.5, linewidths=1)
    if contour_args is not None:
        default_contour_args.update(contour_args)
    contour_args = default_contour_args




    for i in range(nattr): #row
        for j in range(nattr): # column
            nSub = i * nattr + j + 1
            ax = fig.add_subplot(nattr, nattr, nSub)
            if i == j:
                ax.hist(data[:,i], bins=100)
                if (limits != None):
                    ax.set_xlim(limits[i])
            else:
                scatter_contour_color(data[:,j], data[:,i], threshold=200, log_counts=False, ax=ax,
                                histogram2d_args=dict(bins=20), filled_contour=filled_contour,
                                plot_args=plot_args,
                                contour_args=contour_args)

                # plt fit ellipses
                if mixtures != None:
                    for k in range(mixtures.n_components):
                        mean = mixtures.means_[k][[j,i]]
                        cov = mixtures.covars_[k][[j,i]][:,[j,i]]
                        if cov.ndim == 1:
                            cov = np.diag(cov)
                        draw_ellipse(mean, cov, ax=ax, fc='none', ec='k', zorder=2, scales=[1])
                if (limits != None):
                    ax.set_xlim(limits[j])
                    ax.set_ylim(limits[i])
                else:
                    amin, amax = selectBounds(data[:,j])
                    ax.set_xlim((amin,amax))
                    amin, amax = selectBounds(data[:,i])
                    ax.set_ylim((amin,amax))
            if (i==0):
                ax.set_title(labels[j])
            if (j==0):
                ax.set_ylabel(labels[i])

 

def plotPairwiseClassifier(data_test, type_test, type_pred, fig, colormap=np.array(['k','y','b','r']),
                           labels=None, limits=None, nbins = 11, plot_args=None, contour_args=None, 
                           filled_contour=False, **kwargs):
    '''Plot a set of pairwise correlations and their classifications'''
    
    # set up plotting parameters and dimensions
    npts, nattr = data_test.shape
    if labels is None:
        labels = ['var%d'%i for i in range(nattr)]

    default_plot_args = dict(marker='.', zorder=1)
    if plot_args is not None:
        default_plot_args.update(plot_args)
    plot_args = default_plot_args


    default_contour_args = dict(zorder=2, alpha=0.5, linewidths=1)
    if contour_args is not None:
        default_contour_args.update(contour_args)
    contour_args = default_contour_args
   
    for i in range(nattr): #row
        for j in range(nattr): # column
            nSub = i * nattr + j + 1
            ax = fig.add_subplot(nattr, nattr, nSub)
            if i == j:
                for name in np.unique(type_test):
                    ax.hist(data_test[:,i][type_test==name], nbins, color=colormap[name], 
                            linestyle='solid',normed=True, histtype='step')
                    ax.hist(data_test[:,i][type_pred==name], nbins, color=colormap[name], 
                            linestyle='dashed',normed=True, histtype='step')
                if (limits != None):
                    ax.set_xlim(limits[i])
            else:
                if (limits != None):
                    ax.set_xlim(limits[j])
                    ax.set_ylim(limits[i])
                else:
                    amin, amax = selectBounds(data_test[:,j])
                    ax.set_xlim((amin,amax))
                    amin, amax = selectBounds(data_test[:,i])
                    ax.set_ylim((amin,amax))
                
                ax.scatter(data_test[:,j],data_test[:,i], marker='.', edgecolors='None',
                           c=colormap[type_pred],zorder=5)
            if (i==0):
                ax.set_title(labels[j])
            if (j==0):
                ax.set_ylabel(labels[i])


