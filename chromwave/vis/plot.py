import itertools
import os

import numpy
import pandas
from matplotlib import pyplot
from sklearn import metrics
from collections import namedtuple
from pylab import figure, title,  plot, xlabel, ylabel, subplot, ylim, hist, suptitle, gca
from numpy import argsort, std,  sqrt, arange, asarray
from scipy.special import erfinv
from scipy import stats

from . import viz_sequence as viz_sequence

__all__=[
    'plot_roc',
    'plot_scatter'
]

def compute_auc(true, pred, regression = True):
    pred_sorted = sorted(pred, reverse=True)
    # then sort the true values to keep them associated with their prediction
    true_sorted = [x for (z,x) in sorted(zip(pred.flatten(),true), reverse=True)]
    if regression:
        threshold = numpy.mean(true) + 4*numpy.std(true)
        # find values which exceed threshold
        labels = [y > threshold for y in true_sorted]
    else:
        labels = [y >0 for y in true_sorted]

    fpr, tpr, thresholds = metrics.roc_curve(labels, pred_sorted)
    roc_auc = metrics.auc(fpr, tpr)
    return roc_auc,fpr, tpr

def plot_roc(true, pred, output_dir, dataset = 'training',regression=True):
    # sort the predicted values by their intensity
    roc_auc, fpr, tpr = compute_auc(true,pred,regression)
    fig=pyplot.figure()
    pyplot.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    pyplot.plot([0, 1], [0, 1], 'k--')
    pyplot.xlim([0.0, 1.0])
    pyplot.ylim([0.0, 1.05])
    pyplot.xlabel('false positive rate')
    pyplot.ylabel('true positive rate')
    pyplot.title('Receiver operating characteristic:' + dataset + ' set')
    pyplot.legend(loc="lower right")
    pyplot.savefig(os.path.join(output_dir,'roc-' + dataset + '.pdf'))
    pyplot.close(fig)
    return roc_auc

def plot_scatter(x, y, output_dir, plotname, xlabel, ylabel):
    fig=pyplot.figure()
    pyplot.plot(x, y, 'ro')
    pyplot.xlabel(xlabel)
    pyplot.ylabel(ylabel)
    pyplot.title(plotname)
    pyplot.savefig(os.path.join(output_dir, plotname + '.pdf'))
    pyplot.close(fig)


def plot_weights(array,
                 figsize=(20,2),
                 height_padding_factor=0.2,
                 length_padding=1.0,
                 subticks_frequency=1.0,
                 colors=viz_sequence.default_colors,
                 plot_funcs=viz_sequence.default_plot_funcs,
                 highlight={},title=None):
    fig = pyplot.figure(figsize=figsize)
    fig.suptitle(title)
    ax = fig.add_subplot(111)

    viz_sequence.plot_weights_given_ax(ax=ax, array=array,
        height_padding_factor=height_padding_factor,
        length_padding=length_padding,
        subticks_frequency=subticks_frequency,
        colors=colors,
        plot_funcs=plot_funcs,
        highlight=highlight)


    return fig




def plot_profile_predictions(original, predictions, output_path, plot_name, plot_title = None):
    column_names=['Precitions_'+str(i) for i in range(len(predictions))]
    column_names.extend(['Smoothed_profile_'+str(i) for i in range(len(original))])
    pandas.DataFrame(numpy.vstack((predictions, original)).transpose(),
                            columns=column_names).plot(subplots=True,
                                                                       title=plot_title,
                                                                       legend='upper right')
    pyplot.savefig(os.path.join(output_path, plot_name+ ".png"))
    pyplot.close()




def plot_fit_diagnostics(data,data_smooth,output_path,smoothing_function, plot_name='Residual_Tests'):
    if not os.path.isdir(output_path):
        os.makedirs(output_path)
    residuals = [y - y_s for y, y_s in zip(data_smooth, data)]

    xs = numpy.hstack([y for y in data])
    ys = numpy.hstack([y for y in data_smooth])
    res = numpy.hstack([y for y in residuals])

    idx = numpy.random.choice(range(len(xs)), size=int(len(xs) / 5000), replace=False)

    fig = plot_residual_tests(xdata=xs[idx], yopts=ys[idx], res=res[idx], fct_name=smoothing_function)[0]

    fig.savefig(os.path.join(output_path,plot_name+ '.png'), format='png')
    pyplot.close(fig)





def plot_confusion_matrix(cm, output_path,classes=None,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=pyplot.cm.Blues,  plot_name=None, write_cell_values=False):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, numpy.newaxis]
        title = title + ' normalised'
        print("Normalized confusion matrix")
        # add ticks to colour bar to enforce the range is from 0 to 1
        v = numpy.linspace(0, 1.0, 10, endpoint=True)
    else:
        print('Confusion matrix, without normalization')
        v=numpy.linspace(0, cm.shape[0], 10, endpoint=True)
    #print(cm)
    pyplot.figure()
    pyplot.imshow(cm, interpolation='nearest', cmap=cmap)
    pyplot.title(title)

    cbar = pyplot.colorbar(ticks = v, format='%.1f')
    cbar.set_ticks(v)
    if normalize:
        pyplot.clim([0, 1])

    if classes is None:
        classes = range(cm.shape[-1])
    tick_marks = numpy.arange(len(classes))
    pyplot.xticks(tick_marks, classes, rotation=45)
    pyplot.yticks(tick_marks, classes)
    if write_cell_values:
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            pyplot.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    pyplot.tight_layout()
    pyplot.ylabel('True label')
    pyplot.xlabel('Predicted label')

    pyplot.savefig(os.path.join(output_path, plot_name + '.png'), format='png')
    pyplot.close()

### Necessary functions from pyqt_fit

ResidualMeasures = namedtuple("ResidualMeasures", "scaled_res res_IX prob normq")

def residual_measures(res):
    """
    Compute quantities needed to evaluate the quality of the estimation, based solely
    on the residuals.

    :rtype: :py:class:`ResidualMeasures`
    :returns: the scaled residuals, their ordering, the theoretical quantile for each residuals,
        and the expected value for each quantile.
    """
    IX = argsort(res)
    scaled_res = res[IX] / std(res)

    prob = (arange(len(scaled_res)) + 0.5) / len(scaled_res)
    normq = sqrt(2) * erfinv(2 * prob - 1)

    return ResidualMeasures(scaled_res, IX, prob, normq)

_restestfields = "res_figure qqplot dist_residuals"
ResTestResult = namedtuple("ResTestResult", _restestfields)
Plot1dResult = namedtuple("Plot1dResult", "figure estimate data CIs " + _restestfields)




def plot_residual_tests(xdata, yopts, res, fct_name, xname="X", yname='Y', res_name="residuals",
                        sorted_yopts=None, scaled_res=None, normq=None, fig=None):
    """
    Plot, in a single figure, all four residuals evaluation plots: :py:func:`plot_residuals`,
    :py:func:`plot_dist_residuals`, :py:func:`scaled_location_plot` and :py:func:`qqplot`.

    :param ndarray xdata:        Explaining variables
    :param ndarray yopt:         Optimized explained variables
    :param str     fct_name:     Name of the fitted function
    :param str     xname:        Name of the explaining variables
    :param str     yname:        Name of the dependant variables
    :param str     res_name:     Name of the residuals
    :param ndarray sorted_yopts: ``yopt``, sorted to match the scaled residuals
    :param ndarray scaled_res:   Scaled residuals
    :param ndarray normq:        Estimated value of the quantiles for a normal distribution

    :type  fig: handle or None
    :param fig: Handle of the figure to put the plots in, or None to create a new figure

    :rtype: :py:class:`ResTestResult`
    :returns: The handles to all the plots
    """
    if fig is None:
        fig = figure()
    else:
        try:
            figure(fig)
        except TypeError:
            figure(fig.number)

    xdata = asarray(xdata)
    yopts = asarray(yopts)
    res = asarray(res)

    subplot(2, 2, 1)
# First subplot is the residuals


    if scaled_res is None or sorted_yopts is None or normq is None:
        scaled_res, res_IX, _, normq = residual_measures(res)
        sorted_yopts = yopts[res_IX]


# Q-Q plot
    qqp = qqplot(scaled_res, normq)

    subplot(2, 2, 2)
# Distribution of residuals
    drp = plot_dist_residuals(res)

    suptitle("Residual Test for {}".format(fct_name))

    return ResTestResult(fig,  qqp, drp)



def plot_dist_residuals(res):
    """
    Plot the distribution of the residuals.

    :returns: the handle toward the histogram and the plot of the fitted normal distribution
    """
    ph = hist(res, normed=True)
    xr = arange(res.min(), res.max(), (res.max() - res.min()) / 1024)
    yr = stats.norm(0, res.std()).pdf(xr)
    pn = plot(xr, yr, 'r--')
    xlabel('Residuals')
    ylabel('Frequency')
    title('Distributions of the residuals')
    return ph, pn



def qqplot(scaled_res, normq):
    """
    Draw a Q-Q Plot from the sorted, scaled residuals (i.e. residuals sorted
    and normalized by their standard deviation)

    :param ndarray scaled_res: Scaled residuals
    :param ndarray normq:      Expected value for each scaled residual, based on its quantile.

    :returns: handle to the data plot
    """
    qqp = []
    qqp += plot(normq, scaled_res, '+')
    qqp += plot(normq, normq, 'r--')
    xlabel('Theoretical quantiles')
    ylabel('Normalized residuals')
    title('Normal Q-Q plot')
    return qqp