#!/usr/bin/env python

# ***************************************************
# ********************* import **********************
# ***************************************************
import os
import sys
import logging
import numpy as np
import pandas as pd
import matplotlib as mpl
from scipy.optimize import curve_fit
mpl.use('Agg')
from matplotlib.ticker import AutoMinorLocator
import matplotlib.pyplot as plt
# ***************************************************



# ***************************************************
# ******************** arguments ********************
# ***************************************************
# path
path = '../'
path2data = path + 'data/'

# logger
logger = logging.getLogger('Track plots')
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('\n%(name)16s | %(asctime)s | %(message)s\n', datefmt='%d-%m-%y %H:%M:%S'))
logger.addHandler(handler)

# line width
lw1 = 1.0
lw2 = 0.8
lw3 = 0.6

# font size
fs1 = 12

# grays
gray1 = '0.40'
gray2 = '0.70'
gray3 = '0.10'

# plot parameters
plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
plt.rc('text', usetex = True)
plt.rcParams['axes.linewidth']    = lw2
plt.rcParams['axes.edgecolor']    = '0.0'
plt.rcParams['xtick.top']         = True
plt.rcParams['ytick.right']       = True
plt.rcParams['xtick.direction']   = 'in'
plt.rcParams['ytick.direction']   = 'in'
plt.rcParams['xtick.major.size']  = 3.5
plt.rcParams['ytick.major.size']  = 3.5
plt.rcParams['xtick.major.width'] = 0.9
plt.rcParams['ytick.major.width'] = 0.9
plt.rcParams['xtick.minor.size']  = 2.2
plt.rcParams['ytick.minor.size']  = 2.2
plt.rcParams['xtick.minor.width'] = 0.6
plt.rcParams['ytick.minor.width'] = 0.6
plt.rcParams['xtick.labelsize']   = fs1
plt.rcParams['ytick.labelsize']   = fs1
plt.rcParams['errorbar.capsize']  = 3
# ***************************************************



# ***************************************************
# ******************** functions ********************
# ***************************************************
def createDir(path, folder):
  path2out = path + folder + '/'
  bashCmd = 'mkdir -p ' + path2out
  os.system(bashCmd)
  return path2out


def poly2(x, a, b, c):
  y = a * x**2 + b * x + c
  return y


def rSquared(ydata, xdata, popt, f):
  res = residuals(ydata, xdata, popt, f)
  ssTot = np.sum((ydata - np.mean(ydata))**2)
  R2 = 1 - (res / ssTot)
  return R2


def rSquaredAdj(R2, n, p):
  R2adj = 1 - (1 - R2) * (n - p) / (n - p - 1)
  return R2adj


def standardError(ydata, xdata, popt, f):
  res = residuals(ydata, xdata, popt, f)
  N = len(xdata)
  SE = np.sqrt(res / N)
  return SE


def residuals(ydata, xdata, popt, f):
  res = ydata - f(xdata, *popt)
  res = np.sum(res**2)
  return res


def plotData(path, plotName, raw):

  # data
  s_TER  = raw['s-TER']
  s_AER  = raw['s-AER']
  SR_TER = raw['SR-TER']
  SR_AER = raw['SR-AER']
  DF_TER = raw['DF-TER']
  DF_AER = raw['DF-AER']

  s_all  = np.array([*s_TER, *s_AER])
  SR_all = np.array([*SR_TER, *SR_AER])
  DF_all = np.array([*DF_TER, *DF_AER])

  s_range = np.arange(0, 10, 0.1)
  SR_coeffs = [0.026, -0.111, 1.398]
  DF_coeffs = [0.4, -6.1, 50]
  SR_est = poly2(s_range, *SR_coeffs)
  DF_est = poly2(s_range, *DF_coeffs)

  # curve fit
  s_range_VOL = np.arange(min([min(s_AER), min(s_TER)]) - 0.1, max([max(s_AER), max(s_TER)]) + 0.2, 0.1)
  popt_SR_all, pcov_SR_all = curve_fit(poly2, s_all, SR_all)
  popt_SR_TER, pcov_SR_TER = curve_fit(poly2, s_TER, SR_TER)
  popt_SR_AER, pcov_SR_AER = curve_fit(poly2, s_AER, SR_AER)
  popt_DF_all, pcov_DF_all = curve_fit(poly2, s_all, DF_all)
  popt_DF_TER, pcov_DF_TER = curve_fit(poly2, s_TER, DF_TER)
  popt_DF_AER, pcov_DF_AER = curve_fit(poly2, s_AER, DF_AER)
  print(' SF_all: ', *popt_SR_all, " ", rSquared(SR_all, s_all, popt_SR_all, poly2), rSquaredAdj(rSquared(SR_all, s_all, popt_SR_all, poly2), len(s_all), 1), standardError(SR_all, s_all, popt_SR_all, poly2))
  print(' SF_TER: ', *popt_SR_TER, " ", rSquared(SR_TER, s_TER, popt_SR_TER, poly2), rSquaredAdj(rSquared(SR_TER, s_TER, popt_SR_TER, poly2), len(s_TER), 1), standardError(SR_TER, s_TER, popt_SR_TER, poly2))
  print(' SF_AER: ', *popt_SR_AER, " ", rSquared(SR_AER, s_AER, popt_SR_AER, poly2), rSquaredAdj(rSquared(SR_AER, s_AER, popt_SR_AER, poly2), len(s_AER), 1), standardError(SR_AER, s_AER, popt_SR_AER, poly2))
  print(' DF_all: ', *popt_DF_all, " ", rSquared(DF_all, s_all, popt_DF_all, poly2), rSquaredAdj(rSquared(DF_all, s_all, popt_DF_all, poly2), len(s_all), 1), standardError(DF_all, s_all, popt_DF_all, poly2))
  print(' DF_TER: ', *popt_DF_TER, " ", rSquared(DF_TER, s_TER, popt_DF_TER, poly2), rSquaredAdj(rSquared(DF_TER, s_TER, popt_DF_TER, poly2), len(s_TER), 1), standardError(DF_TER, s_TER, popt_DF_TER, poly2))
  print(' DF_AER: ', *popt_DF_AER, " ", rSquared(DF_AER, s_AER, popt_DF_AER, poly2), rSquaredAdj(rSquared(DF_AER, s_AER, popt_DF_AER, poly2), len(s_AER), 1), standardError(DF_AER, s_AER, popt_DF_AER, poly2))

  #define the figure
  fig, (ax1, ax2) = plt.subplots(figsize = (2, 1), nrows = 2, ncols = 1)

  # title
  text = 'a) SF'
  ax1.text(-0.17, 1.04, '%s' % (text), ha = 'left', fontsize = fs1, transform = ax1.transAxes)

  # ax1 plot
  ax1.plot(s_range, SR_est, '-', label = 'theo', lw = lw3, c = 'black')
  ax1.plot(s_range_VOL, poly2(s_range_VOL, *popt_SR_TER), '--', label = '', lw = lw1, c = gray1)
  ax1.plot(s_range_VOL, poly2(s_range_VOL, *popt_SR_AER), '--', label = '', lw = lw1, c = gray2)
  ax1.plot(s_TER, SR_TER, 'o', label = 'TER', ms = 4, mec = gray3, mfc = gray1, mew = lw3)
  ax1.plot(s_AER, SR_AER, 'o', label = 'AER', ms = 4, mec = gray1, mfc = gray2, mew = lw3)

  # title
  text = 'b) DF'
  ax2.text(-0.17, 1.04, r'%s' % (text), ha = 'left', fontsize = fs1, transform = ax2.transAxes)

  # ax2 plot
  ax2.plot(s_range, DF_est, '-', label = 'theo', lw = lw3, c = 'black')
  ax2.plot(s_range_VOL, poly2(s_range_VOL, *popt_DF_TER), '--', label = '', lw = lw1, c = gray1)
  ax2.plot(s_range_VOL, poly2(s_range_VOL, *popt_DF_AER), '--', label = '', lw = lw1, c = gray2)
  ax2.plot(s_TER, DF_TER, 'o', label = '', ms = 4, mec = gray3, mfc = gray1, mew = lw3)
  ax2.plot(s_AER, DF_AER, 'o', label = '', ms = 4, mec = gray1, mfc = gray2, mew = lw3)

  # x and y ticks
  ax1.set_xticks(np.arange(0, 10, 1))
  ax2.set_xticks(np.arange(0, 10, 1))
  ax1.set_yticks(np.arange(1.0,  2.0, 0.2))
  ax2.set_yticks(np.arange(20, 45, 5))
  ax1.xaxis.set_ticklabels([])

  # y and y minor ticks
  ax1.xaxis.set_minor_locator(AutoMinorLocator(2))
  ax2.xaxis.set_minor_locator(AutoMinorLocator(2))
  ax1.yaxis.set_minor_locator(AutoMinorLocator(2))
  ax2.yaxis.set_minor_locator(AutoMinorLocator(2))

  # x and y limits
  ax1.set_xlim( 2.90,  5.55)
  ax2.set_xlim( 2.90,  5.55)
  ax1.set_ylim( 1.17,  1.67)
  ax2.set_ylim(23,    42)

  # x and y lables
  ax1.set_ylabel(r'$\textrm{SF}$ (Hz)')
  ax2.set_ylabel(r'$\textrm{DF}$ (\%)')
  ax2.set_xlabel(r'running speed (m/s)')
  
  # align labels
  fig.align_labels()

  # legend
  lgd_theo = mpl.lines.Line2D([], [], ls = '-', lw = lw3, color = 'black', label = 'theo')
  lgd_TER  = mpl.lines.Line2D([], [], ls = '--', lw = lw1, color = gray1, marker = 'o', ms = 4, mec = gray3, mfc = gray1, mew = lw3, label = 'TER')
  lgd_AER  = mpl.lines.Line2D([], [], ls = '--', lw = lw1, color = gray2, marker = 'o', ms = 4, mec = gray1, mfc = gray2, mew = lw3, label = 'AER')
  lgd = ax1.legend(handles = [lgd_TER, lgd_AER, lgd_theo], bbox_to_anchor = (0.0, 1.20, 1., 0), loc = 9, ncol = 3, handlelength = 2.05, handletextpad = 0.2, labelspacing = 1.0, borderpad = 0.0, mode = "expand", borderaxespad = 0, prop = {'size': fs1})
  lgd.get_frame().set_lw(0)

  # set figure
  plt.subplots_adjust(left = 0.15, right = 0.98, top = 0.92, bottom = 0.08, wspace = 0.0, hspace = 0.20)
  fig.set_size_inches(cm2inch(8.0), cm2inch(13.0))

  # save the figure
  fle = path + plotName
  fig.savefig(fle, dpi = 400)

  # close
  plt.close()


def cm2inch(value):
  return value / 2.54
# ***************************************************



# ***************************************************
# ********************* main ************************
# ***************************************************
try:
  logger.info('Starting plotting...')

  # create plot directory
  path2plot = createDir(path, 'plot')

  # read csv file
  raw = pd.read_csv(path2data + 'data-track.csv')

  # plot
  plotName = 'track.pdf' 
  plotData(path2plot, plotName, raw)

  logger.info('Plotting done.')
except Exception as err:
  logger.error('%s: Exit: a problem in the main script happened (%s)' % (type(err).__name__, str(err)))
  sys.exit(1)
# ***************************************************
