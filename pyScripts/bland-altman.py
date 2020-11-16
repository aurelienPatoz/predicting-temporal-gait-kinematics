#!/usr/bin/env python

# ***************************************************
# ********************* import **********************
# ***************************************************
import os
import sys
import logging
import argparse
import numpy as np
import pandas as pd
from scipy.odr import *
from openpyxl import load_workbook
import matplotlib as mpl
mpl.use('Agg')
from matplotlib.ticker import AutoMinorLocator
import matplotlib.pyplot as plt
# ***************************************************



# ***************************************************
# ******************** arguments ********************
# ***************************************************
# path
path = '../'
path2data = path + 'data-bland-altman/'

# logger
logger = logging.getLogger('Bland-altman plots')
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('\n%(name)16s | %(asctime)s | %(message)s\n', datefmt='%d-%m-%y %H:%M:%S'))
logger.addHandler(handler)

# line width
lw1 = 1.0
lw2 = 0.8

# font size
fs1 = 12

# grays
gray1 = '0.45'
gray2 = '0.75'
gray3 = '0.90'

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


def differences(df, col1, col2):
  return (df[col1] - df[col2]).to_numpy()


def means(df, col1, col2):
  return 0.5 * (df[col1] + df[col2]).to_numpy()


def dataBlandAltman(df, what):
  tmp = df[df['what'] == what]
  bias       = tmp['bias'].to_numpy()[0]
  biasUpr    = tmp['bias_upr'].to_numpy()[0]
  biasLwr    = tmp['bias_lwr'].to_numpy()[0]
  LOAupr     = tmp['LOA-upr'].to_numpy()[0]
  LOAupr_upr = tmp['LOA-upr_upr'].to_numpy()[0]
  LOAupr_lwr = tmp['LOA-upr_lwr'].to_numpy()[0]
  LOAlwr     = tmp['LOA-lwr'].to_numpy()[0]
  LOAlwr_upr = tmp['LOA-lwr_upr'].to_numpy()[0]
  LOAlwr_lwr = tmp['LOA-lwr_lwr'].to_numpy()[0]
  slope      = tmp['slope'].to_numpy()[0]
  ntrcpt     = tmp['y-ntrcpt'].to_numpy()[0]
  R2         = tmp['R2'].to_numpy()[0]
  return bias, biasUpr, biasLwr, LOAupr, LOAupr_upr, LOAupr_lwr, LOAlwr, LOAlwr_upr, LOAlwr_lwr, slope, ntrcpt, R2


def fitAndConfInt(path, what):
  df = pd.read_csv(path+what+'.csv')
  fit = df['fit'].to_numpy()
  lwr = df['lwr'].to_numpy()
  upr = df['upr'].to_numpy()
  return fit, lwr, upr


def plotData(path, plotName, mean1, diff1, fit1, lwr1, upr1, bias1, biasUpr1, biasLwr1, LOAupr1, LOAupr_upr1, LOAupr_lwr1, LOAlwr1, LOAlwr_upr1, LOAlwr_lwr1, slope1, ntrcpt1, Rsqr1, mean2, diff2, fit2, lwr2, upr2, bias2, biasUpr2, biasLwr2, LOAupr2, LOAupr_upr2, LOAupr_lwr2, LOAlwr2, LOAlwr_upr2, LOAlwr_lwr2, slope2, ntrcpt2, Rsqr2):

  # extended x data 
  newMean1 = np.arange(min(mean1) - 0.025, max(mean1) + 0.026, 0.005)
  newMean2 = np.arange(min(mean2) - 1.50, max(mean2) + 1.50, 0.25)

  # min and max for x axis
  prctgX = 0.025
  xMin1 = newMean1[0]  - prctgX * abs(newMean1[0] - newMean1[-1])
  xMax1 = newMean1[-1] + prctgX * abs(newMean1[0] - newMean1[-1])
  xMin2 = newMean2[0]  - prctgX * abs(newMean2[0] - newMean2[-1])
  xMax2 = newMean2[-1] + prctgX * abs(newMean2[0] - newMean2[-1])

  # min and max for y axis
  prctgY = 0.075
  yMin1 = LOAlwr_lwr1 - prctgY * abs(LOAupr_upr1 - LOAlwr_lwr1)
  yMax1 = LOAupr_upr1 + prctgY * abs(LOAupr_upr1 - LOAlwr_lwr1)
  yMin2 = LOAlwr_lwr2 - prctgY * abs(LOAupr_upr2 - LOAlwr_lwr2)
  yMax2 = LOAupr_upr2 + prctgY * abs(LOAupr_upr2 - LOAlwr_lwr2)

  #define the figure
  fig, (ax1, ax2) = plt.subplots(figsize = (2, 1), nrows = 2, ncols = 1)

  # title
  text = 'a) SF'
  ax1.text(-0.28, 1.16, '%s' % (text), ha = 'left', transform = ax1.transAxes)

  # ax1 plot
  ax1.axhline(y = 0,           ls = '-',  color = gray1, lw = lw1)
  ax1.axhline(y = biasLwr1,    ls = ':',  color = gray2, lw = lw2)
  ax1.axhline(y = bias1,       ls = '--', color = gray2, lw = lw2)
  ax1.axhline(y = biasUpr1,    ls = ':',  color = gray2, lw = lw2)
  ax1.axhline(y = LOAupr_lwr1, ls = ':',  color = gray2, lw = lw2)
  ax1.axhline(y = LOAupr1,     ls = '--', color = gray2, lw = lw2)
  ax1.axhline(y = LOAupr_upr1, ls = ':',  color = gray2, lw = lw2)
  ax1.axhline(y = LOAlwr_lwr1, ls = ':',  color = gray2, lw = lw2)
  ax1.axhline(y = LOAlwr1,     ls = '--', color = gray2, lw = lw2)
  ax1.axhline(y = LOAlwr_upr1, ls = ':',  color = gray2, lw = lw2)

  ax1.fill_between([xMin1, xMax1], [biasLwr1],    [biasUpr1],    color = gray2, alpha = '0.4')
  ax1.fill_between([xMin1, xMax1], [LOAlwr_lwr1], [LOAlwr_upr1], color = gray3, alpha = '0.3')
  ax1.fill_between([xMin1, xMax1], [LOAupr_lwr1], [LOAupr_upr1], color = gray3, alpha = '0.3')

  ax1.plot(mean1, diff1, 'o', label = '', ms = 5, mec = 'black', mfc = 'white', mew = lw2)

  ax1.plot(newMean1, lwr1, '--', color = 'black', lw = lw1)
  ax1.plot(newMean1, fit1, '-',  color = 'black', lw = lw1)
  ax1.plot(newMean1, upr1, '--', color = 'black', lw = lw1)

  sign1 = '-' if ntrcpt1 < 0 else '+' 
  ax1.text(1, 1.05, r'$\Delta \textrm{SF} = %4.2f \, \overline{\textrm{SF}} %1s %4.2f\ \ \ \ \ \ \ \ \ \ R^2 = %4.2f$' %(slope1, sign1, abs(ntrcpt1), Rsqr1), ha = 'right', transform = ax1.transAxes)

  # ax2 plot
  ax2.axhline(y = 0,           ls = '-',  color = gray1, lw = lw1)
  ax2.axhline(y = biasLwr2,    ls = ':',  color = gray2, lw = lw2)
  ax2.axhline(y = bias2,       ls = '--', color = gray2, lw = lw2)
  ax2.axhline(y = biasUpr2,    ls = ':',  color = gray2, lw = lw2)
  ax2.axhline(y = LOAupr_lwr2, ls = ':',  color = gray2, lw = lw2)
  ax2.axhline(y = LOAupr2,     ls = '--', color = gray2, lw = lw2)
  ax2.axhline(y = LOAupr_upr2, ls = ':',  color = gray2, lw = lw2)
  ax2.axhline(y = LOAlwr_lwr2, ls = ':',  color = gray2, lw = lw2)
  ax2.axhline(y = LOAlwr2,     ls = '--', color = gray2, lw = lw2)
  ax2.axhline(y = LOAlwr_upr2, ls = ':',  color = gray2, lw = lw2)

  ax2.fill_between([xMin2, xMax2], [biasLwr2],    [biasUpr2],    color = gray2, alpha = '0.4')
  ax2.fill_between([xMin2, xMax2], [LOAlwr_lwr2], [LOAlwr_upr2], color = gray3, alpha = '0.3')
  ax2.fill_between([xMin2, xMax2], [LOAupr_lwr2], [LOAupr_upr2], color = gray3, alpha = '0.3')

  ax2.plot(mean2, diff2, 'o', label = '', ms = 5, mec = 'black', mfc = 'white', mew = lw2)

  ax2.plot(newMean2, lwr2, '--', color = 'black', lw = lw1)
  ax2.plot(newMean2, fit2, '-',  color = 'black', lw = lw1)
  ax2.plot(newMean2, upr2, '--', color = 'black', lw = lw1)

  text = 'b) DF'
  ax2.text(-0.28, 1.16, r'%s' % (text), ha = 'left', transform = ax2.transAxes)

  sign2 = '-' if ntrcpt2 < 0 else '+' 
  ax2.text(1, 1.05, r'$\Delta \textrm{DF} = %4.2f \, \overline{\textrm{DF}} %1s %4.2f\ \ \ \ \ \ \ \ R^2 = %4.2f$' %(slope2, sign2, abs(ntrcpt2), Rsqr2), ha = 'right', transform = ax2.transAxes)

  # x and y ticks
  #ax1.set_xticks(np.arange(0, 10, 0.5))
  #ax2.set_xticks(np.arange(0, 500, 50))
  ax1.set_yticks(np.arange(- 0.3,  0.45, 0.15))
  ax2.set_yticks(np.arange(-10,   15,    5))

  # y and y minor ticks
  ax1.xaxis.set_minor_locator(AutoMinorLocator(2))
  ax2.xaxis.set_minor_locator(AutoMinorLocator(2))
  ax1.yaxis.set_minor_locator(AutoMinorLocator(2))
  ax2.yaxis.set_minor_locator(AutoMinorLocator(2))

  # x and y limits
  ax1.set_xlim(xMin1, xMax1)
  ax2.set_xlim(xMin2, xMax2)
  ax1.set_ylim(yMin1, yMax1)
  ax2.set_ylim(yMin2, yMax2)

  # x and y lables
  ax1.set_ylabel(r'$\Delta \textrm{SF}$ (Hz)')
  ax1.set_xlabel(r'$\overline{\textrm{SF}}$ (Hz)')
  ax2.set_ylabel(r'$\Delta \textrm{DF}$ (\%)')
  ax2.set_xlabel(r'$\overline{\textrm{DF}}$ (\%)')
  
  # align labels
  fig.align_labels()

  plt.subplots_adjust(left = 0.22, right = 0.98, top = 0.92, bottom = 0.08, wspace = 0.0, hspace = 0.55)
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

  path2plot = createDir(path, 'plot')

  raw = pd.read_csv(path2data + 'raw-data.csv')
  ba  = pd.read_csv(path2data + 'bland-altman.csv')

  what1 = 'SR'
  what2 = 'DF'
  ###### data for what1 ######
  # columns 1 and 2 for what1
  col11, col12 = what1, what1 + 'theo'
  # differences and means
  diff1 = differences(raw, col11, col12)
  mean1 = means(      raw, col11, col12)
  # fit and confidence intervals of the fit
  fit1, lwr1, upr1 = fitAndConfInt(path2data, what1)
  # bias, LOA, ...
  bias1, biasUpr1, biasLwr1, LOAupr1, LOAupr_upr1, LOAupr_lwr1, LOAlwr1, LOAlwr_upr1, LOAlwr_lwr1, slope1, ntrcpt1, Rsqr1 = dataBlandAltman(ba, what1)
  ###### data for what2 ######
  # columns 1 and 2 for what2
  col21, col22 = what2, what2 + 'theo'
  # differences and means
  diff2 = differences(raw, col21, col22)
  mean2 = means(      raw, col21, col22)
  # fit and confidence intervals of the fit
  fit2, lwr2, upr2 = fitAndConfInt(path2data, what2)
  # bias, LOA, ...
  bias2, biasUpr2, biasLwr2, LOAupr2, LOAupr_upr2, LOAupr_lwr2, LOAlwr2, LOAlwr_upr2, LOAlwr_lwr2, slope2, ntrcpt2, Rsqr2 = dataBlandAltman(ba, what2)
  # plot
  plotName = 'bland-altman.pdf' 
  plotData(path2plot, plotName, mean1, diff1, fit1, lwr1, upr1, bias1, biasUpr1, biasLwr1, LOAupr1, LOAupr_upr1, LOAupr_lwr1, LOAlwr1, LOAlwr_upr1, LOAlwr_lwr1, slope1, ntrcpt1, Rsqr1, mean2, diff2, fit2, lwr2, upr2, bias2, biasUpr2, biasLwr2, LOAupr2, LOAupr_upr2, LOAupr_lwr2, LOAlwr2, LOAlwr_upr2, LOAlwr_lwr2, slope2, ntrcpt2, Rsqr2)

  logger.info('Plotting done.')
except Exception as err:
  logger.error('%s: Exit: a problem in the main script happened (%s)' % (type(err).__name__, str(err)))
  sys.exit(1)
# ***************************************************
