#!/usr/bin/env python
# -*- coding: utf-8 -*-

from waffle.processing import *
import sys
import argparse

import multiprocessing as mp

def main(runList,chanList):

    #print(runList)
    #print(chanList)
    #sys.exit()

    #data processing
    proc = DataProcessor()

    proc.tier0(runList, min_signal_thresh=100, chanList=chanList)
	# proc.load_nonlinearities_to_db(runList)
    proc.tier1(runList, num_threads=mp.cpu_count())

    #calibration, simple analysis
    #proc.calibrate(runList)
    #proc.ae_cut(runList, )
    #proc.baseline_cuts(runList, )


if __name__=="__main__":
	parser = argparse.ArgumentParser()

	parser.add_argument('-c', '--chan', help='delimited list of channels input', type=str)
	parser.add_argument('-r', '--run', help='run number', type=int)

	args = parser.parse_args()

	chanList = [int(item) for item in args.chan.split(',')]

	runList = [args.run]

	main(runList,chanList)
