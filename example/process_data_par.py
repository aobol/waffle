#!/usr/bin/env python
# -*- coding: utf-8 -*-

from waffle.processing import *
import sys
import argparse

import multiprocessing as mp

def main(args):

    chanList = [int(item) for item in args.chan.split(',')]

    runs = [int(item) for item in args.runs.split(',')]
    if(len(runs)==1):
        runList = [runs[0]]
    elif(len(runs)==2):
        runList = np.arange(runs[0],runs[1])
    elif(len(runs)>=3):
        runList = np.array(runs)

    for r in runList:
        call = "qsub process_data.sh -r %d -c %s" % (r,(','.join(str(c) for c in chanList)))

        if args.dryrun==True:
            print(call)
        else:
            os.system(call)
   

if __name__=="__main__":
	parser = argparse.ArgumentParser()

	parser.add_argument('-c', '--chan', help='delimited list of channels input', type=str)
	parser.add_argument('-r', '--runs', help='delimited list of run range input', type=str)
	parser.add_argument('-d', '--dryrun', help='don\'t actually qsub it, just tell us how you would do it', action="store_true")

	args = parser.parse_args()
    
	main(args)