# -*- coding: utf-8 -*-
from ambulanceLocationInstance import AmbulanceLocationInstance
import logging
import os

additional_cuts = ['dominated_edges','partitions']

logging.basicConfig(level=logging.INFO)
files = ['Ambulance instances/Region02.txt']
logger = logging.getLogger(__name__)

calculateBinary = True

def makelen(string,length):
    string += ' '*max(0,length-len(string))
    return string[:length]
    
for file in files:
    logger.info('File: ' + file)
    name = os.path.basename(os.path.splitext(file)[0])

    objectives={}
    nodes={}
    runtimes = {}
    
    binaryJobs = {'relaxed' : False}
    if calculateBinary:
        binaryJobs['binary'] =  True
        
        
    for binary_label,binary in binaryJobs.items():
        for formulation_label,additional_cuts in {'simple' : [],'extended' : additional_cuts}.items():
            alp = AmbulanceLocationInstance.solveFromFile(file,name=name,binary=binary,additionalCuts=additional_cuts)
            label = formulation_label+'_'+binary_label
            objectives[label] = alp.objective
            runtimes[label] = alp.duration
            nodes[label] = alp.nodes
            
    
    
    for name,data in {'OBJECTIVE' : objectives,'DURATION' : runtimes,'NODES' : nodes}.items():
        formatStr = '{0:.3E}'
        
        logger.info(makelen(name,22)+'Binary      Relaxed')
        logger.info('simple formulation:   ' + formatStr.format(data['simple_binary']) + '   ' +formatStr.format(data['simple_relaxed']))
        logger.info('extended formulation: ' + formatStr.format(data['extended_binary']) + '   ' +formatStr.format(data['extended_relaxed']))
        logger.info('')
        
    
    
    
    #problem.solveFromFile(fi)
    