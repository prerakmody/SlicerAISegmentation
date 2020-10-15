#!/usr/bin/env python-real

import os
import sys
from sys import argv
import time
import nrrd
import numpy as np
from pathlib import Path

import src.util as myUtil

def main(params):

    # Step 0 - Parse Params
    print (' - [processImgViaCLI()] params: ',params)
    inputVolumeNodeTmpPath = params[1]
    outputVolumeTmpPath = params[3]
    inputVolumeNodename = params[5]
    modelFolderPath = str(params[7])
    predThreshold = float(params[9])

    # Step 1 - Get image data and init model
    inputVolArray, inputVolHeader = nrrd.read(inputVolumeNodeTmpPath)
    print (' - [processImgViaCLI()] inputVolArray: ', inputVolArray.shape)

    # Step 2 - Predict
    if 1:
        sliceTemporaryPath = Path(inputVolumeNodeTmpPath).parent.absolute()
        tmpPath = Path(sliceTemporaryPath).joinpath(inputVolumeNodename)
        myUtil.progressBarInit()
        modelGridParams = myUtil.predict(inputVolArray, None, modelFolderPath, tmpPath)
        # modelGridParams = myUtil.predictTemp(inputVolArray, None, modelFolderPath, tmpPath)
        print (' - [processImgViaCLI()] modelGridParams: ', modelGridParams)
        
        # Step 3 - Combine predictions
        yPredictProb = myUtil.joinGrids(inputVolArray.shape, modelGridParams, tmpPath)
        yPredictProb[yPredictProb < predThreshold] = 0
        yPredict = np.argmax(yPredictProb, axis=3)
        yPredict = yPredict.astype(np.int8)
        myUtil.progressBarUpdate(0.5)
        myUtil.progressBarClose(0.5)
        
        # Step 4 - Write output
        # nrrd.write(str(outputVolumeTmpPath), yPredict, inputVolHeader)
        outputVolumeTmpPath = Path(tmpPath).joinpath('result.nrrd')
        print (' - writing to :', outputVolumeTmpPath)
        nrrd.write(str(outputVolumeTmpPath), yPredict, inputVolHeader)
        
        return yPredict

    

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # for i, each in enumerate(sys.argv):print (' - sys.argv[{}]: ', i, sys.argv[i])
        main(sys.argv[1:])
    else:
        print (' - [ERROR][AISegmenationCLI.py] len(sys.argv): ', len(sys.argv))

"""
for i, each in enumerate(sys.argv):print (' - sys.argv[{}]: ', i, sys.argv[i])
        
"""