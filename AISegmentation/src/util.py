import qt
import sys
import time
import slicer
import traceback
import numpy as np
from pathlib import Path

# from AISegmentation.src.config import myConfig

############################################################
#                     3D SLICER RELATED                    #
############################################################
def getLabelMapVolumeNodeFromVolumeNode(volNode):

  spacing = volNode.GetSpacing()
  origin = volNode.GetOrigin()
  directions = np.zeros((3,3), dtype=np.double)
  name = volNode.GetName()

  volNode.GetIJKToRASDirections(directions)
  node = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLLabelMapVolumeNode', name)
  node.SetSpacing(spacing[0], spacing[1], spacing[2])
  node.SetOrigin(origin[0], origin[1], origin[2])
  node.SetIJKToRASDirections(directions)
  slicer.mrmlScene.AddNode(node)
  return node

def convertLabelMapVolumeToSegmentationMap(labelMapVolumeNode):

  name = labelMapVolumeNode.GetName() + '_segmentation'
  segmentationNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLSegmentationNode', name)
  slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(labelMapVolumeNode, segmentationNode)
  segmentationNode.GetSegmentation().SetConversionParameter(
        slicer.vtkBinaryLabelmapToClosedSurfaceConversionRule.GetSmoothingFactorParameterName(),"0.0")
  segmentationNode.CreateClosedSurfaceRepresentation()
  slicer.mrmlScene.RemoveNode(labelMapVolumeNode)
  return segmentationNode

def updateLabelMapVolumeNode(labelMapVolumeNode, array_numpy):
  slicer.util.updateVolumeFromArray(labelMapVolumeNode, array_numpy)

def resetAxesOrderFromRawFile(array_numpy):
  return np.moveaxis(array_numpy, [2,1,0], [0,1,2])
############################################################
#                        QT RELATED                        #
############################################################

class DoubleProgressBar(qt.QDialog):

  def __init__(self):
    super().__init__()
    self.initUI()
  
  def initUI(self):
    self.setWindowTitle('AI Countouring Progress Bar')
    self.maxProgressBar1 = 100
    self.maxProgressBar2 = 100
    self.runStatus = False

    self.progressBar1 = qt.QProgressBar(self)
    self.progressBar1.setGeometry(5, 5, 300, 25)
    self.progressBar1.setMaximum(self.maxProgressBar1)
    self.progressBar1.setValue(0)
    self.progressBar1.setFormat('Prediction: {}%'.format(0))

    self.progressBar2 = qt.QProgressBar(self)
    self.progressBar2.setGeometry(5, 35, 300, 25)
    self.progressBar2.setMaximum(self.maxProgressBar2)
    self.progressBar2.setValue(0)
    self.progressBar2.setFormat('Post-Processing: {}%'.format(0))
    slicer.app.processEvents()

    self.show()
  
  def updateProgressBar1(self, value):
    self.runStatus = True
    if value > self.progressBar1.value:   
      self.progressBar1.setValue(value)
      self.progressBar1.setFormat('Prediction: {}%'.format(value))
    slicer.app.processEvents()
  
  def updateProgressBar2(self, value):
    self.runStatus = True
    self.progressBar1.setValue(100)
    self.progressBar1.setFormat('Prediction: {}%'.format(100))

    if value > self.progressBar2.value:
      self.progressBar2.setValue(value)
      self.progressBar2.setFormat('Post-Processing: {}%'.format(value))
    slicer.app.processEvents()

def createProgressDialog(parent=None, value=0, maximum=100, labelText="", windowTitle="Processing...", autoClose=False,**kwargs):
    """Display a modal QProgressDialog.
    Go to `QProgressDialog documentation <http://pyqt.sourceforge.net/Docs/PyQt4/qprogressdialog.html>`_ to
    learn about the available keyword arguments.
    Examples::
      # Prevent progress dialog from automatically closing
      progressbar = createProgressIndicator(autoClose=False)
      # Update progress value
      progressbar.value = 50
      # Update label text
      progressbar.labelText = "processing XYZ"
    """
    import qt # qt.qVersion()
    progressIndicator = qt.QProgressDialog()  #(parent if parent else self.mainWindow())
    progressIndicator.minimumDuration = 0
    progressIndicator.maximum = maximum
    progressIndicator.value = value
    progressIndicator.windowTitle = windowTitle
    progressIndicator.labelText = labelText
    
    # progressIndicator.setWindowFlag(qt.WindowCloseButtonHint, False)

    for key, value in kwargs.items():
      if hasattr(progressIndicator, key):
        setattr(progressIndicator, key, value)
      else:
        print (' - [ERROR][createProgressDialog()] key: ', key)

    return progressIndicator

def getGPUInfo(additional_text=''):
    import py3nvml.py3nvml as py3nvml
    py3nvml.nvmlInit()
    deviceCount = py3nvml.nvmlDeviceGetCount()
    if deviceCount > 0:
      handle = py3nvml.nvmlDeviceGetHandleByIndex(0)
      name = py3nvml.nvmlDeviceGetName(handle)
      info = py3nvml.nvmlDeviceGetMemoryInfo(handle)
      mb_used = info.used >> 20 
      mb_total = info.total >> 20

      status_str = '{} : {} / {} MB'.format(name, mb_used, mb_total)
      if len(additional_text):
        status_str = status_str + ' || ' + additional_text
      slicer.util.mainWindow().statusBar().showMessage(status_str)
    else:
      slicer.util.mainWindow().statusBar().showMessage('No GPU found on system')
    slicer.app.processEvents()

def getTFlowGPUStatus():
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    if len(gpus):
      tf.config.experimental.set_memory_growth(gpus[0], True)
      getGPUInfo()
      return 1
    else:
      return 0

class WorkerSignals(qt.QObject):
  finished = qt.Signal()
  error = qt.Signal(tuple)
  result = qt.Signal(object)
  progress = qt.Signal(int)

class Worker(qt.QRunnable):
  def __init__(self, fn, *args, **kwargs):
    super(Worker, self).__init__()

    # Store constructor arguments (re-used for processing)
    self.fn = fn
    self.args = args
    self.kwargs = kwargs
    self.signals = WorkerSignals()    

    # Add the callback to our kwargs
    self.kwargs['progress_callback'] = self.signals.progress

  @qt.Slot()
  def run(self):
    # Retrieve args/kwargs here; and fire processing using them
    try:
        result = self.fn(*self.args, **self.kwargs)
    except:
        traceback.print_exc()
        exctype, value = sys.exc_info()[:2]
        self.signals.error.emit((exctype, value, traceback.format_exc()))
    else:
        self.signals.result.emit(result)  # Return the result of the processing
    finally:
        self.signals.finished.emit()    

def printOutput(s):
    print(s)

############################################################
#                   DEEP LEARNING RELATED                  #
############################################################

def loadModel(modelFolderPath):
    import tensorflow as tf
    import src.model as mymodel
    
    model = mymodel.ModelUNet3D(class_count=10)
    optimizer = tf.keras.optimizers.Adam()
    ckptObj = tf.train.Checkpoint(optimizer=optimizer, model=model)
    ckptObj.restore(save_path=tf.train.latest_checkpoint(str(modelFolderPath))).expect_partial()

    modelTrainingParams = {
      'classTotal' : 10
      ,'wGrid' : 96
      ,'hGrid' : 96
      ,'dGrid' : 96
      ,'wOverlap' : 20
      ,'hOverlap' : 20
      ,'dOverlap' : 20
    }
    return model, modelTrainingParams

def predictTemp(img_array, model, modelPath, tmpPath, progress_callback):
  modelTrainingParams = {
      'classTotal' : 10
      ,'wGrid' : 96
      ,'hGrid' : 96
      ,'dGrid' : 96
      ,'wOverlap' : 20
      ,'hOverlap' : 20
      ,'dOverlap' : 20
    }
  
  for _ in range(10):
    time.sleep(1)
  
  return modelTrainingParams

def predict(img_array, model, modelPath, tmpPath, progress_callback):

  # Step 1 - Get Model
  getGPUInfo()
  modelTrainingParams = {}
  if model is None:
    model, modelTrainingParams = loadModel(modelPath)
    tmpPath.mkdir(exist_ok=True, parents=True)
  getGPUInfo()

  # Step 2 - Get grid-based dataloader
  gridSize = (modelTrainingParams['wGrid'], modelTrainingParams['hGrid'], modelTrainingParams['dGrid'])
  gridOverlap = (modelTrainingParams['wOverlap'], modelTrainingParams['hOverlap'], modelTrainingParams['dOverlap'])
  dataloader = getDataLoaderGrid(img_array, gridSize=gridSize, gridOverlap=gridOverlap)

  # Step 3 - Get progress bar
  totalGrids = getTotalGrids(img_array.shape, gridSize, gridOverlap)
  progressDialogModel = createProgressDialog(parent=None, value=0, maximum=totalGrids, windowTitle='Predicting contours ... ')

  # Step 4 - Predict
  count = 1
  for X, meta in dataloader:
    y_predict = model(X)
    filePath = Path(tmpPath).joinpath(meta)
    np.save(filePath, y_predict.numpy())
    progressDialogModel.setValue(count)
    progressDialogModel.labelText = str(count) + '/' + str(totalGrids) + ' - ' + meta
    getGPUInfo(' || Predicting({}%)'.format(int(count/totalGrids)*100))
    slicer.app.processEvents()
    count += 1
  
  progressDialogModel.close()
  getGPUInfo()
  return modelTrainingParams

def joinGrids(volumeShape, modelGridParams, savePath):

  # Step 1 - Capture grid stats
  wTotal, hTotal, dTotal = volumeShape
  wGrid = modelGridParams['wGrid']
  hGrid = modelGridParams['hGrid']
  dGrid = modelGridParams['dGrid']
  wOverlap = modelGridParams['wOverlap']
  hOverlap = modelGridParams['hOverlap']
  dOverlap = modelGridParams['dOverlap']
  classTotal = modelGridParams['classTotal'] 
  wGridStarts = splitRange(wTotal, widthGrid=wGrid, widthOverlap=wOverlap, resType='boundary')
  wGridCount = len(wGridStarts)
  hGridStarts = splitRange(hTotal, widthGrid=hGrid, widthOverlap=hOverlap, resType='boundary')
  hGridCount = len(hGridStarts)
  dGridStarts = splitRange(dTotal, widthGrid=dGrid, widthOverlap=dOverlap, resType='boundary')
  dGridCount = len(dGridStarts)
  wOverlapLastGrid = abs(wGridStarts[-1][0] - wGridStarts[-2][1])
  hOverlapLastGrid = abs(hGridStarts[-1][0] - hGridStarts[-2][1])
  dOverlapLastGrid = abs(dGridStarts[-1][0] - dGridStarts[-2][1])
  
  # Step 2 - Create temp variables
  yPredictInit = np.zeros((wTotal + (wGridCount-2)*wOverlap + wOverlapLastGrid
                          , hTotal + (hGridCount-2)*hOverlap + hOverlapLastGrid
                          , dTotal + (dGridCount-2)*dOverlap + dOverlapLastGrid   
                          , classTotal
                          ), dtype=np.float32)
  yPredictW    = np.zeros((wTotal
                          , hTotal + (hGridCount-2)*hOverlap + hOverlapLastGrid
                          , dTotal + (dGridCount-2)*dOverlap + dOverlapLastGrid
                          , classTotal
                          ), dtype=np.float32)
  yPredictH    = np.zeros((wTotal, hTotal, dTotal + (dGridCount-2)*dOverlap + dOverlapLastGrid, classTotal), dtype=np.float32)
  yPredictD    = np.zeros((wTotal, hTotal, dTotal, classTotal), dtype=np.float32)
  
  # Step 3 - Extract w,h,d of grid starting points
  wStarts = []
  hStarts = []
  dStarts = []
  filePathRef = ''
  for filePath in savePath.iterdir():
    filePathRef = filePath
    tmp = Path(filePath).parts[-1].split('_')
    wStarts.append(int(tmp[-10]))
    hStarts.append(int(tmp[-6]))
    dStarts.append(int(tmp[-2]))
  wStarts = sorted(list(set(wStarts)))
  hStarts = sorted(list(set(hStarts)))
  dStarts = sorted(list(set(dStarts)))

  # Step 4 - Create mega prediction grid
  totalGrids = wGridCount*hGridCount*dGridCount
  progressDialogModel = createProgressDialog(parent=None, value=0, maximum=totalGrids, windowTitle='Stitching data ... ')
  count = 1
  for wId, wStart in enumerate(wStarts):
    for hId, hStart in enumerate(hStarts):
        for dId, dStart in enumerate(dStarts):
          progressDialogModel.setValue(count)
          progressDialogModel.labelText = str(count) + '/' + str(totalGrids)
          slicer.app.processEvents()
          count += 1
          filePathGridParts = list(filePathRef.parts)
          filePathGridPartsFilename = filePathGridParts[-1]
          filePathGridPartsFilename = filePathGridPartsFilename.split('_')
          filePathGridPartsFilename[-10] = str(wStart)
          filePathGridPartsFilename[-9] = str(wStart + wGrid)
          filePathGridPartsFilename[-6] = str(hStart)
          filePathGridPartsFilename[-5] = str(hStart + hGrid)
          filePathGridPartsFilename[-2] = str(dStart)
          filePathGridPartsFilename[-1] = str(dStart + dGrid) + '.npy'
          filePathGridPartsFilename = '_'.join(filePathGridPartsFilename)
          filePathGridParts[-1] = filePathGridPartsFilename
          filePathGrid = Path(*filePathGridParts)

          if Path(filePathGrid).exists():
            yPredGrid = np.load(filePathGrid)
            if wId != len(wStarts) - 1:
              wStartThis = wStart + wId*wOverlap
            else:
              wStartThis = wStart + (wId-1)*wOverlap + wOverlapLastGrid
            wEndThis = wStartThis + wGrid
            if hId != len(hStarts) - 1:
              hStartThis = hStart + hId*hOverlap
            else:
              hStartThis = hStart + (hId-1)*hOverlap + hOverlapLastGrid
            hEndThis = hStartThis + hGrid

            if dId != len(dStarts) - 1:
              dStartThis = dStart + dId*hOverlap
            else:
              dStartThis = dStart + (dId-1)*dOverlap + dOverlapLastGrid
            dEndThis = dStartThis + dGrid

            yPredictInit[wStartThis:wEndThis, hStartThis:hEndThis, dStartThis:dEndThis] = yPredGrid
            # filePathGrid.unlink()

          else:
            print (' - [ERROR] filePathGrid: ', filePathGrid)
  
  ############# Step 6.4 - Reshape yPredictInit --> yPredictW --> yPredictH --> yPredictD
  if 1:
    # Step 6.4.1 - fill width data
    wIdxsFillPutarray = []
    wIdxsFillPullarray = []
    lenPutPrev = 0
    lenPullprev = 0
    for wId, wStart in enumerate(wStarts):
      if wId == 0:
        wIdxsFillPutarray.extend(list(np.arange(0, wGrid-wOverlap)))
        wIdxsFillPullarray.extend(list(np.arange(0, wGrid-wOverlap)))
      elif wId !=0 and wId < len(wStarts) - 2:
        wIdxsFillPutarray.extend(list(np.arange(wStart+wOverlap, wStart+wOverlap + wGrid - 2*wOverlap)))
        wIdxsFillPullarray.extend(list(np.arange((wId)*wGrid + wOverlap, (wId+1)*wGrid - wOverlap)))
      elif wId == len(wStarts) - 2:
        wIdxsFillPutarray.extend(list(np.arange(wStart+wOverlap, wStart+wOverlap + wGrid - wOverlap - wOverlapLastGrid)))
        wIdxsFillPullarray.extend(list(np.arange((wId)*wGrid + wOverlap, (wId+1)*wGrid - wOverlapLastGrid)))
      elif wId == len(wStarts) - 1:
        wIdxsFillPutarray.extend(list(np.arange(wStart+wOverlapLastGrid, wStart+wOverlapLastGrid + wGrid-wOverlapLastGrid)))
        wIdxsFillPullarray.extend(list(np.arange((wId)*wGrid + wOverlapLastGrid, (wId+1)*wGrid)))
      # print (len(wIdxsFillPutarray), len(wIdxsFillPullarray), len(wIdxsFillPutarray) - lenPutPrev, len(wIdxsFillPullarray)-lenPullprev)
      lenPutPrev = len(wIdxsFillPutarray)
      lenPullprev = len(wIdxsFillPullarray)
    yPredictW[wIdxsFillPutarray,:,:,:] = yPredictInit[wIdxsFillPullarray,:,:,:]

    # Step 6.4.2 - avg/max width data
    for wId, wStart in enumerate(wStarts):
        if wId > 0 and wId < len(wStarts) - 1:
          wIdxsFillPutarray = np.arange(wStart, wStart + wOverlap)
          wIdxsFillPullarray1 = np.arange(wId*wGrid - wOverlap, wId*wGrid)
          wIdxsFillPullarray2 = np.arange(wId*wGrid           , wId*wGrid + wOverlap)
          yPredictW[wIdxsFillPutarray,:,:,:] = (yPredictInit[wIdxsFillPullarray1,:,:,:] + yPredictInit[wIdxsFillPullarray2,:,:,:])/2.0
        elif wId == len(wStarts) - 1:
          wIdxsFillPutarray = np.arange(wStart, wStart + wOverlapLastGrid)
          wIdxsFillPullarray1 = np.arange(wId*wGrid - wOverlapLastGrid, wId*wGrid)
          wIdxsFillPullarray2 = np.arange(wId*wGrid                   , wId*wGrid + wOverlapLastGrid)
          yPredictW[wIdxsFillPutarray,:,:,:] = (yPredictInit[wIdxsFillPullarray1,:,:] + yPredictInit[wIdxsFillPullarray2,:,:,:])/2.0
    
    # Step 6.5.1 - fill height data
    hIdxsFillPutarray = []
    hIdxsFillPullarray = []
    lenPutPrev = 0
    lenPullprev = 0
    for hId, hStart in enumerate(hStarts):
      if hId == 0:
        hIdxsFillPutarray.extend(list(np.arange(0, hGrid-hOverlap)))
        hIdxsFillPullarray.extend(list(np.arange(0, hGrid-hOverlap)))
      elif hId !=0 and hId < len(hStarts) - 2:
        hIdxsFillPutarray.extend(list(np.arange(hStart+hOverlap, hStart+hOverlap + hGrid - 2*hOverlap)))
        hIdxsFillPullarray.extend(list(np.arange((hId)*hGrid + hOverlap, (hId+1)*hGrid - hOverlap)))
      elif hId == len(hStarts) - 2:
        hIdxsFillPutarray.extend(list(np.arange(hStart+hOverlap, hStart+hOverlap + hGrid - hOverlap - hOverlapLastGrid)))
        hIdxsFillPullarray.extend(list(np.arange((hId)*hGrid + hOverlapLastGrid, (hId+1)*hGrid - hOverlap)))
      elif hId == len(hStarts) - 1:
        hIdxsFillPutarray.extend(list(np.arange(hStart+hOverlapLastGrid, hStart+hOverlapLastGrid + hGrid-hOverlapLastGrid)))
        hIdxsFillPullarray.extend(list(np.arange((hId)*hGrid + hOverlapLastGrid, (hId+1)*hGrid)))
      # print (len(hIdxsFillPutarray), len(hIdxsFillPullarray), len(hIdxsFillPutarray) - lenPutPrev, len(hIdxsFillPullarray)-lenPullprev)
      lenPutPrev = len(hIdxsFillPutarray)
      lenPullprev = len(hIdxsFillPullarray)
    yPredictH[:,hIdxsFillPutarray,:,:] = yPredictW[:,hIdxsFillPullarray,:,:]

    # Step 6.5.2 - avg/max height data
    for hId, hStart in enumerate(hStarts):
      if hId > 0 and hId < len(hStarts) - 1:
        hIdxsFillPutarray = np.arange(hStart, hStart + hOverlap)
        hIdxsFillPullarray1 = np.arange(hId*hGrid - hOverlap, hId*hGrid)
        hIdxsFillPullarray2 = np.arange(hId*hGrid           , hId*hGrid + hOverlap)
        yPredictH[:,hIdxsFillPutarray,:,:] = (yPredictW[:,hIdxsFillPullarray1,:,:] + yPredictW[:,hIdxsFillPullarray2,:,:])/2.0
      elif hId == len(hStarts) - 1:
        hIdxsFillPutarray = np.arange(hStart, hStart + hOverlapLastGrid)
        hIdxsFillPullarray1 = np.arange(hId*hGrid - hOverlapLastGrid, hId*hGrid)
        hIdxsFillPullarray2 = np.arange(hId*hGrid                   , hId*hGrid + hOverlapLastGrid)
        yPredictH[:,hIdxsFillPutarray,:,:] = (yPredictW[:,hIdxsFillPullarray1,:,:] + yPredictW[:,hIdxsFillPullarray2,:,:])/2.0

    # Step 6.5.1 - fill depth data
    dIdxsFillPutarray = []
    dIdxsFillPullarray = []
    lenPutPrev = 0
    lenPullprev = 0
    for dId, dStart in enumerate(dStarts):
      if len(dStarts) >= 3:
        if dId == 0:
          dIdxsFillPutarray.extend(list(np.arange(0, dGrid-dOverlap)))
          dIdxsFillPullarray.extend(list(np.arange(0, dGrid-dOverlap)))
        elif dId !=0 and dId < len(dStarts) - 2:
          dIdxsFillPutarray.extend(list(np.arange(dStart+dOverlap, dStart+dOverlap + dGrid - 2*dOverlap)))
          dIdxsFillPullarray.extend(list(np.arange((dId)*dGrid + dOverlap, (dId+1)*dGrid - dOverlap)))
        elif dId == len(dStarts) - 2:
          dIdxsFillPutarray.extend(list(np.arange(dStart+dOverlap, dStart+dOverlap + dGrid - dOverlap - dOverlapLastGrid)))
          dIdxsFillPullarray.extend(list(np.arange((dId)*dGrid + dOverlap, (dId+1)*dGrid - dOverlapLastGrid)))
        elif dId == len(dStarts) - 1:
          dIdxsFillPutarray.extend(list(np.arange(dStart+dOverlapLastGrid, dStart+dOverlapLastGrid + dGrid-dOverlapLastGrid)))
          dIdxsFillPullarray.extend(list(np.arange((dId)*dGrid + dOverlapLastGrid, (dId+1)*dGrid)))
      elif len(dStarts) == 2:
        if dId == 0:
          dIdxsFillPutarray.extend(list(np.arange(0, dGrid-dOverlapLastGrid)))
          dIdxsFillPullarray.extend(list(np.arange(0, dGrid-dOverlapLastGrid)))
        else:
          dIdxsFillPutarray.extend(list(np.arange(dStart+dOverlapLastGrid, dStart+dOverlapLastGrid + dGrid-dOverlapLastGrid)))
          dIdxsFillPullarray.extend(list(np.arange((dId)*dGrid + dOverlapLastGrid, (dId+1)*dGrid)))
      # print (len(dIdxsFillPutarray), len(dIdxsFillPullarray), ' || ',len(dIdxsFillPutarray) - lenPutPrev, len(dIdxsFillPullarray)-lenPullprev, ' || ', dOverlap - dOverlapLastGrid)
      lenPutPrev = len(dIdxsFillPutarray)
      lenPullprev = len(dIdxsFillPullarray)
    yPredictD[:,:,dIdxsFillPutarray,:] = yPredictH[:,:,dIdxsFillPullarray,:]

    # Step 6.5.2 - avg/max depth data
    for dId, dStart in enumerate(dStarts):
      if dId > 0 and dId < len(dStarts) - 1:
        dIdxsFillPutarray = np.arange(dStart, dStart + dOverlap)
        dIdxsFillPullarray1 = np.arange(dId*dGrid - dOverlap, dId*dGrid)
        dIdxsFillPullarray2 = np.arange(dId*dGrid           , dId*dGrid + dOverlap)
        yPredictD[:,:,dIdxsFillPutarray,:] = (yPredictH[:,:,dIdxsFillPullarray1,:] + yPredictH[:,:,dIdxsFillPullarray2,:])/2.0
      elif dId == len(dStarts) - 1:
        dIdxsFillPutarray = np.arange(dStart, dStart + dOverlapLastGrid)
        dIdxsFillPullarray1 = np.arange(dId*dGrid - dOverlapLastGrid, dId*dGrid)
        dIdxsFillPullarray2 = np.arange(dId*dGrid                   , dId*dGrid + dOverlapLastGrid)
        yPredictD[:,:,dIdxsFillPutarray,:] = (yPredictH[:,:,dIdxsFillPullarray1,:] + yPredictH[:,:,dIdxsFillPullarray2,:])/2.0
  
  getGPUInfo(additional_text='Prediction completed! Rendering results')
  return yPredictD

############################################################
#                     DATALOADER RELATED                   #
############################################################

def splitRange(widthTotal, widthGrid, widthOverlap, resType='boundary'):
  resRange = []
  resBoundary = []

  A = np.arange(widthTotal)
  wStart = 0
  wEnd = widthGrid
  while(wEnd < len(A)):
    resRange.append(np.arange(wStart, wEnd))
    resBoundary.append([wStart, wEnd])
    wStart = wStart + widthGrid - widthOverlap
    wEnd = wStart + widthGrid
  
  resRange.append(np.arange(len(A)-widthGrid, len(A)))
  resBoundary.append([len(A)-widthGrid, len(A)])
  if resType == 'boundary':
    return resBoundary
  elif resType == 'range':
    return resRange

def getTotalGrids(volumeShape, gridSize, gridOverlap):
  # gridSize = [W,H,D]
  wGridCount = len(splitRange(volumeShape[2], widthGrid=gridSize[0], widthOverlap=gridOverlap[0], resType='boundary'))
  hGridCount = len(splitRange(volumeShape[1], widthGrid=gridSize[1], widthOverlap=gridOverlap[1], resType='boundary'))
  dGridCount = len(splitRange(volumeShape[0], widthGrid=gridSize[2], widthOverlap=gridOverlap[2], resType='boundary'))
  totalGrids = wGridCount*hGridCount*dGridCount
  return totalGrids

def getDataLoaderGrid(img, gridSize, gridOverlap, minHU=-400, maxHU=1000):
    """
     - img: [D,H,W]
    """
    try:
      import numpy as np
      import tensorflow as tf

      img = np.array(img)
      
      
      gridsIdxsWidth  = splitRange(img.shape[0], gridSize[0], gridOverlap[0], resType='boundary')
      gridsIdxsHeight = splitRange(img.shape[1], gridSize[1], gridOverlap[1], resType='boundary')
      gridsIdxsDepth  = splitRange(img.shape[2], gridSize[2], gridOverlap[2], resType='boundary')
      # print (' -[utils.getDataLoaderGrid()] totalGrids: ', len(gridsIdxsWidth)*len(gridsIdxsHeight)*len(gridsIdxsDepth))

      for gridIdxsWidth in gridsIdxsWidth:
        for gridIdxsHeight in gridsIdxsHeight:
          for gridIdxsDepth in gridsIdxsDepth:
            grid = img[gridIdxsWidth[0]:gridIdxsWidth[1],gridIdxsHeight[0]:gridIdxsHeight[1],gridIdxsDepth[0]:gridIdxsDepth[1]]
            grid[grid <= minHU] = minHU
            grid[grid >= maxHU] = maxHU
            grid = (grid - minHU) / (maxHU - minHU)
            grid = np.expand_dims(grid, axis=0)
            grid = np.expand_dims(grid, axis=-1)
      
            grid = tf.constant(grid, dtype=tf.float32)
            meta = 'ypred__w_{}_{}__h_{}_{}__d_{}_{}'.format(gridIdxsWidth[0],gridIdxsWidth[1],gridIdxsHeight[0],gridIdxsHeight[1],gridIdxsDepth[0],gridIdxsDepth[1])
            yield (grid, meta)
    
    except:
      traceback.print_exc()
      return None
      