import gc
from pickle import TRUE
import time
import importlib
import traceback
import numpy as np
from pathlib import Path
from numpy.lib.npyio import save

import vtk, qt, ctk, slicer
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin, modulePath

import src.util as myUtil
import src.config as myConfig

DEBUG_QUICK=False
DEBUG_FILE=False
DEBUG_MODEL=True

####################################################
#                   MODULE INIT                    #
####################################################

class AISegmentation(ScriptedLoadableModule):

  def __init__(self, parent):
    ScriptedLoadableModule.__init__(self, parent)
    self.parent.title = "AISegmentation" 
    self.parent.categories = ["Machine Learning"] 
    self.parent.dependencies = ["AISegmentationCLI"] 
    self.parent.contributors = ["Prerak Mody (LUMC, The Netherlands)"]
    self.parent.helpText = """
        This module performs automated segmentation of the head and neck CT data
        """
    self.parent.acknowledgementText = """
        This project was funded by Varian Medical Systems, Netherlands 
        """

    # Additional initialization step after application startup is complete
    # slicer.app.connect("startupCompleted()", registerSampleData)

class AISegmentationSettingsPanel(ctk.ctkSettingsPanel):
  def __init__(self, *args, **kwargs):
    ctk.ctkSettingsPanel.__init__(self, *args, **kwargs)


####################################################
#                   MODULE UI                      #
####################################################

class AISegmentationWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
  
  def __init__(self, parent=None):
    """
    Called when the user opens the module the first time and the widget is initialized.
    """
    ScriptedLoadableModuleWidget.__init__(self, parent)
    VTKObservationMixin.__init__(self)  # needed for parameter node observation
    self._logic = None
    self._parameterNode = None
    self._updatingGUIFromParameterNode = False

  def setup(self):
    """
    Called when the user opens the module the first time and the widget is initialized.
    """
    ScriptedLoadableModuleWidget.setup(self) # reload and test buttons (can remove in production releases)

    # Step 0 - Debug params
    self._debug()

    # Step 1 - Input Section
    myInputCollapsibleButton = ctk.ctkCollapsibleButton()
    myInputCollapsibleButton.text = "Inputs"
    self.layout.addWidget(myInputCollapsibleButton)
    
    # Step 1.1.0 - Create Form
    parametersFormLayout = qt.QFormLayout(myInputCollapsibleButton)

    # Step 1.1.1 - Load Button Raw
    self.img = slicer.vtkMRMLScalarVolumeNode()
    self.loadButtonRaw = qt.QPushButton("Load CT Scan")
    self.loadButtonRaw.toolTip = "Loads a .nrrd file"
    self.loadButtonRaw.enabled = True
    parametersFormLayout.addRow(self.loadButtonRaw)
    self.folderPathRaw = None
    
    # Step 1.1.2 - Load Button Mask
    self.mask = slicer.vtkMRMLScalarVolumeNode()
    self.loadButtonMask = qt.QPushButton("Load Atlas")
    self.loadButtonMask.toolTip = "Load a .nrrd file"
    self.loadButtonMask.enabled = True
    parametersFormLayout.addRow(self.loadButtonMask)

    # Step 1.1.3 - Load Button Model
    self.loadButtonModel = qt.QPushButton("Load AI Contouring Model")
    self.loadButtonModel.toolTip = "Loads a 3D Unet"
    self.loadButtonModel.enabled = False
    parametersFormLayout.addRow(self.loadButtonModel)

    # Step 1.1.4 - Load Button Model
    self.releaseButtonModel = qt.QPushButton("Free GPU Memory")
    self.releaseButtonModel.toolTip = "Deletes the deep learning model inside the GPU"
    self.releaseButtonModel.enabled = False
    parametersFormLayout.addRow(self.releaseButtonModel)

    # Step 1.1.5 - Test Button
    self.testButtonModel = qt.QPushButton("Test Button")
    self.testButtonModel.toolTip = "Just a temporary debuggin button"
    self.testButtonModel.enabled = True
    parametersFormLayout.addRow(self.testButtonModel)
    
    # Step 1.2 - Button connectors
    self.loadButtonRaw.connect('clicked(bool)', self._onLoadButtonRaw) # self.loadButtonRaw.clicked.connect(self.onLoadButtonRaw)
    self.loadButtonRaw.connect("clicked(bool)", self._resetModelButtonStatus)
    self.loadButtonMask.connect('clicked(bool)', self._onLoadButtonMask)
    self.loadButtonModel.connect('clicked(bool)', self._onLoadButtonModel)
    self.releaseButtonModel.connect('clicked(bool)', self._onReleaseButtonModel)
    self.testButtonModel.connect('clicked(bool)', self._onTestButtonModel)
    
    # Step 2 - Logic module
    self._logic = AISegmentationLogic()
    
    # Step 3 - Load modules
    self._loadPythonModules()
    myUtil.getGPUInfo()

    # Step 4 - Tmp
    if DEBUG_QUICK:
      self.loadButtonRaw.clicked()
      time.sleep(1)
      # self.loadButtonModel.clicked()

  def _debug(self):
    if DEBUG_MODEL:
      self._folderPathModel = 'D:/HCAI/Project1-AutoSeg/Code/competitions/2015_MICCAI_HNAuto/models/UNet3D_DiceWeighted_NNeig_seed42_v2/ckpt_epoch300'
    else:
      self._folderPathModel = str(Path.home())
    if DEBUG_FILE:
      self._folderPathRaw = 'D:/HCAI/Project1-AutoSeg/Code/competitions/medical_dataloader/data2/HaN_MICCAI2015/processed/test_offsite/data_3D/0522c0555'
    else:
      self._folderPathRaw = str(Path.home())

  def _loadPythonModules(self):
    """
    This loads or installs python modules that are needed for computation in this module
    Note: 3D Slicer's version of python comes only with some python packages pre-built  
    """
    
    def progressCallbackLoading(progressDialog, progressLabel, progressValue):
        progressDialog.labelText = 'Loading %s ...' % progressLabel
        slicer.app.processEvents()
        progressDialog.setValue(progressValue)
        time.sleep(0.2)
        slicer.app.processEvents()
        # cancelled = progressDialog.wasCanceled
        # return cancelled
    
    def progressCallbackInstalling(progressDialog, progressLabel, progressValue):
        progressDialog.labelText = 'Installing %s ...' % progressLabel
        slicer.app.processEvents()
        progressDialog.setValue(progressValue)
        slicer.app.processEvents()
        
        # cancelled = progressDialog.wasCanceled
        # return cancelled

    import time
    import importlib
    pythonModules = {
      'tensorflow':'tensorflow'
      , 'py3nvml': 'py3nvml'
      , 'nrrd': 'pynrrd'
    }

    progressDialog = myUtil.createProgressDialog(parent=self, value=0, maximum=len(pythonModules), windowTitle='Python Modules'
                              #, kwargs={'setCancelButton':None}
                          )
    count = 1
    for pythonModule in pythonModules:
      try:
        progressCallbackLoading(progressDialog, progressLabel=pythonModule, progressValue=count)
        importlib.import_module(pythonModule)
      except:
        try:
          progressCallbackInstalling(progressDialog, progressLabel=pythonModule, progressValue=count)
          slicer.util.pip_install(pythonModules[pythonModule])
          progressCallbackLoading(progressDialog, progressLabel=pythonModule, progressValue=count)
          importlib.import_module(pythonModule)
        except:
          traceback.print_exc()
      count += 1
    progressDialog.close()

    # self.util = importlib.import_module('util', 'src')
    # self.config = importlib.import_module('config', 'src')

  def _onLoadButtonRaw(self):
    
    title = 'Select raw image file '
    self.pathRawFile = qt.QFileDialog().getOpenFileName(qt.QFileDialog(), title, self._folderPathRaw)
      
    if len(self.pathRawFile):
      if Path(self.pathRawFile).exists():
        self.img = slicer.util.loadVolume(self.pathRawFile) # <class 'MRMLCorePython.vtkMRMLScalarVolumeNode'>
        file_identifier = Path(self.pathRawFile).parts[-2] + '__' + Path(self.pathRawFile).parts[-1]
        self.img.SetName(file_identifier.split('.')[0])
        self._setSliceNodeVisibility()
        self._setAllSegmentationNodesVisibility(visible=False)
        self._folderPathRaw = Path(self.pathRawFile).parent.absolute()
      else:
        print (' - [ERROR][MyModuleWidget.onLoadButtonRaw()] self.pathRawFile) does not exist: ', self.pathRawFile)
    else:
      print (' - [INFO][MyModuleWidget.onLoadButtonRaw()] No file selected for self.pathRawFile')
    
  def _onLoadButtonMask(self):
    parent = qt.QFileDialog()
    title = 'Select mask file '
    self.pathMaskFile = qt.QFileDialog().getOpenFileName(parent, title, self._folderPathRaw)

    if len(self.pathMaskFile):
      try:
        self.mask = slicer.util.loadSegmentation(self.pathMaskFile) # <class 'MRMLCorePython.vtkMRMLSegmentationNode'>
      except:
        tmp = slicer.util.loadLabelVolume(self.pathMaskFile)
        print (tmp)
        self.mask = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLSegmentationNode')
        slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(tmp, self.mask)
        slicer.mrmlScene.RemoveNode(tmp)

      self.mask.GetSegmentation().SetConversionParameter(
        slicer.vtkBinaryLabelmapToClosedSurfaceConversionRule.GetSmoothingFactorParameterName(),"0.0")
      self.mask.CreateClosedSurfaceRepresentation()

      self._logic.processMask(self.mask)
      self._folderPathRaw = Path(self.pathMaskFile).parent.absolute()
    else:
      print (' - [INFO][MyModuleWidget.onLoadButtonRaw()] No file selected for self.pathRawFile')
    
  def _onLoadButtonModel(self):
    parent = qt.QFileDialog()
    title = 'Select model directory '
    self.modelPath = qt.QFileDialog().getExistingDirectory(parent, title, self._folderPathModel)
    
    if len(self.modelPath):
      if Path(self.modelPath).exists():
        self.releaseButtonModel.enabled = True
        # self._logic.processImg(self.img, self.modelPath)
        self._logic.processImageViaCLI(self.img, self.modelPath)
      else:
        print (' - [ERROR][MyModuleWidget.onLoadButtonModel()] Issue with model path: ', self.modelPath)
    else:
      print (' - [ERROR][MyModuleWidget.onLoadButtonModel()] No model path is provided: ', self.modelPath)

  def _onReleaseButtonModel(self):
    self.logic.releaseModel()

  def _resetModelButtonStatus(self):
    gpu_present = myUtil.getTFlowGPUStatus()
    if self.img.GetDisplayNodeID() is not None and gpu_present:
      self.loadButtonModel.enabled = True

  def _setSliceNodeVisibility(self):
    slicer.util.getNode('vtkMRMLSliceNodeRed').SetSliceVisible(1)
    slicer.util.getNode('vtkMRMLSliceNodeGreen').SetSliceVisible(1)
    slicer.util.getNode('vtkMRMLSliceNodeYellow').SetSliceVisible(1)
  
  def _setAllSegmentationNodesVisibility(self, visible=False):

    for segNode in slicer.util.getNodesByClass('vtkMRMLSegmentationNode'):
      displayNode = segNode.GetDisplayNode()
      if displayNode is not None:
        displayNode.SetAllSegmentsVisibility(visible)

  def _onTestButtonModel(self):
    progressBar = myUtil.DoubleProgressBar()
    for i in range(11):
      progressBar.updateProgressBar1(i*10)
      time.sleep(0.2)
    print (progressBar.progressBar1.value)

    for i in range(11):
      progressBar.updateProgressBar2(i*10)
      time.sleep(0.2)
    print (progressBar.progressBar2.value)

  def _cleanup(self):
    """
    Called when the application closes and the module widget is destroyed.
    """
    pass
  
  def _onReload(self):

    ScriptedLoadableModuleWidget.onReload(self)


####################################################
#                 MODULE LOGIC                     #
####################################################

class AISegmentationLogic(ScriptedLoadableModuleLogic):
  
  def __init__(self):
    """
    Called when the logic class is instantiated. Can be used for initializing member variables.
    """
    ScriptedLoadableModuleLogic.__init__(self)
    self.LABELMAP = myConfig.LABEL_MAP
    self.LABELCOLORS = myConfig.LABEL_COLORS

    self.model = None

  def getLabelName(self, labelId):
    for labelName in self.LABELMAP:
      if int(self.LABELMAP[labelName]) == int(labelId):
        return labelName

  def releaseModel(self):
    myUtil.getGPUInfo()
    if self.model is not None:
      print (self.model)
      del self.model
      self.model = None
      myUtil.getGPUInfo()
    myUtil.getGPUInfo()
  
  def processMask(self, mask):
    for seg_id in range(mask.GetSegmentation().GetNumberOfSegments()):
        seg_obj = mask.GetSegmentation().GetNthSegment(seg_id) # <class 'vtkSegmentationCorePython.vtkSegment'>
        seg_obj.SetName(self.getLabelName(seg_obj.GetLabelValue()))
        seg_obj.SetColor(np.array(self.LABELCOLORS[seg_obj.GetLabelValue()])[:-1]/255.0)

  def processImageViaCLI(self, volNode, modelPath):
    try:

      # Step 1 - Setup CLI parameters
      self.predLabelMapVolumeNode = myUtil.getLabelMapVolumeNodeFromVolumeNode(volNode)
      inputVolumeNodeName = volNode.GetName()
      self.tmpPath = Path(slicer.app.temporaryPath).joinpath(inputVolumeNodeName)
      
      cliParameters = {
        'inputVolume': volNode.GetID()
        , 'outputVolume': self.predLabelMapVolumeNode.GetID()
        , 'inputVolumeName': inputVolumeNodeName
        , 'modelPath': modelPath
        , 'predThresh': 0.6
      }

      # Step 2 - Run CLI node
      MyModuleCLI = slicer.modules.aisegmentationcli
      cliNode = None
      self.asyncFlag = True
      if self.asyncFlag:
        cliNode = slicer.cli.run(MyModuleCLI, cliNode, cliParameters)
      else:
        cliNode = slicer.cli.runSync(MyModuleCLI, cliNode, cliParameters)

      # Step 3 - Track progress
      self.progressBar = myUtil.DoubleProgressBar()
      self.cliObserver = cliNode.AddObserver('ModifiedEvent', self.onCliModified)

    except:
      traceback.print_exc()

  def onCliModified(self, caller, event):

    if caller.IsA('vtkMRMLCommandLineModuleNode'):
      progress = caller.GetProgress()
      print (' - caller.GetStatus(): ', caller.GetStatus(), caller.GetStatusString(), progress, self.progressBar.progressBar1.value, self.progressBar.progressBar2.value)

      if caller.GetStatus() == 2: # Running
        if self.progressBar.progressBar2.value == 0:
          self.progressBar.updateProgressBar1(progress*2)
          if int(progress*2) == 100:
            self.progressBar.updateProgressBar2(1)
            myUtil.getGPUInfo('Prediction Done')  
          else:
            myUtil.getGPUInfo('Predicting: {}%'.format(progress*2))
        elif self.progressBar.progressBar1.value == 100:
          self.progressBar.updateProgressBar2(progress*2)
          if progress*2 == 100:
            myUtil.getGPUInfo('Post-Processing Done')
          else:
            myUtil.getGPUInfo('Post-Processing: {}%'.format(progress*2))

      elif caller.GetStatus() == 32: # Completed
        self.progressBar.close()

        print ('')
        print (' ==================== OUTPUT ===================== ')
        for each in caller.GetOutputText().split('\n'): 
          print (each)
        
        # Setup Output
        predSegmentationNode = None
        if self.asyncFlag:
          import nrrd
          results, _  = nrrd.read(Path(self.tmpPath, 'result.nrrd'))
          results = myUtil.resetAxesOrderFromRawFile(results) 
          myUtil.updateLabelMapVolumeNode(self.predLabelMapVolumeNode, results)
          predSegmentationNode = myUtil.convertLabelMapVolumeToSegmentationMap(self.predLabelMapVolumeNode)
        else:
          predSegmentationNode = myUtil.convertLabelMapVolumeToSegmentationMap(self.predLabelMapVolumeNode)
        
        self.processMask(predSegmentationNode)
        myUtil.getGPUInfo(additional_text='Rendering completed')
    
      elif caller.GetStatus() == 96: # Error
        self.progressBar.close()

        print ('')
        print (' ==================== OUTPUT ===================== ')
        for each in caller.GetOutputText().split('\n'): 
          print (each)
        
        print ('')
        print (' ==================== ERROR ===================== ')
        print (caller.GetErrorText())
        myUtil.getGPUInfo('Some Error')
    
    else:
      print (' - Different caller: ', caller)

  def processImg(self, imgNode, modelPath, verbose=True):
    
    try:

      # Step 1 - Get image data
      img_array = slicer.util.arrayFromVolume(imgNode) # [D,H,W] slicer.util.arrayFromVolume(img)
      img_array = np.moveaxis(img_array, [0,1,2], [2,1,0]) # [D,W,H] -> [W,H,D]
      print (' - [processImg()] img: ', img_array.shape)

      # Step 2 - Predict
      if 1:
        tmpPath = Path(slicer.app.temporaryPath).joinpath(imgNode.GetAttribute(myConfig.ATTR_FILENAME).split('.')[0])
        modelGridParams = myUtil.predict(img_array, self.model, modelPath, tmpPath)
        # modelGridParams = myUtil.predictTemp(img_array, self.model, modelPath, tmpPath)

      # Step 3 - Combine predictions
      if 1:
        yPredictProb = myUtil.joinGrids(img_array.shape, modelGridParams, tmpPath)
          
        # Step 4 - Show predictions
        yPredictProb[yPredictProb < 0.6] = 0
        yPredict = np.argmax(yPredictProb, axis=3)
        yPredict = np.moveaxis(yPredict, [2,1,0], [0,1,2]) 
        yPredict = yPredict.astype(np.int8)
        
        t99 = time.time()
        # predMask.CreateDefaultDisplayNodes()
        predLabelMapVolumeNode = myUtil.getNewLabelMapVolumeNode(spacing=imgNode.GetSpacing(), origin=imgNode.GetOrigin())
        t0 = time.time()
        slicer.util.updateVolumeFromArray(predLabelMapVolumeNode, yPredict)
        print (time.time() - t0)
        t0 = time.time()
        predSegmentationNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLSegmentationNode')
        # predSegmentationNode.CreateDefaultDisplayNodes()
        # predSegmentationNode.SetReferenceImageGeometryParameterFromVolumeNode(imgNode)
        slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(predLabelMapVolumeNode, predSegmentationNode)
        print (time.time() - t0)
        # slicer.mrmlScene.RemoveNode(predLabelMapVolumeNode)
        predSegmentationNode.GetSegmentation().SetConversionParameter(
            slicer.vtkBinaryLabelmapToClosedSurfaceConversionRule.GetSmoothingFactorParameterName(),"0.0")
        t0 = time.time()
        predSegmentationNode.CreateClosedSurfaceRepresentation()
        print (time.time() - t0)
        self.processMask(predSegmentationNode)
        myUtil.getGPUInfo(additional_text='Rendering completed')
        print (time.time() - t99)

        print ('---------------------------------------------')
        print (' - img spacing: ', imgNode.GetSpacing())
        print (' - pred spacing: ', predLabelMapVolumeNode.GetSpacing())
        print (' - img origin: ', imgNode.GetOrigin())
        print (' - pred origin: ', predLabelMapVolumeNode.GetOrigin())
        print (' - img size: ', slicer.util.arrayFromVolume(imgNode).shape)
        print (' - pred size: ', slicer.util.arrayFromVolume(predLabelMapVolumeNode).shape)

    except:
      traceback.print_exc()
  

####################################################
#                 MODULE TEST                      #
####################################################
class AISegmentationTest(ScriptedLoadableModuleTest):
  """
  This is the test case for your scripted module.
  Uses ScriptedLoadableModuleTest base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def setUp(self):
    """ Do whatever is needed to reset the state - typically a scene clear will be enough.
    """
    slicer.mrmlScene.Clear()

  def runTest(self):
    """Run as few or as many tests as needed here.
    """
    pass

  def test_AISegmentation1(self):
    pass