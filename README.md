# SlicerAISegmentation
This 3D Slicer Extension contains modules on loading CT data (3D volume) and auto-contouring (a.k.a segmenting) multiple organs in the Head and Neck (HaN) Region 


# Modules
3D Slicer as a tool for medical image analysis is made of many modules, each implementing a unique functionality. A 3D Slicer extension can be composed of multiple modules.

## Module 1 - AISegmentation
This is a [3D Slicer Python Scripted Module](https://www.slicer.org/wiki/Documentation/Nightly/Developers/Modules#Scripted_Modules) which accept a 3D CT volume. Mostly, this module handles the UI related aspects of this extension.

## Module 2 - AISegmentation CLI
This is a [3D Slicer Command Line Interface(CLI)](https://www.slicer.org/wiki/Documentation/Nightly/Developers/Modules#Command_Line_Interface_(CLI)). This can asynchronously run a Neural Network (so as to not block the main 3D Slicer UI thread) and return predicted contours. To be run asynchronously, it has to be called from within module 1

# Installation
1. Clone this repository
2. Download [3DSlicer](https://download.slicer.org/) [Version >= 4.11]
3. After installation
    - Open the application
    - Ctrl + F = "Welcome to Slicer". 
        - Ctrl + F is a shortcut to find a module within 3D Slicer
    - Click on _Customize Slicer_ in the "Welcome to Slicer" module
    - In the window that open, click on _Modules_
    - Click the double arrow (>>) correponding to _Additional Module Paths_
        - Click on _Add_
    - Add the _AISegmentation_ and _AISegmentationCLI_ module folders
    - Restart 3D Slicer
4. After restart
    - Ctrl + F = "AISegmentation"
    - Click on _Load CT Scan_ and select a CT scan (as a 3D volume)
    - Click on _Load AI Contouring Model_ (and load a tensorflow model directory)
        - After a model is selected, the application show progress bars.
        - Once the progress bars are finised, the predicted contours are shown in the 3D Viewer

# Pending
 - [ ] Adding DSC score (in Module 1)
 - [ ] Fixed Windowing on HaN CT scans (in Module 1) 


# References
 - [3D Slicer Python FAQ](https://www.slicer.org/wiki/Documentation/Nightly/Developers/FAQ/Python_Scripting)