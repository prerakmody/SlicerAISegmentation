# AISegmentation
This is a [3D Slicer Python Scripted Module](https://www.slicer.org/wiki/Documentation/Nightly/Developers/Modules#Scripted_Modules) which accepts a 3D CT volume. 

# Code
1. [AISegmentation.py](.AISegmentation.py)
 - This contains four main classes
    - Entry point
        - slicer.ScriptedLoadableModule
    - UI Class  
        - slicer.ScriptedLoadableModuleWidget
        - Note: Through this class's functions, the CLI module is called
    - Logic Class
        - slicer.ScriptedLoadableModuleLogic
            - Note: This is only used if we want the NNet to run on the same thread as the UI thread. So the reader can ignore the [src](.src/) folder in this directory if thats not the requirement.
    - Testing Class (not implemented)  
        - slicer.ScriptedLoadableModuleTest

2. [CMakeLists.txt](CMakeLists.txt)
 - This is used if the module is submitted to the main 3D Slicer repo


# TO DO
 - [ ] Adding DSC score
 - [ ] Fixed Windowing on HaN CT scans 
 - [ ] Implement Testing Class 