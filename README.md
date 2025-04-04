# XARLabs_Task
This repository provides a pipeline for fully automated processing of head CT/MRI scans and detection of nose tips from face point-cloud and face_mesh

### Script Overview

### Script Overview

- **DICOM_reader.py**  
  Reads a DICOM dataset, extracts image metadata, and reconstructs the image volume as a 3D matrix.

- **DeePMVLM.py**  
  Extracts facial landmarks from a 3D head/face mesh.  
  ⚠️ *Note:* To use this script, make sure to place the `Deep-MVLM` folder inside the directory containing these scripts.

- **Head_extraction.py**  
  Automatically segments the patient's head using VTK libraries.

- **pointcloud_aligner.py**  
  Contains a class for point cloud registration using ICP with perturbation to improve convergence.

- **pointcloud_nose_tip.py**  
  Detects the nose tip from a point cloud using a geometric heuristic.

- **Main.py**  
  Run this script to perform head segmentation, nose tip detection, and registration between the face point cloud and the head isosurface.  
  📁 The directory path to set in the code should be the one containing the DICOM dataset.

- **.ply** and **.stl** files  
  These contain the face point clouds and segmented head models of two patients.


![result](https://github.com/user-attachments/assets/ef2a1ed5-5373-41ea-8829-ad4b6b96f746)
