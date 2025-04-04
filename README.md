# XARLabs_Task
This repository provides a pipeline for fully automated processing of head CT/MRI scans and detection of nose tips from face point-cloud and face_mesh

### Script Overview

- **DICOM_reader.py**  
  Reads a DICOM dataset, extracts image metadata, and builds a 3D matrix of the image volume.

- **DeePMVLM.py**  
  Extracts facial landmarks from a head/face mesh.

- **Head_extraction.py**  
  Automatically segments the patient’s head using VTK-based processing.

- **pointcloud_aligner.py**  
  Implements a class for point cloud registration using ICP with perturbation to improve convergence.

- **pointcloud_nose_tip.py**  
  Detects the nose tip from a point cloud using a geometric heuristic.
  
![result](https://github.com/user-attachments/assets/ef2a1ed5-5373-41ea-8829-ad4b6b96f746)
