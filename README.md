# XARLabs_Task
This repository provides a pipeline for fully automated processing of head CT/MRI scans and detection of nose tips from face point-cloud and face_mesh

### Script Overview

- **DICOM_reader.py**  
  Reads a DICOM dataset, extracts image metadata, and reconstructs the image volume as a 3D matrix.

- **DeepMVLM.py**  
  Extracts facial landmarks from a 3D head/face mesh.  
  ⚠️ *Note:* To use this script, download the github repo of the model here: https://github.com/RasmusRPaulsen/Deep-MVLM and make sure to place it inside the directory containing these scripts.
  ⚠️ *Note:* Inside `Deep-MVLM/configs`, edit the `DTU3D-depth-MRI.json` file in this way:
  ```json
  "pre-align": {
		"align_center_of_mass" : true,
		"rot_x": -90,
		"rot_y": 0,
		"rot_z": 0,
		"scale": 1,
		"write_pre_aligned": true
	}

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

- **requirements.txt**   
  Contains the libraries required

![result](https://github.com/user-attachments/assets/a7287f5d-d68c-414c-a7be-2918fab3cee8)
