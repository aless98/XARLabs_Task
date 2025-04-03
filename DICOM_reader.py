import numpy as np
import pydicom.fileset
import pyvista as pv
import os
import pydicom
from pydicom.filereader import dcmread
import matplotlib.pyplot as plt
import sys
import glob
import vtk
from vtkmodules.util.numpy_support import vtk_to_numpy, numpy_to_vtk
import nibabel as nib


# Function to READ the DICOM folder and extract the image slices
def Parse_DICOM(folder_path):
    
    # searching in the directory until i find the DICOMDIR file
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.upper() == "DICOMDIR":
                dicom_path = os.path.join(root, file)
  
    dicomdir = dcmread(dicom_path)
    base_dir = os.path.dirname(dicom_path)

    dicom_files = []

    # Finding the images and reading them one by one
    for record in dicomdir.DirectoryRecordSequence:
        if record.DirectoryRecordType == "IMAGE":
            ref_file = record.ReferencedFileID  # ad esempio ['SOTTODIR', 'IMMAGINE001']
            dicom_file = dcmread(os.path.join(base_dir,*ref_file))
            dicom_files.append(dicom_file)
    

    # Read all the dico_files and storing them in a list. I am skipping all the file that do not have a "SliceLocation" è.g SCOUT scans
    slices = []
    for f in dicom_files:
        if hasattr(f, "SliceLocation"):
            slices.append(f)           
    
    slices = sorted(slices, key=lambda s: s.SliceLocation) 
   
    # Creating a 3D matrix representing the volume of the image (i am stacking the slices along the z direction of the matrix)
    img_shape = list(slices[0].pixel_array.shape)
    img_shape.append(len(slices))
    image_3d = np.zeros(img_shape)
    for i, s in enumerate(slices):
        img2d = s.pixel_array
        
        image_3d[:, :, i] = img2d       
    
    # Extracting useful information about the Images: PixelSpacing(dx,dy,dz), origin and orientation 
    pixel_spacing = slices[0].PixelSpacing 
    slice_thickness = slices[0].SpacingBetweenSlices 
    origin = slices[0].ImagePositionPatient 
    image_orient = slices[0].ImageOrientationPatient
    
    #Constructing the Direction_Matrix from ijk--> RCS(Reference Coordinate System).
    row_dir = np.array(image_orient[0:3])  
    col_dir = np.array(image_orient[3:6])  
    slice_dir = np.cross(row_dir, col_dir)

    direction_matrix_np = np.stack((row_dir, col_dir, slice_dir), axis=1)  # shape (3,3)


    return image_3d, origin, pixel_spacing, slice_thickness, direction_matrix_np

