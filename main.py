import numpy as np
import pyvista as pv
from pydicom.filereader import dcmread
import matplotlib.pyplot as plt
from vtkmodules.util.numpy_support import vtk_to_numpy, numpy_to_vtk
import open3d as o3d
import numpy as np
import pyvista as pv
import Head_extraction
import pointcloud_nose_tip
import DeepMVLM
import pointcloud_aligner
import time

# Directory where the DICOM files are stored
dicom_dir = "PZ1"

# Automatically extract the complete head as a mesh from the DICOM Images and save it as an .stl file
start_DICOM_extraction = time.time() 
Head_extraction.Head_segmentation(dicom_dir) #--> comment this if the head is already segmented
end_DICOM_extraction = time.time() 

print(f"Execution time for DICOM head extraction: {end_DICOM_extraction - start_DICOM_extraction:.4f} seconds") # compute the time needed to extract the face isosurface

# Load both the complete face extracted from the DICOM (head.stl) and the pointlcoud of the face --> N.B the pointcloud was trasformed randomly in the space to mimic a real-case scenrio in which the reference system are different
head_from_DICOM = pv.read("head1.stl")
face_pcd = pv.read("pcd_face1.ply")

# DICOM_nose_tip: Use the Deep learning based 3D landmark placement model to extract the DTU3D standard landamrk set (we are then only interested in the nose landmark)
start_DL = time.time() 
DICOM_nose_tip = DeepMVLM.DL_nose_tip("head1.stl")
end_DL = time.time()

print(f"Execution time for DL nose tip detection: {end_DL - start_DL:.10f} seconds") # compute the time needed to extract the nose tip from the face isosurface using Deep-Learning

# pcd_nose_tip: Extract the nose-tip using geometric heuristic
start_pcd = time.time() 
pcd_nose_tip = pointcloud_nose_tip.extract_nose_from_pcd(face_pcd)
end_pcd = time.time()

print(f"Execution time for pcd nose tip detection: {end_pcd - start_pcd:.10f} seconds") # compute the time needed to extract the nose tip from the face pointcloud

######### Plotter: for visualization purposes only #################################
pl = pv.Plotter()
pl.add_mesh(head_from_DICOM, color="peachpuff")
pl.add_mesh(face_pcd, color="red", point_size=1, render_points_as_spheres=True)
pl.add_mesh(pv.Sphere(center=DICOM_nose_tip, radius=3), color="red")
pl.add_axes_at_origin(x_color='red', y_color='green', z_color='blue')
pl.show_axes()
pl.add_mesh(pv.Sphere(center=pcd_nose_tip, radius=3), color="yellow")
pl.show()
###################################################################################

###############Registration########################################################

######### Convert the data from Pyvista to Open3D to use the voxel_down_sample function --> N.B this is used to accellerate the registration algorithm
source_pcd = o3d.geometry.PointCloud()
source_pcd.points = o3d.utility.Vector3dVector(np.asarray(face_pcd.points))

target_pcd = o3d.geometry.PointCloud()
target_pcd.points = o3d.utility.Vector3dVector(np.asarray(head_from_DICOM.points))

downsampled_source = source_pcd.voxel_down_sample(voxel_size = 2)
downsampled_target = target_pcd.voxel_down_sample(voxel_size = 2)

# Class containing the function to perform ICP+perturbation ---> we avoid local minima and arrive at the best solution.
start_regis = time.time() 
aligner = pointcloud_aligner.PointCloudAligner(tolerance = 0.01, max_iterations = 100, alpha = 5)
T_registration = aligner.align(np.asarray(downsampled_source.points),np.asarray(downsampled_target.points))
end_regis = time.time() 

print(f"Execution time for registration: {end_regis - start_regis:.10f} seconds") # compute the time needed to extract the nose tip from the face pointcloud

# The computed T_registration is the transformation that maps the point from the face pointcloud refrence frame to the head refrence frame #

# We transform the pointcloud to align it with the head
face_pcd.transform(T_registration, inplace = True)
registered_nose_tip = T_registration[:3,:3] @ pcd_nose_tip + T_registration[:3,3]

######### Plotter: for visualization purposes only #################################
pl = pv.Plotter()
pl.add_mesh(head_from_DICOM, color="peachpuff", opacity=0.8)
pl.add_mesh(pv.PolyData(face_pcd.points), color="red", point_size=4, render_points_as_spheres=True)
pl.add_mesh(pv.Sphere(center=DICOM_nose_tip, radius=3), color="red", label="DICOM Nose Tip")
pl.add_mesh(pv.Sphere(center=registered_nose_tip, radius=3), color="yellow", label="PCD Nose Tip")
pl.add_legend([["DICOM Nose Tip", "red"],["PCD Nose Tip", "yellow"]])
pl.show()
###################################################################################

# we compute the error between the 2 extracted and then aligned nose_tip points#

print(f"error_between_nose_tips:{np.linalg.norm(DICOM_nose_tip-registered_nose_tip)}")

