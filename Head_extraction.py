import numpy as np
import pyvista as pv
import vtk
from vtkmodules.util.numpy_support import numpy_to_vtk
import DICOM_reader


pv.global_theme.allow_empty_mesh = True

# Function to perform MArchingCube algorithm and extract the patient head isosurface
def MarchingCubes(image , value):
    marchingCubes = vtk.vtkMarchingCubes()
    marchingCubes.SetInputData(image)
    marchingCubes.SetValue(0 , value)
    marchingCubes.Update()
    # Cleaning to merge duplicate points, and/or remove unused points and/or remove degenerate cells
    clean = vtk.vtkCleanPolyData()
    clean.SetInputConnection(marchingCubes.GetOutputPort())
    clean.Update()
    # Smoothing of the surface 
    smoother = vtk.vtkWindowedSincPolyDataFilter()
    smoother.SetInputConnection(clean.GetOutputPort())
    smoother.SetNumberOfIterations(40)
    smoother.BoundarySmoothingOn()
    smoother.SetFeatureAngle(120)
    smoother.SetPassBand(0.1)
    smoother.Update()
    return smoother.GetOutput()

#Function to perform Head segmentation
def Head_segmentation(dicom_dir):
    
    # Extract the 3D volume matrix from the DICOM file
    image_volume, origin, pixel_spacing, slice_thickness, direction_matrix_np = DICOM_reader.Parse_DICOM(dicom_dir)

    # Convert the numpy image array in VTK array 
    vtk_array = numpy_to_vtk(
        num_array=np.ravel(image_volume,order='F'),  
        deep=True,
        array_type=vtk.VTK_SHORT 
    )
    
    
    spacing = np.array([pixel_spacing[0],pixel_spacing[1],slice_thickness])
    origin = np.asarray([origin[0], origin[1], origin[2]])

    vtk_matrix = vtk.vtkMatrix3x3()
    vtk_matrix.Identity()
    for i in range(3):
        for j in range(3):
            vtk_matrix.SetElement(i, j, direction_matrix_np[i, j])
    
    # Create a vtkImageData instance and store the image matrix extracted from the DICOM
    image_data = vtk.vtkImageData()
    image_data.SetDimensions(image_volume.shape)
    image_data.GetPointData().SetScalars(vtk_array)
    image_data.SetSpacing(spacing)
    image_data.SetOrigin(origin)
    image_data.SetDirectionMatrix(vtk_matrix)
    
    # Create the AFFINE matrix to map from ijk(image space) --> LPS(Patient space). Info about the change of reference systems in imaging can be found here https://slicer.readthedocs.io/en/latest/user_guide/coordinate_systems.html
    affine = np.eye(4)
    affine[:3, :3] = direction_matrix_np @ np.diag(spacing) 
    affine[:3, 3] = origin 

    # Create a vtkTransform matrix and populate it with the AFFINE
    vtk_matrix = vtk.vtkMatrix4x4()
    vtk_matrix.Identity()
    for i in range(4):
        for j in range(4):
            vtk_matrix.SetElement(i, j, affine[i, j])
    
    # We need to invert it to make it work because vtk internally apply already the inverted transform
    transform = vtk.vtkTransform()
    transform.SetMatrix(vtk_matrix)
    transform.Inverse()
 
    # Create the flip matrix. VTK uses its own convention. To orient the head as the standard LPS (X pointing left, Y pointing towards and Z pointing up) we need to swap the x and y axis
    swap_matrix = np.array([
    [0, 1, 0, 0],  # New X = -Y
    [1, 0, 0, 0],  # New Y = -X
    [0, 0, 1, 0],
    [0, 0, 0, 1]  # Z unchanged
    ])

    flip_vtk_matrix = vtk.vtkMatrix4x4()
    for i in range(4):
        for j in range(4):
            flip_vtk_matrix.SetElement(i, j, swap_matrix[i, j])
  
    flip_transform = vtk.vtkTransform()
    flip_transform.SetMatrix(flip_vtk_matrix)

    # Apply the 2 concatenated matrix to ensure correct alignment. This transform will be used later to transform the pose of the head isosurface
    final_transform = vtk.vtkTransform()
    final_transform.Concatenate(flip_transform)
    final_transform.Concatenate(transform)
    
    # Normalize the pixel intensity between 0-1 to ensure easy interpretability even acroos different patients
    normalize = vtk.vtkImageShiftScale()
    normalize.SetInputData(image_data)
    normalize.SetShift(-image_data.GetScalarRange()[0])  
    range_span = image_data.GetScalarRange()[1] - image_data.GetScalarRange()[0]
    normalize.SetScale(1.0 / range_span)  
    normalize.SetOutputScalarTypeToFloat()
    normalize.Update()


    # Tresholding to extract everyting that is different from background (0). We extract all the head with the internal structures
    threshold = vtk.vtkImageThreshold()
    threshold.SetInputData(normalize.GetOutput())
    threshold.ThresholdByUpper(0.1)
    threshold.ReplaceInOn()
    threshold.SetInValue(1)
    threshold.ReplaceOutOn()
    threshold.SetOutValue(0)
    threshold.Update()
    
    # closing algorithm to fill gaps and holes form the tresholding
    closing = vtk.vtkImageOpenClose3D()
    closing.SetInputData(threshold.GetOutput())  
    closing.SetOpenValue(0)       
    closing.SetCloseValue(1)          
    closing.SetKernelSize(8,8,8) #kernel size can be increased or diminished to fill bigger/smaller holes respectively
    closing.Update()


    # Extracting the isosurface
    mesh = MarchingCubes(closing.GetOutput(), 0.5)
    
    # Removing all the isolated components to keep just the head
    connectivity = vtk.vtkPolyDataConnectivityFilter()
    connectivity.SetInputData(mesh)
    connectivity.SetExtractionModeToLargestRegion()
    connectivity.Update()
    
    # NOT MANDATORY: Applying this we can close the hole of the neck or not
    filling = vtk.vtkFillHolesFilter()
    filling.SetInputData(connectivity.GetOutput())
    filling.SetHoleSize(10000)
    filling.Update()

    # Applying the tranformation matrix computed above to move from ijk to LPS
    transform_filter_final = vtk.vtkTransformPolyDataFilter()
    transform_filter_final.SetInputData(filling.GetOutput())
    transform_filter_final.SetTransform(final_transform)
    transform_filter_final.Update()

    # 5. Wrap in pyvista for easy visualization
    mesh = pv.wrap(transform_filter_final.GetOutput())
    mesh = mesh.triangulate()
    mesh.flip_normals() # normals needed to be flipped because they were pointing inside the head
    ######### Plotter: for visualization purposes only #################################
    pl = pv.Plotter()
    pl.add_mesh(mesh, color="peachpuff")
    pl.add_axes_at_origin(x_color='red', y_color='green', z_color='blue')
    pl.show_axes()
    pl.show()
    ###################################################################################
    
    # Save the mesh as a .stl
    mesh.save("head2.stl")

   
























