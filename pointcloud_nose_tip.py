import numpy as np
import pyvista as pv
from sklearn.decomposition import PCA

# This function align the pointcloud in the center of its reference system and align the direction of principal axis with the XYZ of the refrence system (always in LPS) to ensure consistency
def pca_align_pointcloud(pointcloud): 

    points = pointcloud.points
    centroid = np.mean(points, axis=0)
    centered = points - centroid

    pca = PCA(n_components=3)
    pca.fit(centered)
    principal_axes = pca.components_  
    
    # Aligning the 3 principal components of the pointcloud with the three axis of the refrence system. 
    R = np.vstack([principal_axes[1], principal_axes[2], principal_axes[0]]).T 
    aligned_points = centered @ R
    
    #Little trick here. Since PCA doesn't fix axis signs, we check the Y direction (the one pointing towards or from the face): if it's mostly positive (i.e., anterior), we flip X and Y to match LPS (posterior)
    y_coords = aligned_points[:, 1]
    positive_fraction = np.sum(y_coords > 0) / len(y_coords)
    if positive_fraction > 0.5:
           flip = np.eye(3)
           flip[0,0]*=-1
           flip[1,1]*=-1
           aligned_points = aligned_points @ flip
           R = R @ flip

    aligned_cloud = pv.PolyData(aligned_points)
     
    return aligned_cloud, R, centroid


# Function to perform nose_tip extraction
def extract_nose_from_pcd(pcd):
        # Recentering and alignment with the referenc esystem in LPS
        recentered_head, R, centroid = pca_align_pointcloud(pcd)

        # Nose extraction. After correct alignment, the nose will be the point in the face pointcloud with the lowest value (y points towards the face from the center of the pointcloud). 
        # To mitigate the impact of outliers or imperfections in the point cloud the nose tip is estimated by averaging the 10 points with the smallest Y-coordinate.
        x_vals = recentered_head.points[:, 1]
        min_x_idx = np.argsort(x_vals)[:10]
        nose_tip = np.mean(recentered_head.points[min_x_idx], axis=0)

        # After extracting the nose_tip we can recompute the coordinate of that point in the original coordinate system and ensure consistency in the following steps
        original_nose_tip = nose_tip @ R.T + centroid
        orig_nose_area = recentered_head.points[min_x_idx] @ R.T + centroid
 
        ######### Plotter: for visualization purposes only #################################
        #pl = pv.Plotter()
        #pl.add_mesh(recentered_head, color="yellow", point_size=5, render_points_as_spheres=True)
        #pl.add_mesh(pv.PolyData(orig_nose_area), color="yellow", point_size=8, render_points_as_spheres=True)
        #pl.add_mesh(pv.PolyData(pcd.points), color="peachpuff", point_size=2, render_points_as_spheres=True)
        #pl.add_mesh(pv.Sphere(center=original_nose_tip, radius=3), color="red")
        #pl.add_axes_at_origin(x_color='red', y_color='green', z_color='blue')
        #pl.show()
        ###################################################################################


        return original_nose_tip

        



        
        