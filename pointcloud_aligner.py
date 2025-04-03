import numpy as np
import open3d as o3d
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation as Rot
from scipy.optimize import minimize, approx_fprime


class PointCloudAligner:
    def __init__(self, tolerance=1.1, max_iterations=100, alpha=5):
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.alpha = alpha

   
    @staticmethod
    def find_closest_points(floating, reference):
        tree = KDTree(reference)
        distances, indices = tree.query(floating, workers=-1)
        closest_points = reference[indices]
        return closest_points, distances
 

    @staticmethod
    def compute_rms(source_points, target_points):
        squared_distances = np.linalg.norm((target_points - source_points), axis=1)
        sum_distances = np.sum(squared_distances**2)
        rms = np.sqrt(sum_distances / len(squared_distances))
        return rms

    def perturbation(self,T_init, T_temp, alpha):
        R_init = T_init[:3, :3]
        
        R_temp = T_temp[:3, :3]
        t_temp = T_temp[:3, 3]

        r_init = Rot.from_matrix(R_init).as_euler('xyz')
        r_temp = Rot.from_matrix(R_temp).as_euler('xyz')  
        r_p = r_temp + alpha * (r_init - r_temp)
        R_p = Rot.from_euler('xyz', r_p).as_matrix()

        t_p = t_temp

        T_pert = np.eye(4)
        T_pert[:3, :3] = R_p
        T_pert[:3, 3] = t_p

        return T_pert

    @staticmethod
    def objective_function(params, source_points, target_points):
        angles = params[:3]
        translation = params[3:]
        rotation_matrix = Rot.from_euler('xyz', angles).as_matrix()
        transformed_points = (rotation_matrix @ source_points.T).T + translation
        squared_distances = np.linalg.norm((target_points - transformed_points), axis=1)
        sum_distances = np.sum(squared_distances**2)
        rms = np.sqrt(sum_distances / len(squared_distances))
        return rms

    @staticmethod
    def gradient(params, source_points, target_points):
        eps = 1e-6
        grad = approx_fprime(params, PointCloudAligner.objective_function, eps, source_points, target_points)
        return grad

    def align(self, source_points, target_points):

        # Step 1: Initial alignment using optimization
        params_0 = np.zeros((6,))
        T_init = np.eye(4)
        
        centroid = np.mean(source_points,axis=0)    
        source_points=(source_points-centroid)
        T_0=np.eye(4)
        T_0[:3,3]=-centroid
        
        
        closest_points, distances = self.find_closest_points(source_points, target_points)
    

        result = minimize(
            fun=self.objective_function,
            x0=params_0,
            args=(source_points, closest_points),
            method='BFGS',
            jac=self.gradient,
            options={'disp': False, 'smaxiter': 100}
        )

      

        translation = result.x[3:]
        rotation = Rot.from_euler('xyz', result.x[:3]).as_matrix()
        T_init[:3, :3] = rotation
        T_init[:3, 3] = translation

        # Step 2: ICP with perturbation

        initial_points = (rotation @ source_points.T).T + translation
        closest_points, distances = self.find_closest_points(initial_points, target_points)
        initial_rms = self.compute_rms(initial_points, closest_points)
        

        source_cloud = o3d.geometry.PointCloud()
        source_cloud.points = o3d.utility.Vector3dVector(source_points)

        target_cloud = o3d.geometry.PointCloud()
        target_cloud.points = o3d.utility.Vector3dVector(target_points)

      
        best_rms = initial_rms
        T_optimal = T_init
        T_temp=np.eye(4)

        for i in range(self.max_iterations):
            # Perform ICP
            result = o3d.pipelines.registration.registration_icp(
                source_cloud, target_cloud,
                1000,  # Maximum correspondence distance
                T_init,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=500)
            )

            T_temp = result.transformation
            T_temp = np.array(T_temp, copy=True)

            transformed_points = (T_temp[:3, :3] @ source_points.T) + T_temp[:3, 3].reshape(-1, 1)
            transformed_points = transformed_points.T

            closest_points, distances = self.find_closest_points(transformed_points, target_points)
            rms = self.compute_rms(transformed_points, closest_points)

            print("RMS: ", rms)

            if rms < best_rms:
                T_optimal = T_temp
                best_rms = rms

            if best_rms < self.tolerance:
                print(f"Converged at iteration {i}")
                break

            # Apply perturbation
            T_pert = self.perturbation(T_init, T_temp,self.alpha)
            T_init = T_pert
        
        print("best rms:",best_rms)

        T_optimal = T_optimal @ T_0
    
        return T_optimal


