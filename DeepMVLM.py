import pyvista as pv
import os
import sys

project_root = os.path.abspath("Deep-MVLM")
sys.path.append(project_root)

import argparse
from parse_config import ConfigParser
import deepmvlm
from utils3d import Utils3D

pv.global_theme.allow_empty_mesh = True

# Function to use the Deepl-Learning model to Extract the facial landamrks (73) according to the DTU3D standard. We then extract just the nose_tip (landamrk 46). The github repo of the network can be found here: https://github.com/RasmusRPaulsen/Deep-MVLM
def DL_nose_tip(head):
    parser = argparse.ArgumentParser(description='Deep-MVLM')
    parser.add_argument('-c', '--config', default=r"Deep-MVLM\configs\DTU3D-depth-MRI.json")
    config = ConfigParser(parser)

    dm = deepmvlm.DeepMVLM(config)
    landmarks = dm.predict_one_file(head)
    #dm.visualise_mesh_and_landmarks(head, landmarks)


    return landmarks[45,:]
