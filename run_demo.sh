#!/bin/bash
# ==============================================================================
# Script: run_demo.sh
# Description:
#   This script demonstrates the extraction of geopatch descriptors using 
#   the chambre_rot dataset. The rectification step is implemented in C++ 
#   (located in the 'geopatch' folder). Rectified patches are saved for 
#   further use in the CNN pipeline (TensorFlow).
# ==============================================================================


#Run GeoPatch mapping (GeoPatch - Step 1)
python3 geopatch/run.py --input /src/SimulationICCV/chambre_rot --output /src/demo_output --mode patch
#Run GeoPatch CNN feature extractor on rectified patches (GeoPatch - Step 2)
python3 matchGeopatchCNN.py --input /src/demo_output/results/chambre_rot