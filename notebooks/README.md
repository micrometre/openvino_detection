# Vehicle Detection And Recognition with OpenVINO™


## Notebook Contents

This notebook uses both a detection model and a classification model from Open Model Zoo. The number and location of vehicles in an image can be analyzed by using vehicle detection. Vehicle attribute recognition can assist in the statistics of vehicle characteristics in traffic analysis scenario. The detection model is used to detect vehicle position, which is then cropped to a single vehicle before it is sent to a classification model to recognize attributes of the vehicle. 

Overview of the pipeline: 
![flowchart](https://user-images.githubusercontent.com/47499836/157867076-9e997781-f9ef-45f6-9a51-b515bbf41048.png)

For more information about the pre-trained models, refer to the [Intel](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/intel) and [public](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public) models documentation from Open Model Zoo.

## Installation Instructions

This is a self-contained example that relies solely on its own code.</br>
We recommend running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).

<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/vehicle-detection-and-recognition/README.md" />





# # Vehicle Detection And Recognition with OpenVINO™
# 
# This tutorial demonstrates how to use two pre-trained models from [Open Model Zoo](https://github.com/openvinotoolkit/open_model_zoo): [vehicle-detection-0200](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/intel/vehicle-detection-0200) for object detection and [vehicle-attributes-recognition-barrier-0039](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/intel/vehicle-attributes-recognition-barrier-0039) for image classification. Using these models, you will detect vehicles from raw images and recognize attributes of detected vehicles.
# ![flowchart](https://user-images.githubusercontent.com/47499836/157867076-9e997781-f9ef-45f6-9a51-b515bbf41048.png)
# 
# As a result, you can get:
# 
# ![result](https://user-images.githubusercontent.com/47499836/157867020-99738b30-62ca-44e2-8d9e-caf13fb724ed.png)
# 
# 
# #### Table of contents:
# 
# - [Imports](#Imports)
# - [Download Models](#Download-Models)
# - [Load Models](#Load-Models)
#     - [Get attributes from model](#Get-attributes-from-model)
#     - [Helper function](#Helper-function)
#     - [Read and display a test image](#Read-and-display-a-test-image)
# - [Use the Detection Model to Detect Vehicles](#Use-the-Detection-Model-to-Detect-Vehicles)
#     - [Detection Processing](#Detection-Processing)
#     - [Recognize vehicle attributes](#Recognize-vehicle-attributes)
#         - [Recognition processing](#Recognition-processing)
#     - [Combine two models](#Combine-two-models)
# 
# 
# ### Installation Instructions
# 
# This is a self-contained example that relies solely on its own code.
# 
# We recommend  running the notebook in a virtual environment. You only need a Jupyter server to start.
# For details, please refer to [Installation Guide](https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/README.md#-installation-guide).
# 
# <img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/vehicle-detection-and-recognition/vehicle-detection-and-recognition.ipynb" />
# 

# ## Imports
# [back to top ⬆️](#Table-of-contents:)
# 
# Import the required modules.

# In[ ]:


#get_ipython().run_line_magic('pip', 'install -q "openvino>=2023.1.0" opencv-python tqdm')

#get_ipython().run_line_magic('pip', 'install -q "matplotlib>=3.4"')
