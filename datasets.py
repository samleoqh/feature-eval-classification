#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# Author: Q.Liu
# Date: 26.10.2017
# Dataset for mandatory project 2
#
#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

output_dir = './output'
data_dir = './data'

train_data = ['mosaic1_train.mat',
              'training_mask.mat']

test_data = ['mosaic2_test.mat',
             'mosaic3_test.mat']

textures = {'texture1':['texture1dx1dy0.mat',           # angle 0 degree
                        'texture1dx1dymin1.mat',        # angle 45 degree
                        'texture1dx0dymin1.mat',        # angle 90 degree
                        'texture1dxmin1dymin1.mat'],    # angle 135 degree

            'texture2':['texture2dx1dy0.mat',           # angle 0 degree
                        'texture2dx1dymin1.mat',        # angle 45 degree
                        'texture2dx0dymin1.mat',        # angle 90 degree
                        'texture2dxmin1dymin1.mat'],    # angle 135 degree

            'texture3':['texture3dx1dy0.mat',           # angle 0 degree
                        'texture3dx1dymin1.mat',        # angle 45 degree
                        'texture3dx0dymin1.mat',        # angle 90 degree
                        'texture3dxmin1dymin1.mat'],    # angle 135 degree

            'texture4':['texture4dx1dy0.mat',           # angle 0 degree
                        'texture4dx1dymin1.mat',        # angle 45 degree
                        'texture4dx0dymin1.mat',        # angle 90 degree
                        'texture4dxmin1dymin1.mat']     # angle 135 degree
            }
