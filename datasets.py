#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# Author: Q.Liu
# Date: 26.10.2017
# Dataset for mandatory project 2
#
#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
import os, cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import seaborn as sns
import pandas as pd


train_data = ['mosaic1_train.mat',
              'training_mask.mat']

test_data = ['mosaic2_test.mat',
             'mosaic3_test.mat']

angle_oder = ['angle-0',
              'angle-45',
              'angle-90',
              'angle-135'
              ]

# original data used to analyse textures glcm style
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


cnfg_set0 = {
    'group':        ['texture'], # ['quadrant-4', 'quadrant-16','texture'],
    'angle':        ['0','135'],
    'quadrant-4':   [],
    'quadrant-16':  [],
    'texture':      ['contrast', 'homogeneity'],
    'infor':    ['train gaussian classifier with texture features',
                 'test gaussian classifier with texture features',
                 'test gaussian classifier with texture features' ],
    'images':   ['cnfg_set0_classified_test1.png',
                 'cnfg_set0_classified_test2.png' ],
    'confmx':   ['cnfg_set0_eval_results_of_test1.png',
                 'cnfg_set0_eval_results_of_test2.png']
}

cnfg_set1 = {
    'group':        ['quadrant-16'], # ['quadrant-4', 'quadrant-16','texture'],
    'angle':        ['0','135'],
    'quadrant-4':   [],
    'quadrant-16':  ['q2','q3','q7'],
    'texture':      ['contrast', 'homogeneity'],
    'infor':    ['train gaussian classifier with cnfg_set1',
                 'test gaussian classifier with cnfg_set1',
                 'test gaussian classifier with cnfg_set1' ],
    'images':   ['cnfg_set1_classified_test1.png',
                 'cnfg_set1_classified_test2.png' ],
    'confmx':   ['cnfg_set1_eval_results_of_test1.png',
                 'cnfg_set1_eval_results_of_test2.png']
}

cnfg_set2 = {
    'group':        ['quadrant-4', 'quadrant-16'], # ['quadrant-4', 'quadrant-16','texture'],
    'angle':        ['0','135'],
    'quadrant-4':   ['q2'],
    'quadrant-16':  ['q2','q3','q10'],
    'texture':      ['contrast', 'homogeneity'],
    'infor':    ['train gaussian classifier with cnfg_set2',
                 'test gaussian classifier with cnfg_set2',
                 'test gaussian classifier with cnfg_set2' ],
    'images':   ['cnfg_set2_classified_test1.png',
                 'cnfg_set2_classified_test2.png' ],
    'confmx':   ['cnfg_set2_eval_results_of_test1.png',
                 'cnfg_set2_eval_results_of_test2.png']
}

cnfg_set3 = {
    'group':        ['quadrant-4', 'quadrant-16'],
    'angle':        ['0','135'],
    'quadrant-4':   ['q1'],
    'quadrant-16':  ['q3','q7','q10'],
    'texture':      ['contrast', 'homogeneity'],
    'infor':    ['train gaussian classifier with cnfg_set3',
                 'test gaussian classifier with cnfg_set3',
                 'test gaussian classifier with cnfg_set3' ],
    'images':   ['cnfg_set3_classified_test1.png',
                 'cnfg_set3_classified_test2.png' ],
    'confmx':   ['cnfg_set3_eval_results_of_test1.png',
                 'cnfg_set3_eval_results_of_test2.png']
}

cnfg_set4 = {
    'group':        ['quadrant-4', 'quadrant-16','texture'],
    'angle':        ['0','135'],
    'quadrant-4':   [],
    'quadrant-16':  ['q10'],
    'texture':      ['contrast', 'homogeneity'],
    'infor':    ['train gaussian classifier with cnfg_set4',
                 'test gaussian classifier with cnfg_set4',
                 'test gaussian classifier with cnfg_set4' ],
    'images':   ['cnfg_set4_classified_test1.png',
                 'cnfg_set4_classified_test2.png' ],
    'confmx':   ['cnfg_set4_eval_results_of_test1.png',
                 'cnfg_set4_eval_results_of_test2.png']
}

cnfg_set5 = {
    'group':        ['quadrant-4', 'quadrant-16'], # ['quadrant-4', 'quadrant-16','texture'],
    'angle':        ['0'],
    'quadrant-4':   ['q2'],
    'quadrant-16':  ['q2','q3','q10'],
    'texture':      ['contrast', 'homogeneity'],
    'infor':    ['train gaussian classifier with cnfg_set5',
                 'test gaussian classifier with cnfg_set5',
                 'test gaussian classifier with cnfg_set5' ],
    'images':   ['cnfg_set5_classified_test1.png',
                 'cnfg_set5_classified_test2.png' ],
    'confmx':   ['cnfg_set5_eval_results_of_test1.png',
                 'cnfg_set5_eval_results_of_test2.png']
}

def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    return dir_name

def collect_data(configure=None):
    train_data = []
    test1_data = []
    test2_data = []
    for gp in configure['group']:
        if gp is not 'texture':
            qdr = gp+'_'
        else:
            qdr = ''
        for agl in configure['angle']:
            for qf in configure[gp]:
                train_data.append("{0}{1}_img_w31d1_{2}_angle_mosaic1_train.png".format(qdr,qf, agl))
                test1_data.append("{0}{1}_img_w31d1_{2}_angle_mosaic2_test.png".format(qdr, qf, agl))
                test2_data.append("{0}{1}_img_w31d1_{2}_angle_mosaic3_test.png".format(qdr, qf, agl))

    return train_data, test1_data, test2_data

# load all training images, stack them up
def train_loader(train_dir='./data/train', datalist=None):
    train_imgs = []
    for name in datalist:
        data = cv2.imread(os.path.join(train_dir, name),
                          cv2.IMREAD_GRAYSCALE)
        train_imgs.append(data)

    mask = readmat(filename=train_data[1])

    return train_imgs, mask


# load all training images, stack them up
def test_loader(test_dir='./data/test', datalist=None):
    test_imgs = []
    for name in datalist:
        data = cv2.imread(os.path.join(test_dir, name),
                          cv2.IMREAD_GRAYSCALE)
        test_imgs.append(data)

    mask = readmat(filename=train_data[1])

    return test_imgs, mask

# read .mat file, return array-like object
def readmat(data_dir='./data',filename=None):
    if filename is not None:
        train_contents = sio.loadmat(os.path.join(data_dir, filename))
    basename = os.path.splitext(os.path.basename(filename))[0]

    return train_contents[basename]

# visualize texture's glcm to select directions that best separate textures
def show_texture_glcm_image(texturelist=None, angles=None,
                            data_dir='./data',output_dir='./output'):

    if angles is None:
        angles = ['0', '45', '90', '135']

    if texturelist is not None:
        texture_keys = texturelist.keys()
        for texture in texture_keys:
            saved_file = "{}_GLCM_Visualizations.png".format(texture)

            fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))
            mathtitle = r'$^\circ$'
            # cmap = plt.get_cmap('jet')
            cmap = plt.get_cmap('YlGnBu')
            plt.suptitle("{0}_GLCM_Visualization".format(texture), fontsize=14)
            i = 0
            for row in axes:
                for ax in row:
                    ax = visual_matrix(ax=ax, matrix=readmat(data_dir,textures[texture][i]),
                                       index=None, columns=None, cmap=cmap, norm=False)
                    ax.set_title("Angle {0}{1} GLCM".format(angles[i], mathtitle),
                                 fontsize=10)
                    i += 1
                    fig.tight_layout()

            plt.subplots_adjust(left=0, wspace=0.1, top=0.9)
            plt.savefig(os.path.join(output_dir, saved_file), dpi=100, bbox_inches='tight')

    plt.show()


def visual_matrix(ax=None, matrix=None, index=None, columns=None, cmap='jet', norm=False):
    if norm:
        matrix = matrix.astype(np.float32)
        # matrix /= np.sum(matrix, axis=1).astype(np.float32)
        np.divide(matrix,np.sum(matrix,axis=1),out=matrix)
        matrix *= 100
        min = 0
        max = 100
    else:
        min = np.min(matrix)
        max = np.max(matrix)

    data = pd.DataFrame(matrix, columns=columns, index=index)
    yticks = data.index
    xticks = data.columns

    if norm:
        ax = sns.heatmap(data, yticklabels=yticks, xticklabels=xticks, square=True, cmap=cmap,
                         vmin=min, vmax=max, linewidths=.1, ax=ax)
    else:
        ax = sns.heatmap(data, yticklabels=yticks, xticklabels=xticks, square=True, cmap=cmap,
                         vmin=min, vmax=max, linewidths=.1, ax=ax)

    ax.set_yticklabels(yticks[:], rotation=0)
    ax.set_xticklabels(xticks[:], rotation=0)

    return ax
