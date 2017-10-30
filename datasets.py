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

texturekey = ['texture1',
              'texture2',
              'texture3',
              'texture4'
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

# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# feature images extracted by different algorithms
# use these different features set to train gaussian classifier
# to compare the performance by related test sets

train_set0 = ['contrast_img_w31d1_0_angle_mosaic1_train.png',
              'contrast_img_w31d1_135_angle_mosaic1_train.png',
              'homogeneity_img_w31d1_0_angle_mosaic1_train.png',
              'homogeneity_img_w31d1_135_angle_mosaic1_train.png'
              ]

test1_set0 = ['contrast_img_w31d1_0_angle_mosaic2_test.png',
              'contrast_img_w31d1_135_angle_mosaic2_test.png',
              'homogeneity_img_w31d1_0_angle_mosaic2_test.png',
              'homogeneity_img_w31d1_135_angle_mosaic2_test.png'
              ]

test2_set0 = ['contrast_img_w31d1_0_angle_mosaic3_test.png',
              'contrast_img_w31d1_135_angle_mosaic3_test.png',
              'homogeneity_img_w31d1_0_angle_mosaic3_test.png',
              'homogeneity_img_w31d1_135_angle_mosaic3_test.png'
              ]

train_set1 = ['quadrant-4_q1_img_w31d1_0_angle_mosaic1_train.png',
              'quadrant-4_q1_img_w31d1_135_angle_mosaic1_train.png',
              'quadrant-4_q2_img_w31d1_0_angle_mosaic1_train.png',
              'quadrant-4_q2_img_w31d1_135_angle_mosaic1_train.png',
              'quadrant-4_q3_img_w31d1_0_angle_mosaic1_train.png',
              'quadrant-4_q3_img_w31d1_135_angle_mosaic1_train.png'
              ]

test1_set1 = ['quadrant-4_q1_img_w31d1_0_angle_mosaic2_test.png',
              'quadrant-4_q1_img_w31d1_135_angle_mosaic2_test.png',
              'quadrant-4_q2_img_w31d1_0_angle_mosaic2_test.png',
              'quadrant-4_q2_img_w31d1_135_angle_mosaic2_test.png',
              'quadrant-4_q3_img_w31d1_0_angle_mosaic2_test.png',
              'quadrant-4_q3_img_w31d1_135_angle_mosaic2_test.png'
              ]

test2_set1 = ['quadrant-4_q1_img_w31d1_0_angle_mosaic3_test.png',
              'quadrant-4_q1_img_w31d1_135_angle_mosaic3_test.png',
              'quadrant-4_q2_img_w31d1_0_angle_mosaic3_test.png',
              'quadrant-4_q2_img_w31d1_135_angle_mosaic3_test.png',
              'quadrant-4_q3_img_w31d1_0_angle_mosaic3_test.png',
              'quadrant-4_q3_img_w31d1_135_angle_mosaic3_test.png'
              ]

train_set2 = ['quadrant-16_q1_img_w31d1_0_angle_mosaic1_train.png',
              'quadrant-16_q1_img_w31d1_135_angle_mosaic1_train.png',
              'quadrant-16_q2_img_w31d1_0_angle_mosaic1_train.png',
              'quadrant-16_q2_img_w31d1_135_angle_mosaic1_train.png',
              'quadrant-16_q3_img_w31d1_0_angle_mosaic1_train.png',
              'quadrant-16_q3_img_w31d1_135_angle_mosaic1_train.png',
              'quadrant-16_q4_img_w31d1_0_angle_mosaic1_train.png',
              'quadrant-16_q4_img_w31d1_135_angle_mosaic1_train.png'
              ]

test1_set2 = ['quadrant-16_q1_img_w31d1_0_angle_mosaic2_test.png',
              'quadrant-16_q1_img_w31d1_135_angle_mosaic2_test.png',
              'quadrant-16_q2_img_w31d1_0_angle_mosaic2_test.png',
              'quadrant-16_q2_img_w31d1_135_angle_mosaic2_test.png',
              'quadrant-16_q3_img_w31d1_0_angle_mosaic2_test.png',
              'quadrant-16_q3_img_w31d1_135_angle_mosaic2_test.png',
              'quadrant-16_q4_img_w31d1_0_angle_mosaic2_test.png',
              'quadrant-16_q4_img_w31d1_135_angle_mosaic2_test.png'
              ]

test2_set2 = ['quadrant-16_q1_img_w31d1_0_angle_mosaic3_test.png',
              'quadrant-16_q1_img_w31d1_135_angle_mosaic3_test.png',
              'quadrant-16_q2_img_w31d1_0_angle_mosaic3_test.png',
              'quadrant-16_q2_img_w31d1_135_angle_mosaic3_test.png',
              'quadrant-16_q3_img_w31d1_0_angle_mosaic3_test.png',
              'quadrant-16_q3_img_w31d1_135_angle_mosaic3_test.png',
              'quadrant-16_q4_img_w31d1_0_angle_mosaic3_test.png',
              'quadrant-16_q4_img_w31d1_135_angle_mosaic3_test.png'
              ]

train_set3 = ['quadrant-16_q6_img_w31d1_0_angle_mosaic1_train.png',
              'quadrant-16_q6_img_w31d1_135_angle_mosaic1_train.png',
              'quadrant-16_q7_img_w31d1_0_angle_mosaic1_train.png',
              'quadrant-16_q7_img_w31d1_135_angle_mosaic1_train.png',
              'quadrant-16_q8_img_w31d1_0_angle_mosaic1_train.png',
              'quadrant-16_q8_img_w31d1_135_angle_mosaic1_train.png',
              'quadrant-16_q10_img_w31d1_0_angle_mosaic1_train.png',
              'quadrant-16_q10_img_w31d1_135_angle_mosaic1_train.png'
              ]

test1_set3 = ['quadrant-16_q6_img_w31d1_0_angle_mosaic2_test.png',
              'quadrant-16_q6_img_w31d1_135_angle_mosaic2_test.png',
              'quadrant-16_q7_img_w31d1_0_angle_mosaic2_test.png',
              'quadrant-16_q7_img_w31d1_135_angle_mosaic2_test.png',
              'quadrant-16_q8_img_w31d1_0_angle_mosaic2_test.png',
              'quadrant-16_q8_img_w31d1_135_angle_mosaic2_test.png',
              'quadrant-16_q10_img_w31d1_0_angle_mosaic2_test.png',
              'quadrant-16_q10_img_w31d1_135_angle_mosaic2_test.png'
              ]

test2_set3 = ['quadrant-16_q6_img_w31d1_0_angle_mosaic3_test.png',
              'quadrant-16_q6_img_w31d1_135_angle_mosaic3_test.png',
              'quadrant-16_q7_img_w31d1_0_angle_mosaic3_test.png',
              'quadrant-16_q7_img_w31d1_135_angle_mosaic3_test.png',
              'quadrant-16_q8_img_w31d1_0_angle_mosaic3_test.png',
              'quadrant-16_q8_img_w31d1_135_angle_mosaic3_test.png',
              'quadrant-16_q10_img_w31d1_0_angle_mosaic3_test.png',
              'quadrant-16_q10_img_w31d1_135_angle_mosaic3_test.png'
              ]

# useful glcm features for comparision with quadrant features
txfeatures = ['contrast', 'homogeneity', 'cluster_shade']

# quadrants = 16
quads16_features = ['q1', 'q2', 'q3', 'q4',
                          'q5', 'q6', 'q7',
                                'q8', 'q9',
                                      'q10'
                    ]
# quadrants = 4
quads4_features = ['q1', 'q2',
                         'q3'
                   ]

# for quadrants = 4, the selected features are
# q1, q2, and q3
# ::::::::::::::::::::::::::::::::::::::::::
# ---------------
# |      |      |
# |  q1  |  q2  |
# |      |      |
# ---------------
# |      |      |
# |      |  q3  |
# |      |      |
# ---------------
#
# for quadrants = 16, the selected features are
# q1, q2, q3, ... q10
#::::::::::::::::::::::::::::::::::::::::::::
# -----------------------------
# |      |      |      |      |
# |  q1  |  q2  |  q3  |  q4  |
# |      |      |      |      |
# -----------------------------
# |      |      |      |      |
# |      |  q5  |  q6  |  q7  |
# |      |      |      |      |
# -----------------------------
# |      |      |      |      |
# |      |      |  q8  |  q9  |
# |      |      |      |      |
# -----------------------------
# |      |      |      |      |
# |      |      |      |  q10 |
# |      |      |      |      |
# -----------------------------

# load all training images, stack them up
def train_loader(datalist=None):
    train_imgs = []
    for name in datalist:
        data = cv2.imread(os.path.join(train_dir, name),
                          cv2.IMREAD_GRAYSCALE)
        train_imgs.append(data)

    mask = readmat(filename=train_data[1])

    return train_imgs, mask


# load all training images, stack them up
def test_loader(datalist=None):
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

# visualize each GLCM for all textures
# def show_texture_glcm_image(texturelist=None, angles=None,
#                             data_dir='./data',output_dir='./output'):
#
#     if angles is None:
#         angles = ['0', '45', '90', '135']
#
#     if texturelist is not None:
#         texture_keys = texturelist.keys()
#         for texture in texture_keys:
#             saved_file = "{}_GLCM_visual.png".format(texture)
#
#             fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 9))
#             mathtitle = r'$^\circ$'
#             cmap = plt.get_cmap('jet')
#             plt.suptitle("{0}_GLCM_imgs".format(texture), fontsize=14)
#             i = 0
#             for row in axes:
#                 for ax in row:
#                     im = ax.imshow(readmat(data_dir,textures[texture][i]), cmap=cmap)
#                     ax.set_title("Angle {0}{1} GLCM".format(angles[i], mathtitle),
#                                  fontsize=10)
#                     fig.colorbar(im, ax=ax)
#                     i += 1
#                     fig.tight_layout()
#
#             plt.subplots_adjust(left=0, wspace=0.1, top=0.9)
#             plt.savefig(os.path.join(output_dir, saved_file), dpi=100, bbox_inches='tight')
#
#     plt.show()

# visualize texture's glcm to select 2 directions that best separate textures
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
