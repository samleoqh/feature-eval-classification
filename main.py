# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: #
# Date: 26.10.2017                                                            #
# Author: Qignhui L                                                           #
#                                                                             #
# INF4300 Mandatory term project 2017 part-2                                  #
# Featrue evaluation and classification                                       #
#                                                                             #
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: #

# Standard library imports
import os
import sys
import time
from pandas.plotting import radviz
from PIL import Image

# Private libraries
from datasets import *
from features import *
from classifier import *

# all needed folders for this project
output_dir= './output'      # path to save all figures plotted for report
data_dir = './data'         # path to the origin train and text files *.mat
train_folder = 'train'      # folder under data_dir to save all train features extracted from train*.mat files
test_folder = 'test'        # folder under data_dir to save all test features extracted  from test*.mat files

# GLCM directions for extracting features
angles = {
        '0': [0],
        # '45': [np.pi / 4],
        # '90': [np.pi / 2],
        '135': [3 * np.pi / 4]
    }

# useful texture features for comparision with quadrant features
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

def main():

    check_mkdir(output_dir)
    train_dir = check_mkdir(os.path.join(data_dir, train_folder))
    test_dir = check_mkdir(os.path.join(data_dir,test_folder))

    # :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    # plot trained and test images used in this project
    # and split train image into 4 parts/textures just save as png files for report
    view_origi_images()

    print("Visualize glcm matrix")
    visualize_glcm_matrix(texturelist=textures, data_dir=data_dir, output_dir=output_dir)
    plot_quad_features(filename='output/quadrant16_features.csv')
    radviz_quad_features(filename='output/quadrant16_features.csv')

    # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    # extract contrast, homogeneity and cluster shade features for comparision
    # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    txfeature_extract(input_dir=data_dir, img_file=train_data[0], save_dir=train_dir,
                      directions=angles, txfeatures=txfeatures)
    txfeature_extract(input_dir=data_dir, img_file=test_data[0], save_dir=test_dir,
                      directions=angles, txfeatures=txfeatures)
    txfeature_extract(input_dir=data_dir, img_file=test_data[1], save_dir=test_dir,
                      directions=angles, txfeatures=txfeatures)

    # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    # extract 4 quadrant features for train set and test set
    # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    quad4_feature_extract(input_dir=data_dir, img_file=train_data[0], save_dir=train_dir,
                          directions=angles, quadfeatures=quads4_features)
    quad4_feature_extract(input_dir=data_dir, img_file=test_data[0], save_dir=test_dir,
                          directions=angles, quadfeatures=quads4_features)
    quad4_feature_extract(input_dir=data_dir, img_file=test_data[1], save_dir=test_dir,
                          directions=angles, quadfeatures=quads4_features)

    # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    # extract 16 quadrant features for train set and test set
    # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    quad16_feature_extract(input_dir=data_dir, img_file=train_data[0], save_dir=train_dir,
                           directions=angles, quadfeatures=quads16_features)
    quad16_feature_extract(input_dir=data_dir, img_file=test_data[0], save_dir=test_dir,
                           directions=angles, quadfeatures=quads16_features)
    quad16_feature_extract(input_dir=data_dir, img_file=test_data[1], save_dir=test_dir,
                           directions=angles, quadfeatures=quads16_features)

    # # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    # # train and evaluate gaussian classifier with selected different train and test sets
    # # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    classify_experiment(configure=cnfg_set2, train_path=train_dir, test_path=test_dir)
    # classify_experiment(configure=cnfg_set3, train_path=train_dir, test_path=test_dir)
    # plt.show()

    # scores = computer_scores(savefile='scores_matrix.csv')
    # # print(scores)
    # plot_scores(scores, 'scores_visualization.png')
    plt.show()

def view_origi_images():
    # just for viewing original images and save them as png files used in the project report
    train_img = readmat(filename=train_data[0])
    divide_image(img=train_img, savefile='texture.png')
    mask = readmat(filename=train_data[1])
    plot_image(image=train_img, savefile='train_image.png')
    plt.show()
    plot_image(image=mask,savefile='mask_image.png')
    plt.show()
    test1_img = readmat(filename=test_data[0])
    test2_img = readmat(filename=test_data[1])
    plot_image(image=test1_img, savefile='test1_image.png')
    plt.show()
    plot_image(image=test2_img, savefile='test2_image.png')
    plt.show()

def divide_image(img, savefile='texture.png'):
    h, w = img.shape
    height = int(h / 2)
    width = int(w / 2)
    stepsize = width
    p_num = 1
    for x in range(0, h, stepsize):
        px = x
        end_x = x + height
        if end_x > h:
            end_x = h
            px = max(end_x - height, 0)

        for y in range(0, w, stepsize):
            py = y
            end_y = y + width
            if end_y > w:
                end_y = w
                py = max(end_y - width, 0)
            patch = img[px:end_x, py:end_y]
            file_name = get_basename(savefile)+str(p_num)+'.png'
            patch = Image.fromarray(patch.astype(np.uint8))
            patch.save(os.path.join(output_dir, file_name))
            p_num += 1

def plot_quad_features(filename=None):
    df = pd.read_csv(filename, index_col=0)
    df_angle = df.drop(labels=['TEXTURES'], axis=1)
    df_texture = df.drop(labels=['angle'], axis=1)
    # plt.style.use('ggplot')
    plt.style.use(['seaborn-dark']) # ''classic', 'seaborn-dark'
    df_texture = pd.melt(df_texture, "TEXTURES", var_name="Quadrant")
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(11, 6))
    # plt.suptitle("Visualizing quadrant features by swarmplot ", fontsize=12)

    ax1 = sns.swarmplot(x='Quadrant',y='value', hue="TEXTURES", data=df_texture, ax=ax1, cmap='gist_rainbow')
    # ax1 = radviz(df_texture, 'TEXTURES', ax=ax1, colormap='gist_rainbow')
    ax1.set_title('Textures - swarmplot', loc='center', fontsize=10)

    df_angle = pd.melt(df_angle, "angle", var_name="Quadrant")
    # ax2 = radviz(df_angle, 'angle', ax=ax2, colormap='rainbow')
    ax2 = sns.swarmplot(x='Quadrant',y='value',hue="angle", data=df_angle, ax=ax2, cmap='muted')
    ax2.set_title('Angles - swarmplot', loc='center',fontsize=10)
    # andrews_curves(df_angle, 'angle', ax=ax1, colormap='rainbow')
    # andrews_curves(df_texture, 'TEXTURES', ax=ax2, colormap='rainbow')
    fig.tight_layout()
    plt.subplots_adjust(left=0.05, wspace=0.15, top=0.9)
    plt.savefig(os.path.join(output_dir,'Swarmplot_'+get_basename(filename)+'.png'))

def radviz_quad_features(filename=None):
    df = pd.read_csv(filename, index_col=0)
    df_angle = df.drop(labels=['TEXTURES'], axis=1)
    df_texture = df.drop(labels=['angle'], axis=1)
    # plt.style.use('ggplot')
    plt.style.use(['bmh']) # ''classic', 'seaborn-dark'
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(11, 6))
    # plt.suptitle("Visualization of quadrant features using Radviz ", fontsize=12)

    ax1 = radviz(df_texture, 'TEXTURES', ax=ax1, colormap='gist_rainbow')
    ax1.set_title('Textures - radviz', loc='center', fontsize=10)

    ax2 = radviz(df_angle, 'angle', ax=ax2, colormap='rainbow')
    ax2.set_title('Angles - radviz', loc='center',fontsize=10)
    # andrews_curves(df_angle, 'angle', ax=ax1, colormap='rainbow')
    # andrews_curves(df_texture, 'TEXTURES', ax=ax2, colormap='rainbow')
    fig.tight_layout()
    plt.subplots_adjust(left=0.05, wspace=0.15, top=0.9)
    plt.savefig(os.path.join(output_dir,'Radviz_'+get_basename(filename)+'.png'))

    # plt.show()

def plot_scores(scores=None, savefile=None):
    plt.style.use('seaborn-darkgrid') # 'seaborn-darkgrid' 'ggplot'
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
    # plt.suptitle("Visualization of classification scores ", fontsize=12)

    box_color = dict(boxes='DarkBlue', whiskers='DarkGreen',
                     medians='Red', caps='Gray')

    perfromance_scores = scores.drop(labels=['TP', 'FP', 'TN', 'FN', 'dicesim'], axis=0)


    ax1 = perfromance_scores.plot(kind='box', color=box_color, ax=ax1)
    ax1.set_ylabel('Measure values')
    ax2 = perfromance_scores.plot(kind='line', ax=ax2)
    fig.tight_layout()
    plt.subplots_adjust(left=0.06, wspace=0.12, top=0.9)

    if savefile is not None:
        plt.savefig(os.path.join(output_dir, savefile))

    # plt.show()

def compute_scores(conmatrix=None, savefile=None):
    col_idx = ['Texture1', 'Texture2', 'Texture3', 'Texture4']
    row_idx = ['Textures',
               'TP', 'TN', 'FP', 'FN',
               'specificity', 'precision', 'recall', 'f1_score',
               'jaccard', 'dicesim', 'randacc', 'arearoc']

    scores = pd.DataFrame(columns=col_idx, index=row_idx)
    scores.loc['Textures'] = col_idx
    scores.columns = scores.iloc[0]
    scores = scores.reindex(scores.index.drop('Textures'))

    if conmatrix is None:
        # just for debug
        confusion_matrix = [[43832, 1038,   0,           0],
                            [3,     43054,  0,           0],
                            [1328,  0,      39781,       2],
                            [2070,  2303,   1217,    31625]
                            ]
        conmatrix = np.asarray(confusion_matrix)

    col_sum = conmatrix.sum(axis=0) # number of estimated classes
    row_sum = conmatrix.sum(axis=1) # number of true/reference classes
    tol_sum = conmatrix.sum()   # total pixels, a scalar

    M, N = conmatrix.shape
    tp = np.zeros(M, dtype=np.uint)
    tn = np.zeros(M, dtype=np.uint)
    fp = np.zeros(M, dtype=np.uint)
    fn = np.zeros(M, dtype=np.uint)
    for i in range(M):
        for j in range(N):
            if i == j:
                tp[i] = conmatrix[i, j]
            else:
                fp[i] += conmatrix[j,i]
                fn[i] += conmatrix[i,j]

    tn = tol_sum - fp - row_sum

    specificity = tn/(tol_sum-row_sum)
    precision = tp/(tp+fp)              # = tp/col_sum
    recall = tp/(tp+fn)
    f1_score = 2* recall * precision/(recall+precision)
    jaccard = tp / (tp + fp + fn)
    dicesim = 2 * tp / (col_sum + row_sum)
    randacc = (tp + tn) / tol_sum
    arearoc = (tp / row_sum + tn / (tol_sum - row_sum)) / 2

    scores.loc['jaccard'] = np.around(jaccard, 4)
    scores.loc['dicesim'] = np.around(dicesim, 4)
    scores.loc['randacc'] = np.around(randacc, 4)
    scores.loc['arearoc'] = np.around(arearoc, 4)

    scores.loc['TP'] = tp
    scores.loc['TN'] = tn
    scores.loc['FP'] = fp
    scores.loc['FN'] = fn
    scores.loc['specificity'] = np.around(specificity, 4)
    scores.loc['precision'] = np.around(precision, 4)
    scores.loc['recall'] = np.around(recall, 4)
    scores.loc['f1_score'] = np.around(f1_score, 4)

    scores['Average'] = np.around(scores.mean(axis=1), 4)

    if savefile is not None:
        scores.to_csv(os.path.join(output_dir, savefile))
    # print(scores)  # print out for debug
    # scores.loc[['specificity', 'precision', 'recall', 'f1_score']].plot(kind='bar')
    # scores.loc[['jaccard', 'dicesim', 'randacc', 'areaudroc']].plot(kind='bar')
    # scores.loc[['accuray', 'precision', 'recall', 'f1_score']].plot.box()
    # scores.loc[['jaccard', 'dicesim', 'randacc', 'areaudroc']].plot.box()
    # plt.show()
    return scores

def classify_experiment(configure=None, train_path=None, test_path=None):
    trainsets, test1_set, test2_set = collect_data(configure=configure)
    print(configure['infor'][0])
    # clf = train_gaussian_classifier(path=train_path, train_set=configure['train'])
    clf = train_gaussian_classifier(path=train_path, train_set=trainsets)
    labels = ['Texture1', 'Texture2', 'Texture3', 'Texture4']
    # the below code is not necessary, just test the trained classifer by trainset self,
    # almost 100% for all measures, just for fun
    print('examining training performance...')
    confusion_matrix0, classified_img0 = test_classifier(clf=clf, path=train_path, test_set=trainsets)
    print("Evaluating training results - confusion matrix\n", confusion_matrix0)
    score0 = compute_scores(confusion_matrix0, get_basename(configure['confmx'][0].replace("test1", "train")) + '.csv')
    plot_scores(scores=score0, savefile=get_basename(configure['confmx'][0].replace("test1", "train")) + '_score_plot.png')
    figtitle = get_basename(configure['images'][0].replace("test1", "train"))
    plot_image(classified_img0, figtitle, 'rainbow', configure['images'][0].replace("test1", "train"))
    figtitle = get_basename(configure['confmx'][0].replace("test1", "train"))
    visual_confusion_matrix(confusion_matrix0, index=labels, columns=labels,
                            filename=configure['confmx'][0].replace("test1", "train"),
                            title=figtitle)

    # test the trained classifier with 2 test images: test1 and test2 separately.
    # evaluating its performance by confusion matrix and performance matrix
    print(configure['infor'][1])
    confusion_matrix1, classified_img1 = test_classifier(clf=clf, path=test_path, test_set=test1_set)
    print("Evaluation results - confusion matrix\n", confusion_matrix1)
    score1 = compute_scores(confusion_matrix1, get_basename(configure['confmx'][0])+'.csv')
    plot_scores(scores=score1, savefile= get_basename(configure['confmx'][0])+'_score_plot.png')

    print(configure['infor'][2])
    confusion_matrix2, classified_img2 = test_classifier(clf=clf, path=test_path, test_set=test2_set)
    print("Evaluation results - confusion matrix\n", confusion_matrix2)
    score2 = compute_scores(confusion_matrix2, get_basename(configure['confmx'][1]) + '.csv')
    plot_scores(scores=score2, savefile=get_basename(configure['confmx'][1]) + '_score_plot.png')

    # plot classified images
    figtitle = get_basename(configure['images'][0])
    plot_image(classified_img1, figtitle, 'rainbow', configure['images'][0])

    figtitle = get_basename(configure['images'][1])
    plot_image(classified_img2, figtitle, 'rainbow', configure['images'][1])

    # visualize confusion matrix
    figtitle = get_basename(configure['confmx'][0])
    visual_confusion_matrix(confusion_matrix1, index=labels, columns=labels,
                            filename=configure['confmx'][0],
                            title=figtitle)

    figtitle = get_basename(configure['confmx'][1])
    visual_confusion_matrix(confusion_matrix2, index=labels, columns=labels,
                            filename=configure['confmx'][1],
                            title=figtitle)

    return confusion_matrix1, confusion_matrix2

def test_classifier(clf=None, path=None, test_set=None):
    test_imgs, test_mask = test_loader(test_dir=path, datalist=test_set)
    clf.classify(test_imgs)
    classified_img = clf.clf_img
    clf.evaluate(test_mask, classified_img)
    confusion_matrix = clf.confusion_matrix
    return confusion_matrix, classified_img


def train_gaussian_classifier(path=None, train_set = None):
    clf = GaussianClassifier()
    train_imgs, train_label = train_loader(train_dir=path, datalist=train_set)
    clf.train(train_imgs, train_label)
    return clf

def quad16_feature_extract(input_dir=None, img_file='', save_dir='',
                           directions=None, quadfeatures=None, graylevle=16, neighbour=15):

    gray_img = readmat(data_dir=input_dir, filename=img_file)
    winsize = 2 * neighbour + 1  # sliding window size is 31
    offset = 1
    filltype = 'mirror'

    weights = None
    # suffix = get_filename(image_file) + ".png"
    for degree in directions.keys():
        suffix = get_basename(img_file) + ".png"
        print('-----------------------------------------------')
        if degree is 'isotropic':
            print("Start to compute quadrant feature images of {0} with isotropic glcm".format(get_basename(img_file)))
            suffix = 'iso_glcm_' + suffix
            mathtitle = ''
            iso_flag = True
        else:
            print("Start to compute quadrant feature images of {0} with {1} angle glcm".format(get_basename(img_file), degree))
            suffix = "{0}_angle_".format(degree) + suffix
            mathtitle = r'$^\circ$'
            iso_flag = False

        if len(quadfeatures) < 4:
            quad_num = 4
        else:
            quad_num = 16
        # print("Start to computing feature images of {0} with 135 angle direction".format(texture))
        print('----------------')
        start_time = time.time()
        feature_imgs = construct_quad_images(gray_img=gray_img,
                                              win_order=neighbour,
                                              features=quadfeatures,
                                              offsets=[offset],
                                              angles=directions[degree],
                                              fill_type=filltype,
                                              norm=False,
                                              symm=True,
                                              levels=graylevle + 1,
                                              isotropic=iso_flag,
                                              weight=weights,
                                              rescale=True
                                              )
        print("Time of computing quadrant-{0} images win({1}):{2:.1f} s".format(quad_num, winsize, time.time() - start_time))

        for feature_name in quadfeatures:
            savefilename = "quadrant-{0}_{1}_img_w{2}d{3}_{4}".format(quad_num, feature_name, winsize, offset, suffix)
            cv2.imwrite(os.path.join(save_dir, savefilename), feature_imgs[feature_name])

        for div in range(0, len(quadfeatures), int(len(quadfeatures)/2)):
            # visualize each GLCM for a texture
            saved_file = "quadrant-{0}-q{1}_feature_imgs_w{2}d{3}_{4}".format(quad_num, div, winsize, offset, suffix)

            cmap = plt.get_cmap('jet')
            fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 13))
            # mathtitle = r'$^\circ$'
            plt.suptitle("quadrant-{0} feature images of {1} by sliding-W({2}x{2})-offset({3})\n with {4}{5} GLCM"
                         .format(quad_num, get_basename(img_file), winsize, offset, degree, mathtitle),
                         fontsize=12)
            i = 0
            for row in axes:
                for ax in row:
                    if i is 0:
                        im = ax.imshow(gray_img, cmap='gray')
                        ax.set_title("Image-{0}".format(get_basename(img_file)),
                                     fontsize=9)
                        fig.colorbar(im, ax=ax)
                        i += 1
                        fig.tight_layout()
                    elif i <= len(quadfeatures):
                        im = ax.imshow(feature_imgs[quadfeatures[i-1+div]], cmap=cmap)

                        ax.set_title("{0}{1} GLCM {2} image".format(degree, mathtitle, quadfeatures[i-1+div]),
                                     fontsize=9)
                        fig.colorbar(im, ax=ax)
                        i += 1
                        fig.tight_layout()

            plt.subplots_adjust(left=0, wspace=0, top=0.9)
            plt.savefig(os.path.join(output_dir, saved_file), dpi=100, bbox_inches='tight')

    # plt.show()


def quad4_feature_extract(input_dir=None, img_file='', save_dir='',
                          directions=None, quadfeatures=None, graylevle=16, neighbour=15):

    gray_img = readmat(data_dir=input_dir, filename=img_file)
    winsize = 2 * neighbour + 1  # sliding window size is 31
    offset = 1
    filltype = 'mirror'

    weights = None
    # suffix = get_filename(image_file) + ".png"
    for degree in directions.keys():
        suffix = get_basename(img_file) + ".png"
        print('-----------------------------------------------')
        if degree is 'isotropic':
            print("Start to compute quadrant feature images of {0} with isotropic glcm".format(get_basename(img_file)))
            suffix = 'iso_glcm_' + suffix
            mathtitle = ''
            iso_flag = True
        else:
            print("Start to compute quadrant feature images of {0} with {1} angle glcm".format(get_basename(img_file), degree))
            suffix = "{0}_angle_".format(degree) + suffix
            mathtitle = r'$^\circ$'
            iso_flag = False

        if len(quadfeatures) < 4:
            quad_num = 4
        else:
            quad_num = 16
        # print("Start to computing feature images of {0} with 135 angle direction".format(texture))
        print('----------------')
        start_time = time.time()
        feature_imgs = construct_quad_images(gray_img=gray_img,
                                              win_order=neighbour,
                                              features=quadfeatures,
                                              offsets=[offset],
                                              angles=directions[degree],
                                              fill_type=filltype,
                                              norm=False,
                                              symm=True,
                                              levels=graylevle + 1,
                                              isotropic=iso_flag,
                                              weight=weights,
                                              rescale=True
                                              )
        print("Time of computing quadrant-{0} images win({1}):{2:.1f} s".format(quad_num, winsize, time.time() - start_time))

        for feature_name in quadfeatures:
            savefilename = "quadrant-{0}_{1}_img_w{2}d{3}_{4}".format(quad_num, feature_name, winsize, offset, suffix)
            cv2.imwrite(os.path.join(save_dir, savefilename), feature_imgs[feature_name])

        # visualize each GLCM for a texture
        saved_file = "quadrant-{0}_feature_imgs_w{1}d{2}_{3}".format(quad_num, winsize, offset, suffix)

        cmap = plt.get_cmap('jet')
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))
        # mathtitle = r'$^\circ$'
        plt.suptitle("quadrant-{0} feature images of {1} by sliding-W({2}x{2})-offset({3})\n with {4}{5} GLCM"
                     .format(quad_num, get_basename(img_file), winsize, offset, degree, mathtitle),
                     fontsize=14)
        i = 0
        for row in axes:
            for ax in row:
                if i is 0:
                    im = ax.imshow(gray_img, cmap='gray')
                    ax.set_title("Image-{0}".format(get_basename(img_file)),
                                 fontsize=10)
                    fig.colorbar(im, ax=ax)
                    i += 1
                    fig.tight_layout()
                elif i <= len(quadfeatures):
                    im = ax.imshow(feature_imgs[quadfeatures[i - 1]], cmap=cmap)

                    ax.set_title("{0}{1} GLCM {2} image".format(degree, mathtitle, quadfeatures[i - 1]),
                                 fontsize=10)
                    fig.colorbar(im, ax=ax)
                    i += 1
                    fig.tight_layout()

        plt.subplots_adjust(left=0, wspace=0.1, top=0.9)
        plt.savefig(os.path.join(output_dir, saved_file), dpi=100, bbox_inches='tight')

    # plt.show()


def txfeature_extract(input_dir=data_dir, img_file='', save_dir='',
                      directions=None, txfeatures=None, graylevle=16, neighbour=15):

    gray_img = readmat(data_dir=input_dir, filename=img_file)
    winsize = 2 * neighbour + 1  # sliding window size is 31
    offset = 1
    filltype = 'mirror'
    weights = None
    feature_imgs = {}

    for degree in directions.keys():
        suffix = get_basename(img_file) + ".png"
        print('-----------------------------------------------')
        if degree is 'isotropic':
            print("Start to compute feature images of {0} with isotropic glcm".format(get_basename(img_file)))
            suffix = 'iso_glcm_' + suffix
            mathtitle = ''
            iso_flag = True
        else:
            print("Start to compute feature images of {0} with {1} angle glcm".format(get_basename(img_file), degree))
            suffix = "{0}_angle_".format(degree) + suffix
            mathtitle = r'$^\circ$'
            iso_flag = False

        # print("Start to computing feature images of {0} with 135 angle direction".format(texture))
        print('----------------')
        for feature_name in txfeatures:
            start_time = time.time()
            feature_imgs[feature_name] = construct_texture_image(gray_img=gray_img,
                                                                 win_order=neighbour,
                                                                 feature=feature_name,
                                                                 offsets=[offset],
                                                                 angles=directions[degree],
                                                                 fill_type=filltype,
                                                                 norm=False,
                                                                 symm=True,
                                                                 levels=graylevle + 1,
                                                                 isotropic=iso_flag,
                                                                 weight=weights,
                                                                 rescale=True
                                                                 )

            print("Time of computing {0} win({1}):{2:.1f} s".format(feature_name, winsize, time.time() - start_time))
            savefilename = "{0}_img_w{1}d{2}_{3}".format(feature_name, winsize, offset, suffix)
            cv2.imwrite(os.path.join(save_dir, savefilename), feature_imgs[feature_name])

        # visualize each GLCM for a texture
        saved_file = "feature_imgs_w{0}d{1}_{2}".format(winsize, offset, suffix)

        cmap = plt.get_cmap('jet')
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))
        # mathtitle = r'$^\circ$'
        plt.suptitle("Features images of {0} by sliding-W({1}x{1})-offset({2})\n with {3}{4} GLCM"
                     .format(get_basename(img_file), winsize, offset, degree, mathtitle),
                     fontsize=14)
        i = 0
        for row in axes:
            for ax in row:
                if i is 0:
                    im = ax.imshow(gray_img, cmap='gray')
                    ax.set_title("Image-{0}".format(get_basename(img_file)),
                                 fontsize=10)
                    fig.colorbar(im, ax=ax)
                    i += 1
                    fig.tight_layout()
                elif i <= len(txfeatures):
                    im = ax.imshow(feature_imgs[txfeatures[i - 1]], cmap=cmap)

                    ax.set_title("{0}{1} GLCM {2} image".format(degree, mathtitle, txfeatures[i - 1]),
                                 fontsize=10)
                    fig.colorbar(im, ax=ax)
                    i += 1
                    fig.tight_layout()

        plt.subplots_adjust(left=0, wspace=0.1, top=0.9)
        plt.savefig(os.path.join(output_dir, saved_file), dpi=100, bbox_inches='tight')

    # plt.show()

def get_basename(filepath):
    return os.path.splitext(os.path.basename(filepath))[0]

def visual_confusion_matrix(matrix=None, index=None, columns=None,
                            xlabel='Estimated labels', ylabel='True labels', filename=None,
                            title=''):
    min = np.min(matrix)
    max = np.max(matrix)

    norm_matrix = matrix.astype(np.float32)
    # matrix /= np.sum(matrix, axis=1).astype(np.float32)
    np.divide(norm_matrix,np.sum(norm_matrix,axis=1),out=norm_matrix)
    norm_matrix *= 100

    confusion_df = pd.DataFrame(matrix, columns=columns, index=index)
    confusion_df_norm = pd.DataFrame(norm_matrix, columns=columns, index=index)

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(11, 5))
    cmap = plt.get_cmap('YlGnBu')
    # plt.suptitle('Confusion_matrix_Visualization\n'+title+'\n  ', fontsize=12)

    ax1 = sns.heatmap(confusion_df, vmin=min, vmax=max, annot=True, fmt="d", cmap=cmap, ax=ax1)
    ax1.set_title('without normalization', fontsize=10)
    ax1.set(xlabel=xlabel, ylabel=ylabel)
    ax1.set_yticklabels(columns[:], rotation=0)
    ax1.set_xticklabels(index[:], rotation=0)

    fig.tight_layout()

    ax2 = sns.heatmap(confusion_df_norm, vmin=0, vmax=100, annot=True, fmt=".4", cmap=cmap, ax=ax2)
    ax2.set_title('normalized with 0-100%', fontsize=10)
    ax2.set(xlabel=xlabel, ylabel='')
    ax2.set_xticklabels(index[:], rotation=0)
    ax2.set_yticklabels(columns[:], rotation=0)
    # ax2.tick_params(axis='y', which='both', left='off', right='off')
    fig.tight_layout()

    plt.subplots_adjust(left=0.1, wspace=0.2, top=0.85)
    if filename is not None:
        plt.savefig(os.path.join(output_dir, filename), dpi=100, bbox_inches='tight')


# plot function for visualizing and saving images
def plot_image(image, fig_title='',colormap='gray',savefile=None):
    # plt.figure(fig_num)
    plt.style.use('seaborn-white')
    fig, ax1 = plt.subplots(figsize=(6, 6))

    # ax1.set_title(fig_title)
    ax1.imshow(image, cmap=colormap, interpolation='none')
    fig.tight_layout()
    plt.subplots_adjust(left=0, wspace=0, top=0.9)
    # plt.tight_layout()
    if savefile is not None:
        plt.savefig(os.path.join(output_dir, savefile))

    # plt.show()


def quadrant_glcm_analysis( texturelist=None, qnum = 16,
                            data_dir='./data',output_dir='./output'):
    df = pd.DataFrame()
    if texturelist is not None:
        texture_keys = texturelist.keys()
        for texture in texture_keys:
            for idx in range(len(texturelist[texture])):
                glcm = readmat(data_dir=data_dir, filename=texturelist[texture][idx])
                features = get_quad_features(glcm=glcm, quadnum=qnum)
                df1 = pd.DataFrame(features, index=[texture,])
                df1.loc[:, 'angle'] = angle_oder[idx]
                df = df.append(df1)

    df = df.reset_index()
    df = df.rename(columns={"index":"TEXTURES"})
    df = df.sort_values(by=["TEXTURES"], ascending=[True])
    df.to_csv(os.path.join(output_dir, "quadrant{0}_features.csv".format(qnum)))
    return df

def visualize_glcm_matrix(texturelist=None, angles=None,
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
                    ax = heatmap_glcm(ax=ax, matrix=readmat(data_dir,texturelist[texture][i]),
                                       index=None, columns=None, cmap=cmap, norm=False)
                    ax.set_title("Angle {0}{1} GLCM".format(angles[i], mathtitle),
                                 fontsize=10)
                    i += 1
                    fig.tight_layout()

            plt.subplots_adjust(left=0, wspace=0.1, top=0.9)
            plt.savefig(os.path.join(output_dir, saved_file), dpi=100, bbox_inches='tight')

    # plt.show()


def heatmap_glcm(ax=None, matrix=None, index=None, columns=None, cmap='jet', norm=False):
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


if __name__ == '__main__':
    main()
