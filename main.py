# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: #
# Date: 26.10.2017                                                            #
# Author: Qignhui L                                                           #
#                                                                             #
# INF4300 Mandatory term project 2017 part-2                                  #
# Featrue evaluation and classification                                       #
#                                                                             #
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: #
"""

"""

# Standard library imports
import os
import sys
import time
from datasets import *
from features import *
from classifier import *


def main():
    # GLCM directions selected for computing features
    angles = {
        '0': [0],
        # '45': [np.pi / 4],
        # '90': [np.pi / 2]
        '135': [3 * np.pi / 4]
        # 'isotropic': None
    }

    train_file = train_data[0]  # 'mosaic1_train.mat'
    test_file1 = test_data[0]
    test_file2 = test_data[1]
    mask_file = train_data[1]

    train_img = readmat(data_dir=data_dir, filename=train_file)
    test_img1 = readmat(data_dir=data_dir, filename=test_file1)
    test_img2 = readmat(data_dir=data_dir, filename=test_file2)
    mask_img = readmat(data_dir=data_dir, filename=mask_file)

    orig_stdout = sys.stdout
    f = open(os.path.join(output_dir, 'print_outc.txt'), 'a')
    # sys.stdout = f

    # ::::::::::::::::::::::::::
    # extract texture features for comparision with quadrant features
    # ::::::::::::::::::::::::::
    # txfeature_extract(gray_img=train_img, image_file=train_file, save_dir=train_dir,
    #                   directions=angles, txfeatures=txfeatures)
    # txfeature_extract(gray_img=test_img1, image_file=test_file1, save_dir=test_dir,
    #                   directions=angles, txfeatures=txfeatures)
    # txfeature_extract(gray_img=test_img2, image_file=test_file2, save_dir=test_dir,
    #                   directions=angles, txfeatures=txfeatures)

    # ::::::::::::::::::::::::::
    # extract quadrant features
    # ::::::::::::::::::::::::::
    # quad4_feature_extract(gray_img=train_img, image_file=train_file, save_dir=train_dir,
    #                       directions=angles, quadfeatures=quads4_features)
    # quad4_feature_extract(gray_img=test_img1, image_file=test_file1, save_dir=test_dir,
    #                       directions=angles, quadfeatures=quads4_features)
    # quad4_feature_extract(gray_img=test_img2, image_file=test_file2, save_dir=test_dir,
    #                       directions=angles, quadfeatures=quads4_features)

    # quad16_feature_extract(gray_img=train_img, image_file=train_file, save_dir=train_dir,
    #                        directions=angles, quadfeatures=quads16_features)
    # quad16_feature_extract(gray_img=test_img1, image_file=test_file1, save_dir=test_dir,
    #                        directions=angles, quadfeatures=quads16_features)
    # quad16_feature_extract(gray_img=test_img2, image_file=test_file2, save_dir=test_dir,
    #                        directions=angles, quadfeatures=quads16_features)


    # ::::::::::::::::::::::::::
    # train gaussian classifier with selected different sets of features
    # ::::::::::::::::::::::::::
    clf = GaussianClassifier()
    train_imgs , train_label = train_loader(datalist=train_set3)
    print("start to traing gaussian classifier")
    clf.train(train_imgs, train_label)

    # ::::::::::::::::::::::::::
    # evaluate trained gaussian classifier with test dataset
    # ::::::::::::::::::::::::::
    test1_imgs, test1_mask = test_loader(datalist=test1_set3)
    test2_imgs, test2_mask = test_loader(datalist=test2_set3)

    print("start to test gaussian classifier")
    clf.classify(test1_imgs)
    classified_tst1 = clf.clf_img
    clf.evaluate(test1_mask, classified_tst1)
    confusion_matrix1 = clf.confusion_matrix
    print("train_set3, test1 evaluation results\n", confusion_matrix1)

    clf.classify(test2_imgs)
    classified_tst2 = clf.clf_img
    clf.evaluate(test2_mask, classified_tst2)
    confusion_matrix2 = clf.confusion_matrix
    print("train_set3, test2 evaluation results\n", confusion_matrix2)

    figs = 0
    figs = plot_image(classified_tst1, figs,
                      'Classified image', 'rainbow', 'train_set3_classified_test1.png')
    figs = plot_image(classified_tst2, figs,
                      'Classified image', 'rainbow', 'train_set3_classified_test2.png')

    labels = ['texture 1', 'texture 2', 'texture 3', 'texture 4']
    xlb = "Estimated labels"
    ylb = "True labels"

    figtitle = "train_set3_test1 Confusion Matrix"
    visual_matrix(confusion_matrix1, index=labels, columns=labels,
                  xlabel=xlb, ylabel=ylb, filename='train_set3_test1_ConfusionMatrix.png', title=figtitle, norm=False)

    figtitle = "train_set3_test2 Confusion Matrix"
    visual_matrix(confusion_matrix2, index=labels, columns=labels,
                  xlabel=xlb, ylabel=ylb, filename='train_set3_test2_ConfusionMatrix.png', title=figtitle, norm=False)


    sys.stdout = orig_stdout
    f.close()

def quad16_feature_extract(gray_img=None, image_file='', save_dir=train_dir,
                           directions=None, quadfeatures=None, graylevle=16, neighbour=15):

    winsize = 2 * neighbour + 1  # sliding window size is 31
    offset = 1
    filltype = 'mirror'

    weights = None
    # suffix = get_filename(image_file) + ".png"
    for degree in directions.keys():
        suffix = get_filename(image_file) + ".png"
        print('-----------------------------------------------')
        if degree is 'isotropic':
            print("Start to compute quadrant feature images of {0} with isotropic glcm".format(get_filename(image_file)))
            suffix = 'iso_glcm_' + suffix
            mathtitle = ''
            iso_flag = True
        else:
            print("Start to compute quadrant feature images of {0} with {1} angle glcm".format(get_filename(image_file), degree))
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
                         .format(quad_num, get_filename(image_file), winsize, offset, degree, mathtitle),
                         fontsize=12)
            i = 0
            for row in axes:
                for ax in row:
                    if i is 0:
                        im = ax.imshow(gray_img, cmap='gray')
                        ax.set_title("Image-{0}".format(get_filename(image_file)),
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

    plt.show()


def quad4_feature_extract(gray_img=None, image_file='', save_dir=train_dir,
                         directions=None, quadfeatures=None, graylevle=16, neighbour=15):

    winsize = 2 * neighbour + 1  # sliding window size is 31
    offset = 1
    filltype = 'mirror'

    weights = None
    # suffix = get_filename(image_file) + ".png"
    for degree in directions.keys():
        suffix = get_filename(image_file) + ".png"
        print('-----------------------------------------------')
        if degree is 'isotropic':
            print("Start to compute quadrant feature images of {0} with isotropic glcm".format(get_filename(image_file)))
            suffix = 'iso_glcm_' + suffix
            mathtitle = ''
            iso_flag = True
        else:
            print("Start to compute quadrant feature images of {0} with {1} angle glcm".format(get_filename(image_file), degree))
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
                     .format(quad_num, get_filename(image_file), winsize, offset, degree, mathtitle),
                     fontsize=14)
        i = 0
        for row in axes:
            for ax in row:
                if i is 0:
                    im = ax.imshow(gray_img, cmap='gray')
                    ax.set_title("Image-{0}".format(get_filename(image_file)),
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

    plt.show()


def txfeature_extract(gray_img=None, image_file='', save_dir=train_dir,
                      directions=None, txfeatures=None, graylevle=16, neighbour=15):

    winsize = 2 * neighbour + 1  # sliding window size is 31
    offset = 1
    filltype = 'mirror'
    weights = None
    feature_imgs = {}

    for degree in directions.keys():
        suffix = get_filename(image_file) + ".png"
        print('-----------------------------------------------')
        if degree is 'isotropic':
            print("Start to compute feature images of {0} with isotropic glcm".format(get_filename(image_file)))
            suffix = 'iso_glcm_' + suffix
            mathtitle = ''
            iso_flag = True
        else:
            print("Start to compute feature images of {0} with {1} angle glcm".format(get_filename(image_file), degree))
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
                     .format(get_filename(image_file), winsize, offset, degree, mathtitle),
                     fontsize=14)
        i = 0
        for row in axes:
            for ax in row:
                if i is 0:
                    im = ax.imshow(gray_img, cmap='gray')
                    ax.set_title("Image-{0}".format(get_filename(image_file)),
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

def get_filename(filepath):
    return os.path.splitext(os.path.basename(filepath))[0]

def visual_matrix(matrix=None, index=None, columns=None,
                  xlabel='', ylabel='', filename=None,
                  title='', norm=True):
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

    confusion_df = pd.DataFrame(matrix, columns=columns, index=index)

    if norm:
        ax = sns.heatmap(confusion_df, vmin=min, vmax=max, annot=True, fmt=".4", cmap='YlGnBu')
    else:
        ax = sns.heatmap(confusion_df, vmin=min, vmax=max, annot=True, fmt="d", cmap='YlGnBu')

    ax.set(xlabel=xlabel, ylabel=ylabel)
    ax.set_title(title, fontsize=12)

    if filename is not None:
        plt.savefig(os.path.join(output_dir, filename))

    plt.show()

# plot function for visualizing and saving images
def plot_image(image, fig_num=0, fig_title='',colormap='gray',savefile=None):
    plt.figure(fig_num)
    plt.imshow(image,cmap=colormap, interpolation='none')
    plt.title(fig_title)
    plt.tight_layout()
    if savefile is not None:
        plt.savefig(os.path.join(output_dir, savefile))
    fig_num += 1
    plt.show()
    return fig_num

if __name__ == '__main__':
    main()
