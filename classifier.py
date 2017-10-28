# Q.Liu 26.10.2017
# Multivariate gaussian classifier implementation

import numpy as np

class GaussianClassifier(object):
    def __init__(self):
        self.classes = None
        self.num_classes = None
        self.num_features = None
        self.prior = None
        self.class_means = None
        self.class_covmatrix = None
        self.confusion_matrix = None

        self.clf_img = None
        self.posterior_imgs = None


    def train(self, features, mask):
        self.classes = np.unique(mask)[1:]
        self.num_classes = len(self.classes)
        self.num_features = len(features)

        self.prior = np.ones(self.num_classes)/float(self.num_classes)
        self.class_means = np.zeros([self.num_classes,self.num_features])
        self.class_covmatrix = np.zeros([self.num_classes,
                                         self.num_features,
                                         self.num_features])
        for c in range(self.num_classes):
            num_pixels = np.count_nonzero(mask == self.classes[c])
            samples = np.zeros([self.num_features, num_pixels])
            for i in range(self.num_features):
                samples[i,:] = features[i][np.nonzero(mask == self.classes[c])]

            self.class_means[c, :] = np.mean(samples,axis=1)
            self.class_covmatrix[c,:,:] = np.cov(samples)


    def discriminant(self, feature_vector):
        disrims = np.zeros(self.num_classes)
        log_post = np.zeros(self.num_classes)
        for c in range(self.num_classes):
            log_pc = np.log(self.prior[c])
            det_covmatrix = np.linalg.det(self.class_covmatrix[c,:,:])
            log_det = np.log(det_covmatrix)
            mean_diff = feature_vector - self.class_means[c,:]
            inv_cov = np.linalg.inv(self.class_covmatrix[c,:,:])
            gauss_kernel = 1/2*np.dot(mean_diff,np.dot(inv_cov,mean_diff))
            disrims[c] = log_pc - 1/2*log_det - gauss_kernel
            log_post[c] = disrims[c] - self.num_classes/2*np.log(2*np.pi)

        posteriors = np.exp(log_post)
        posteriors /= np.sum(posteriors)

        return disrims, posteriors


    def classify(self,feature_images):
        self.num_features = len(feature_images)
        feature_vector = np.zeros(self.num_features)
        M, N = feature_images[0].shape
        self.clf_img = np.zeros([M, N])
        self.posterior_imgs = np.zeros([self.num_classes, M, N])

        for i in range(M):
            for j in range(N):
                for k in range(self.num_features):
                    feature_vector[k] = feature_images[k][i,j]
                disrims, posteriors = self.discriminant(feature_vector)
                self.clf_img[i,j] = self.classes[disrims.argmax()]

                for c in range(self.num_classes):
                    self.posterior_imgs[c, i, j] = posteriors[c]


    def evaluate(self, eva_mask=None, classified_image=None):
        M, N = classified_image.shape
        self.confusion_matrix = np.zeros([self.num_classes,self.num_classes],dtype=np.uint16)
        for i in range(M):
            for j in range(N):
                if eva_mask[i,j] == 0:
                    continue
                true_class = eva_mask[i,j]
                estimated_class = classified_image[i,j]
                row_idx = list(self.classes).index(int(true_class))
                col_idx = list(self.classes).index(int(estimated_class))
                self.confusion_matrix[row_idx,col_idx] += 1






