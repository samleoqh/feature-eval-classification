# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: #
# Date: 26.10.2017                                                            #
# Author: Qignhui L                                                           #
#                                                                             #
# INF4300 Mandatory term project 2017 part-2                                  #
# Featrue evaluation and classification                                       #
#                                                                             #
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: #
"""
Classification experimentation dataset configuration file
"""

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
              'quadrant-4_q2_img_w31d1_135_angle_mosaic1_train.png'
              # 'quadrant-4_q3_img_w31d1_0_angle_mosaic1_train.png',
              # 'quadrant-4_q3_img_w31d1_135_angle_mosaic1_train.png'
              ]

test1_set1 = ['quadrant-4_q1_img_w31d1_0_angle_mosaic2_test.png',
              'quadrant-4_q1_img_w31d1_135_angle_mosaic2_test.png',
              'quadrant-4_q2_img_w31d1_0_angle_mosaic2_test.png',
              'quadrant-4_q2_img_w31d1_135_angle_mosaic2_test.png'
              # 'quadrant-4_q3_img_w31d1_0_angle_mosaic2_test.png',
              # 'quadrant-4_q3_img_w31d1_135_angle_mosaic2_test.png'
              ]

test2_set1 = ['quadrant-4_q1_img_w31d1_0_angle_mosaic3_test.png',
              'quadrant-4_q1_img_w31d1_135_angle_mosaic3_test.png',
              'quadrant-4_q2_img_w31d1_0_angle_mosaic3_test.png',
              'quadrant-4_q2_img_w31d1_135_angle_mosaic3_test.png'
              # 'quadrant-4_q3_img_w31d1_0_angle_mosaic3_test.png',
              # 'quadrant-4_q3_img_w31d1_135_angle_mosaic3_test.png'
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

experiment3 = {'infor': ['train gaussian classifier with train_set3',
                         'test gaussian classifier with test1_set3',
                         'test gaussian classifier with test2_set3'
                         ],
               'images': ['train_set3_classified_test1.png',
                          'train_set3_classified_test2.png'
                          ],
               'confmx':['train_set3_test1_ConfusionMatrix.png',
                         'train_set3_test2_ConfusionMatrix.png'
                         ],
               'train': train_set3,
               'test1': test1_set3,
               'test2': test2_set3
               }


experiment0 = {'infor': ['train gaussian classifier with train_set0',
                         'test gaussian classifier with test1_set0',
                         'test gaussian classifier with test2_set0'
                         ],
               'images': ['train_set0_classified_test1.png',
                          'train_set0_classified_test2.png'
                          ],
               'confmx':['train_set0_test1_ConfusionMatrix.png',
                         'train_set0_test2_ConfusionMatrix.png'
                         ],
               'train': train_set0,
               'test1': test1_set0,
               'test2': test2_set0
               }

experiment1 = {'infor': ['train gaussian classifier with train_set1',
                         'test gaussian classifier with test1_set1',
                         'test gaussian classifier with test2_set1'
                         ],
               'images': ['train_set1_classified_test1.png',
                          'train_set1_classified_test2.png'
                          ],
               'confmx':['train_set1_test1_ConfusionMatrix.png',
                         'train_set1_test2_ConfusionMatrix.png'
                         ],
               'train': train_set1,
               'test1': test1_set1,
               'test2': test2_set1
               }

experiment2 = {'infor': ['train gaussian classifier with train_set2',
                         'test gaussian classifier with test1_set2',
                         'test gaussian classifier with test2_set2'
                         ],
               'images': ['train_set2_classified_test1.png',
                          'train_set2_classified_test2.png'
                          ],
               'confmx':['train_set2_test1_ConfusionMatrix.png',
                         'train_set2_test2_ConfusionMatrix.png'
                         ],
               'train': train_set2,
               'test1': test1_set2,
               'test2': test2_set2
               }

experiment4 = {'infor': ['train gaussian classifier with train_set1_3',
                         'test gaussian classifier with test1_set1_3',
                         'test gaussian classifier with test2_set1_3'
                         ],
               'images': ['train_set1_3_classified_test1.png',
                          'train_set1_3_classified_test2.png'
                          ],
               'confmx':['train_set1_3_test1_ConfusionMatrix.png',
                         'train_set1_3_test2_ConfusionMatrix.png'
                         ],
               'train': train_set1+train_set3,
               'test1': test1_set1+test1_set3,
               'test2': test2_set1+test2_set3
               }

experiment5 = {'infor': ['train gaussian classifier with train_set1_2',
                         'test gaussian classifier with test1_set1_2',
                         'test gaussian classifier with test2_set1_2'
                         ],
               'images': ['train_set1_2_classified_test1.png',
                          'train_set1_2_classified_test2.png'
                          ],
               'confmx':['train_set1_2_test1_ConfusionMatrix.png',
                         'train_set1_2_test2_ConfusionMatrix.png'
                         ],
               'train': train_set1+train_set2,
               'test1': test1_set1+test1_set2,
               'test2': test2_set1+test2_set2
               }

experiment6 = {'infor': ['train gaussian classifier with train_set0_1',
                         'test gaussian classifier with test1_set0_1',
                         'test gaussian classifier with test2_set0_1'
                         ],
               'images': ['train_set0_1_classified_test1.png',
                          'train_set0_1_classified_test2.png'
                          ],
               'confmx':['train_set0_1_test1_ConfusionMatrix.png',
                         'train_set0_1_test2_ConfusionMatrix.png'
                         ],
               'train': train_set0+train_set1,
               'test1': test1_set0+test1_set1,
               'test2': test2_set0+test2_set1
               }