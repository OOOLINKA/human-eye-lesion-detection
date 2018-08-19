from __init__ import *
import cv2

feat_path = './new_data/VBM4D_results_1024_512_128/'
label_path = './new_data/GT_512_128/'

feats = sio.loadmat(feat_path+'P0217531_Macular Cube 512x128_1-5-2010_16-17-50_OS_sn0223_cube_z_VBM4D.mat')
label = cv2.imread(label_path+'P0217531_Macular Cube 512x128_1-5-2010_16-17-50_OS_sn0223_cube_z.bmp',0)
print (label)
print (np.sum(label==255))




