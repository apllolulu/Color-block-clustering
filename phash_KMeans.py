import cv2 
import numpy as np
import time
import os
from itertools import chain 
from tqdm import tqdm
from sklearn.cluster import KMeans
import shutil


def pHash(imgfile):

	#加载并调整图片为32x32灰度图片
	img=cv2.imread(imgfile, cv2.IMREAD_GRAYSCALE)
	img=cv2.resize(img,(32,32),interpolation=cv2.INTER_CUBIC)
 
	#创建二维列表
	h, w = img.shape[:2]
	vis0 = np.zeros((h,w), np.float32)
	vis0[:h,:w] = img       #填充数据
 
	#二维Dct变换
	vis1 = cv2.dct(cv2.dct(vis0))
	vis1.resize(8,8)
 
	#把二维list变成一维list
	img_list = list(chain.from_iterable(vis1))
 
	#计算均值
	avg = sum(img_list)*1./len(img_list)
	avg_list = ['0' if i<avg else '1' for i in img_list]
 
	#得到哈希值
	return ''.join(['%x' % int(''.join(avg_list[x:x+4]),2) for x in range(0,64,4)])

def hammingDist(s1, s2):
    assert len(s1) == len(s2)
    return sum([ch1 != ch2 for ch1, ch2 in zip(s1, s2)])


if __name__ == "__main__":

    path = './cut_data_v2'
    files = os.listdir(path)
    #print(files)
    phash_set = []
    for file in files:
        file_path = os.path.join(path,file)
        #print("file_path:",file_path)
        try:
            phash_value = pHash(file_path)
            #print("len(phash_value):",len(phash_value))
        except:
            continue    
        # 图片重命名
        src = file_path
        dst = os.path.join(os.path.abspath(path), phash_value + '.jpg')
        os.rename(src, dst) 

        phash_set.append(phash_value)

    # 相似hash融合
    h2h = {}
    for i,h1 in enumerate(tqdm(phash_set)):
        for h2 in phash_set[:i]:
            #print("h2:",h2)
            if hammingDist(h1,h2) <= 4:# 6->4
                s1 = str(h1)
                s2 = str(h2)
                if s1 < s2: s1,s2 = s2,s1
                h2h[s1] = s2

    #print("h2h:",h2h)
    #print("h2h.keys():",h2h.keys())

    phash_h2h = h2h.keys()
    # 聚类分析
    phash_ = []
    for phash_value in phash_h2h:
        phash_value = int(phash_value,16)
        phash_.append(phash_value)

    phash_ = np.array(phash_)
    phash_ = phash_.reshape(-1,1)
    estimator = KMeans(n_clusters=5)#构造聚类器
    estimator.fit(phash_)#聚类
    label_pred = estimator.labels_ #获取聚类标签
    centroids = estimator.cluster_centers_ #获取聚类中心
    inertia = estimator.inertia_ # 获取聚类准则的总和
    print("centroids:",centroids)

    centroids_set = []
    for center in centroids:
        #print("center:",center)
        center_hex = hex(int(center))
        center_str = str(center_hex)
        #print("len(center_str[2:]):",len(center_str[2:]))
        centroids_set.append(center_str[2:])
    
    print("centroids_set:",centroids_set)
    # 移动与聚类中心的距离是6个汉明距离内的图片
    des = './cut_data_v2_res'
    ori_path = './cut_data_v2'
    for i,h1 in enumerate(tqdm(phash_set)):
        for h2 in centroids_set:
            flag = 0
            try:
                if hammingDist(h1,h2) <= 6 and flag ==0:
                    file  = str(h1)+'.jpg'
                    path = os.path.join(ori_path,file)
                    shutil.move(path,des)
                    flag =1
            except:
                continue



