import os

import numpy as np

import tensorflow as tf

from numpy import *
from ReadExcel import readExcel_label
zero_image = []
zero_label = []
one_image = []
one_label=[]
two_image =[]
two_label=[]
three_image=[]
three_label=[]
four_image=[]
four_label=[]
five_image=[]
five_label=[]
six_image=[]
six_label=[]
seven_image=[]
seven_label=[]
eight_image=[]
eight_label=[]
nine_image=[]
nine_label=[]
ratio = 0.2
def get_file(fir_dir):
    data_path = "H:\\modelcheck/USPS_Classification.xlsx"
    temp=readExcel_label(data_path,"Train Label")
    all_train_label = []
    for i in range(0,600):
        temp[i]=int(temp[i])
        all_train_label.append(temp[i]-1)
    a = 0
    filenames = os.listdir(fir_dir)
    filenames.sort(key=lambda x: int(x[:-4]))

    for i in range(0,600):

        if all_train_label[a]==0:
            zero_image.append(fir_dir+"/"+filenames[i])
            zero_label.append(0)
        if all_train_label[a]==1:
            one_image.append(fir_dir+"/"+filenames[i])
            one_label.append(1)
        if all_train_label[a]==2:
            two_image.append(fir_dir+"/"+filenames[i])
            two_label.append(2)
        if all_train_label[a]==3:
            three_image.append(fir_dir+"/"+filenames[i])
            three_label.append(3)
        if all_train_label[a]==4:
            four_image.append(fir_dir+"/"+filenames[i])
            four_label.append(4)
        if all_train_label[a]==5:
            five_image.append(fir_dir+"/"+filenames[i])
            five_label.append(5)
        if all_train_label[a]==6:
            six_image.append(fir_dir+"/"+filenames[i])
            six_label.append(6)
        if all_train_label[a]==7:
            seven_image.append(fir_dir+"/"+filenames[i])
            seven_label.append(7)
        if all_train_label[a]==8:
            eight_image.append(fir_dir+"/"+filenames[i])
            eight_label.append(8)
        if all_train_label[a]==9:
            nine_image.append(fir_dir+"/"+filenames[i])
            nine_label.append(9)
        a = a+1




    image_list = np.hstack((zero_image, one_image, two_image, three_image, four_image, five_image, six_image,
                            seven_image, eight_image, nine_image))
    label_list = np.hstack((zero_label, one_label, two_label, three_label, four_label, five_label, six_label,
                            seven_label, eight_label, nine_label))

    temp = np.array([image_list, label_list])
    temp = temp.transpose()
    np.random.shuffle(temp)  # 打乱顺序
    all_image_list = list(temp[:, 0])  # 取出第0列数据，即图片路径
    all_label_list = list(temp[:, 1])  # 取出第1列数据，即图片标签

    n_sample = len(all_label_list)
    n_val = int(math.ceil(n_sample * ratio))  # 验证集样本数
    n_train = n_sample - n_val  # 训练样本数
    tra_images = all_image_list[0:n_train]
    tra_labels = all_label_list[0:n_train]
    tra_labels = [int(float(i)) for i in tra_labels]
    val_images = all_image_list[n_train:-1]
    val_labels = all_label_list[n_train:-1]
    val_labels = [int(float(i)) for i in val_labels]
    return tra_images, tra_labels, val_images, val_labels


def get_batch(image, label, image_W, image_H, batch_size, capacity):
    # step1：将上面生成的List传入get_batch() ，转换类型，产生一个输入队列queue

    # tf.cast()用来做类型转换

    image = tf.cast(image, tf.string)  # 可变长度的字节数组.每一个张量元素都是一个字节数组

    label = tf.cast(label, tf.int32)

    # tf.train.slice_input_producer是一个tensor生成器

    # 作用是按照设定，每次从一个tensor列表中按顺序或者随机抽取出一个tensor放入文件名队列。

    input_queue = tf.train.slice_input_producer([image, label])

    label = input_queue[1]

    image_contents = tf.io.read_file(input_queue[0])  # tf.read_file()从队列中读取图像

    # step2：将图像解码，使用相同类型的图像

    image = tf.image.decode_jpeg(image_contents, channels=3)

    # jpeg或者jpg格式都用decode_jpeg函数，其他格式可以去查看官方文档

    # step3：数据预处理，对图像进行旋转、缩放、裁剪、归一化等操作，让计算出的模型更健壮。

    image = tf.image.resize_images (image, [image_W,image_H],method=0)

    # 对resize后的图片进行标准化处理

    image = tf.image.per_image_standardization(image)

    # step4：生成batch

    # image_batch: 4D tensor [batch_size, width, height, 3], dtype = tf.float32

    # label_batch: 1D tensor [batch_size], dtype = tf.int32

    image_batch, label_batch = tf.train.batch([image, label], batch_size=batch_size, num_threads=16, capacity=capacity)

    # 重新排列label，行数为[batch_size]

    label_batch = tf.reshape(label_batch, [batch_size])

    #image_batch = tf.cast(image_batch, tf.uint8)    # 显示彩色图像

    image_batch = tf.cast(image_batch, tf.float32)  # 显示灰度图

    # print(label_batch) Tensor("Reshape:0", shape=(6,), dtype=int32)



    return image_batch, label_batch

    # 获取两个batch，两个batch即为传入神经网络的数据


