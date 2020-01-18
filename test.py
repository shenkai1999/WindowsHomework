import os

from ReadExcel import writeExcel

import numpy as np

from PIL import Image

import tensorflow as tf

import matplotlib.pyplot as plt

from CNNmodel import deep_CNN

N_CLASSES = 10


log_dir = './CK+_part/'

lists = ['0', '1','2','3','4',"5",'6','7','8','9']


# 从测试集中随机挑选一张图片看测试结果



def Mytest(IMAGE_ARR):
        with tf.Graph().as_default():
            image = tf.cast(IMAGE_ARR, tf.float32)

            image = tf.image.per_image_standardization(image)

            image = tf.reshape(image, [1, 28, 28, 3])

            # print(image.shape)

            p = deep_CNN(image, 1, N_CLASSES) #将需要识别的图片放进神经网络

            logits = tf.nn.softmax(p) #归一化

            x = tf.placeholder(tf.float32, shape=[28, 28, 3])

            saver = tf.train.Saver()

            sess = tf.Session()

            sess.run(tf.global_variables_initializer()) #variable初始化

            ckpt = tf.train.get_checkpoint_state(log_dir)

            if ckpt and ckpt.model_checkpoint_path:
                # print(ckpt.model_checkpoint_path)

                saver.restore(sess, ckpt.model_checkpoint_path)

                # 调用saver.restore()函数，加载训练好的网络模型

               # print('Loading success')

            prediction = sess.run(logits, feed_dict={x: IMAGE_ARR})

            max_index = np.argmax(prediction)

            #print('预测的标签为：', max_index, lists[max_index])

            #print('预测的结果为：', prediction)
            return lists[max_index]
if __name__ == '__main__':
    result=[]
    for i in range(0,1260):
        i = str(i)
        test_path = "H:/modelcheck/test_image/"+i+".jpg"
        image = Image.open(test_path)
        image = image.resize((28, 28))
        image_arr = np.array(image)
        result.append(Mytest(image_arr))
    #print(result)
    data_path = "H:\\modelcheck/USPS_Classification.xlsx"  # 文件的绝对路径
    writeExcel(data_path,result)

    '''
    img_dir = 'E:/IdentifyType/webServer/static/image/test.jpg'
    image = Image.open(img_dir)
    image = image.resize((28, 28))
    image_arr = np.array(image)
    Mytest(image_arr)
    '''