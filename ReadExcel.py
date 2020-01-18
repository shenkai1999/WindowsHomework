import xlrd   #导入xlrd模块
from numpy import *
import operator
import time
import numpy as np
from PIL import Image
from sklearn import preprocessing
from xlutils.copy import copy


def readExcel(path,name):
        data_path = path  # excle表格路径，需传入绝对路径
        sheetname = name  # excle表格内sheet名
        data = xlrd.open_workbook(data_path)  # 打开excel表格
        table = data.sheet_by_name(sheetname)  # 切换到相应sheet
        rowNum = table.nrows  # 获取表格行数
        colNum = table.ncols  # 获取表格列数

        if rowNum<2:
            print("excle内数据行数小于2")
        else:
            L = []                                                 #列表L存放取出的数据
            for i in range(0,rowNum):                         #从第一行（数据行）开始取数据
                sheet_data = []                                   #定义一个列表用来存放对应数据
                for j in range(0,colNum):                       #j对应列值
                    sheet_data.append(table.row_values(i)[j] ) #把第i行第j列的值取出放到列表里

                L.append(sheet_data)  #一行值取完之后，追加到L列表中

            return L

def writeExcel(path,_data):
    data_path = path  # excle表格路径，需传入绝对路径

    data = xlrd.open_workbook(data_path)  # 打开excel表格

    newWb = copy(data)  # 复制
    newWs = newWb.get_sheet(3);  # 取sheet表
    for i in range(0,1260):
        newWs.write(i, 0, int(_data[i]));  # 写入数据

    data_path = "H:\\modelcheck/my_result.xlsx"
    newWb.save(data_path);  # 保存至result路径

def compare(excel1,excel2):
    a=readExcel_label(excel1,"Test Label")

    b=readExcel_label(excel2,"Test Label")
    count=0

    for i in range(0,1260):
        b[i]=int(b[i])
        a[i]=int(a[i])
        if a[i]!=b[i]:
            count=count+1

    print(a)
    print(b)
    print(count)


def readExcel_label(path, name):
    data_path = path  # excle表格路径，需传入绝对路径
    sheetname = name  # excle表格内sheet名
    data = xlrd.open_workbook(data_path)  # 打开excel表格
    table = data.sheet_by_name(sheetname)  # 切换到相应sheet
    keys = table.row_values(0)  # 第一行作为key值
    rowNum = table.nrows  # 获取表格行数
    colNum = table.ncols  # 获取表格列数

    if rowNum < 2:
        print("excle内数据行数小于2")
    else:
        L = []  # 列表L存放取出的数据
        for i in range(0, rowNum):  # 从第一行（数据行）开始取数据
              # 定义一个列表用来存放对应数据
            for j in range(0, colNum):  # j对应列值
                L.append(table.row_values(i)[j])  # 把第i行第j列的值取出放到列表里


        return L

"""
描述：
  KNN算法实现分类器
参数：
  inputPoint：测试集
  dataSet：训练集
  labels：类别标签
  k:K个邻居
返回值：
  该测试数据的类别
"""

def KNN(test_data,train_data,train_label,k):
    #已知分类的数据集（训练集）的行数
    dataSetSize = train_data.shape[0]
    #求所有距离：先tile函数将输入点拓展成与训练集相同维数的矩阵，计算测试样本与每一个训练样本的距离
    all_distances = np.sqrt(np.sum(np.square(tile(test_data,(dataSetSize,1))-train_data),axis=1))
    #print("所有距离：",all_distances)
    #按all_distances中元素进行升序排序后得到其对应索引的列表
    sort_distance_index = all_distances.argsort()
    #print("文件索引排序：",sort_distance_index)
    #选择距离最小的k个点
    classCount = {}
    for i in range(k):
        #返回最小距离的训练集的索引(预测值)
        voteIlabel = train_label[sort_distance_index[i]]
        #print('第',i+1,'次预测值',voteIlabel)
        classCount[voteIlabel] = classCount.get(voteIlabel,0)+1
    #求众数：按classCount字典的第2个元素（即类别出现的次数）从大到小排序
    sortedClassCount = sorted(classCount.items(), key = operator.itemgetter(1), reverse = True)
    return sortedClassCount[0][0]



"""
将Excel表里面读出的向量转化成16*16的矩阵
"""
def ChangeToMat(data):

    t = 16
    juzhen = []
    for j in range(0, 16):
        temp = []
        for i in range(t - 16, t):
            temp.append(data[i])
        juzhen.append(temp)
        t += 16
    NewMat = np.mat(juzhen)
    return NewMat

"""
描述：
  构建训练集数据向量，及对应分类标签向量
参数：
  无
返回值：
  hwLabels：分类标签矩阵
  trainingMat：训练数据集矩阵
"""
def trainingDataSet():
    data_path = "H:\\模式识别数据/USPS_Classification.xlsx"  # 文件的绝对路径
    sheetname = "Train Feature"
    data = readExcel(data_path, sheetname)
    labelsheetname = "Train Label"
    lable = readExcel_label(data_path,labelsheetname)
    hwLabels = []
    m = 480
    # zeros返回全部是0的矩阵，参数是行和列
    trainingMat = zeros((m, 256))  # m维向量的训练集
    for i in range(m):
        # print (i);
        hwLabels.append(lable[i]-1)
        trainingMat[i, :] = data[i]
    return hwLabels, trainingMat

def handwritingTest():
    """
    hwLabels,trainingMat 是标签和训练数据，
    hwLabels 是一个一维矩阵，代表每个文本对应的标签（即文本所代表的数字类型）
    trainingMat是一个多维矩阵，每一行都代表一个文本的数据，每行有1024个数字（0或1）
    """
    data_path = "H:\\模式识别数据/USPS_Classification.xlsx"  # 文件的绝对路径
    sheetname = "Train Feature"
    Labelsheetname = "Train Label"
    TrainLabel = readExcel_label(data_path, Labelsheetname)
    Traindata = readExcel(data_path, sheetname)

    Testdata =[]
    TestLabel = []
    for i in range(480,600):
        Testdata.append(Traindata[i])
        TestLabel.append(TrainLabel[i]-1)


    hwLabels, trainingMat = trainingDataSet()  # 构建训练集
    errorCount = 0.0  # 错误数
    mTest = len(Testdata)  # 测试集总样本数
    t1 = time.time()
    for i in range(mTest):

        vectorUnderTest = Testdata[i] #测试样本数据矩阵
        # 调用knn算法进行测试
        classifierResult = KNN(vectorUnderTest, trainingMat, hwLabels, 4)
        # 打印测试出来的结果和真正的结果，看看是否匹配
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, TestLabel[i]))
        # 如果测试出来的值和原值不相等，errorCount+1
        if (classifierResult != TestLabel[i]):
            errorCount += 1.0
    print("\nthe total number of tests is: %d" % mTest)  # 输出测试总样本数
    print("the total number of errors is: %d" % errorCount)  # 输出测试错误样本数
    print("the total error rate is: %f" % (errorCount / float(mTest)))  # 输出错误率
    t2 = time.time()
    print("Cost time: %.2fmin, %.4fs." % ((t2 - t1) // 60, (t2 - t1) % 60))  # 测试耗时

def f_handwritingTest():

    data_path = "H:\\模式识别数据/USPS_Classification.xlsx"  # 文件的绝对路径
    sheetname = "Test Feature"

    Testdata = readExcel(data_path, sheetname)



    hwLabels, trainingMat = trainingDataSet()  # 构建训练集

    mTest = len(Testdata)  # 测试集总样本数
    t1 = time.time()
    allresult = []
    for i in range(mTest):

        vectorUnderTest = Testdata[i] #测试样本数据矩阵
        # 调用knn算法进行测试
        classifierResult = KNN(vectorUnderTest, trainingMat, hwLabels, 4)
        # 打印测试出来的结果和真正的结果，看看是否匹配
        print("the classifier came back with: %d" % (classifierResult))

        allresult.append(classifierResult)
    t2 = time.time()
    print("Cost time: %.2fmin, %.4fs." % ((t2 - t1) // 60, (t2 - t1) % 60))  # 测试耗时
    return allresult

"""
将向量转为灰度图片并保存
"""
def save_iamge():
    data_path = "H:\\modelcheck/USPS_Classification.xlsx"  # 文件的绝对路径
    Sheetname = "Test Feature"
    data = readExcel(data_path, Sheetname)
    for i in range(0, 1260):
        data1 = data[i]
        pic1 = ChangeToMat(data1)
        min_max_scaler = preprocessing.MinMaxScaler()
        pic2 = min_max_scaler.fit_transform(pic1)  # 将灰度值转到0-1的区间
        pic2 = pic2 * 255
        image = Image.fromarray(pic2)
        if image.mode == "F":
            image = image.convert('RGB')

        a = str(i)
        save_path = "H:\\modelcheck/test_image/" + a + ".jpg"
        image.save(save_path)


if __name__ == '__main__':
    excel1="H:\\modelcheck/my_result.xlsx"
    excel2="H:\\modelcheck/last_result.xlsx"
    compare(excel1,excel2)


"""

    data_path = "H:\\模式识别数据/USPS_Classification.xlsx"  # 文件的绝对路径
    writedata = f_handwritingTest()
    writeExcel(data_path,writedata)
"""