# HOG_for_classification
提取HOG 特征，配合svm 进行分类，数据集是一个花的数据集，包括玫瑰、小雏菊、向日葵三个类别

### 使用
如果只想跑通，那么先运行 training.m,然后运行classify.m即可，没有给出测试误差，给出的是每个图片的类别标注，并图形显示了

值得注意的是：我在训练时采用了把训练集带入的错误率 resubLoss 和10折交叉验证 kfoldLoss 后者实在是太耗费时间了，可以去除。同时，因为sava classifer 会将训练数据集也保存，太过于臃肿，以后会想想怎么减少模型的大小，让模型更方便移植

### 替换成自己数据集
保证训练集里面按类别分成几个文件夹即可，照葫芦画瓢


### HOG思想一览
http://blog.sina.com.cn/s/blog_6a1bf1310102uxhs.html
