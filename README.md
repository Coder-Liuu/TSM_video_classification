# TSM模型进行视频分类
## 所需环境

有GPU就`paddlepaddle-gpu==2.2.1`,无GPU就`paddlepaddle==2.2.1`

详情请看requirements.txt,文件具有一定兼容性.库的近似版本应该也可以.

## 文件下载

暴力数据集(二分类)可以在下面的连接下载(下载完请放入data文件夹下)

https://aistudio.baidu.com/aistudio/datasetdetail/125525

训练所需的预训练权重,可以在下面连接下载(下载完请放入项目的根目录下)

https://videotag.bj.bcebos.com/PaddleVideo-release2.1/TSM/TSM_k400.pdparams

## 训练步骤
1. 数据集的准备
下载暴力视频数据集,下载完请放入data文件夹下

```
数据集格式如下所示:
├── data
│   └── Violence
│       ├── V_1.mp4
│       ├── V_2.mp4
│   └── NonViolence
│       ├── NV_1.mp4
│       ├── NV_2.mp4
```

2. 下载预训练权重
下载完请放入项目的根目录下

3. 数据集的处理
运行`get_annotation.py`文件,获得数据集的索引文件,在annotation下生成.具体生成格式如下所示:
```
data/data/NonViolence/NV_296.mp4 0
data/data/NonViolence/NV_462.mp4 0
data/data/NonViolence/NV_985.mp4 0
```
4. 训练步骤
运行`train.py`即可开始训练(如果没有GPU的话,建议把`setting.py`中的`log_interval`变量改成1,方便我们更自己的观看进度)

训练中会在output中输出许多模型文件.

5. 预测步骤
首先在`predict.py`更改`model_file`变量指定自己的模型文件.

然后运行`predict.py`即可开始预测(默认预测数据的验证集,如果想进行修改可以定义一个类似与`annotation/violence_val_videos.txt`)进行预测.

## 我想直接运行你的代码
请去Ai Studio直接fork,之后便可以运行我的代码了.

项目连接:https://aistudio.baidu.com/aistudio/projectdetail/3415438

## Q & A
Q: 我想开启top5的计算,应该怎么做的

A: 请取消`utils.py 134行`和`model.py 437行`的注释,注释掉`utils.py 92-93行`

Q: 我使用的自己的数据集,为什么出现了下面的错误
```
SystemError: (Fatal) Blocking queue is killed because the data reader raises an exception.
  [Hint: Expected killed_ != true, but received killed_:1 == true:1.] (at /paddle/paddle/fluid/operators/reader/blocking_queue.h:166)
```
A: 系统没有找到你的annoation中视频的文件,这里出现了问题
```
data/data/NonViolence/NV_296.mp4(没有找到) 0
data/data/NonViolence/NV_462.mp4(没有找到) 0
data/data/NonViolence/NV_985.mp4(没有找到) 0
```

Q: 跑着跑着程序被杀死是什么问题

A: 很有可能是你的视频太大,例如(大于10MB),希望减小batch_size,实在不行删掉这个视频吧.