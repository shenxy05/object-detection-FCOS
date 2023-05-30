## 目标检测 VOC数据集 FCOS相关代码



### Dependencies

- cuda==11.0
- matplotlib==3.5.3
- numpy==1.21.6
- setuptools==65.6.3
- torch==1.7.1
- torchvision==0.8.2
- Pillow==9.5.0



### 训练步骤

1. 准备数据集：将VOC2012数据集下载并解压，将标注文件（`.xml`）和图像文件（`.jpg`）分别放在`VOC2012/Annotations`和`VOC2012/JPEGImages`文件夹下。

   训练集和测试集已提前划分好并将索引文件存放在`VOC2012/ImageSets/Main`中，可以自行替换。

2. 开始训练：训练文件的参数已经设置好，直接运行`python  train_val,py`即可开始训练，同时每训练一个epoch会在测试集上计算两次mAP。

   ```
   python train_val.py  	--epochs 30		# 训练周期数
   						      --batch_size 4	
   						      --n_cpu 16
   						      --n_gpu '0,1'	# 指定GPU
   ```

3. 训练结果：训练过程中每个epoch最后的模型会以`model_{epoch}.pth`的形式保存在`checkpoint`文件夹中，tensorboard事件文件保存在`runs/FCOS`文件夹中，共记录了训练cln_loss、cnt_loss、reg_loss、total_loss、learning_rate、测试mAP、test_loss7个值，并且可视化了训练过程中的`visualize`文件夹中图像的检测结果，可以通过如下命令查看：

   ```
   tensorboard --logdir ./runs/FCOS  --samples_per_plugin=images=100
   ```

### mAP测试

使用VOC数据集评估训练好的模型：在`eval_voc.py`中修改`model_dir`为模型权重文件路径，将`eval_dataset`的`split`参数改为测试集索引txt的文件名，并将txt文件存放在`VOC2012/ImageSets/Main`中，随后直接运行命令：

```
python eval_voc.py
```

### 检测demo

也可以使用训练好的模型直接可视化指定图片的检测结果。

将需要检测的图片放在`test_images`文件夹中，修改`model_dir`为模型权重文件路径，直接运行`detect.py`，得到的检测结果会保存在`out_images`文件夹中。



> 注：训练模型下载见报告中百度网盘链接。



