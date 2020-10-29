# pancreas-segmentation
我们对NIH胰腺数据集82个CT病例进行操作。
针对于MSD数据集可以采用同样的操作.
我们将数据集作了预处理：将数据集分为四份进行四则交叉验证，将CT值设到[-100,240]HU,并进行归一化处理。
MAD-UNet 参考了:Pancreas Segmentation in Abdominal CT Scan: A Coarse-to-Fine Approach 2016.  对这篇论文的所有作者表示感谢.
使用步骤:
1. proces-data  用于数据增强
2. segmen-metrics 评价指标代码
3. slice  切片处理
4. train  该文件夹包含了MAD-UNet,U-Net,Segnet,Attention unet, FCN,等网络结构
5. utils  相关函数配置

如果代码运行过程中遇到问题,请联系qq邮箱:675507239@qq.com

