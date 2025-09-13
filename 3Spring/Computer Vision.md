## Overview

Human Vision System 需要做的事情有：
- visual sensation（感受）
- visual perception（感知，需要处理信息）
- visual motor coordination
而 Computer Vision 可以处理的有：
- Low-Level Vision：处理、提取特征
- Mid-Level Vision：分析局部结构、3D重建
- High-Level Vision：理解
- High-Level Vision：生成
- Vision Language Tasks
- Embodied AI：自动驾驶，具身机器人

## Classic Vision
*这些方法叫做Non-Learning Method。*
### Image
#### Gradient
我们对于每个 pixel(x,y) 用一个函数值 $f(x, y) \in [0, 255]$ 来表示他对应的值（可以有RGB图像和灰度图像）
此时就可以对每一个 pixel 求导 $\nabla f = [\frac{\partial f}{\partial x}, \frac{\partial f}{\partial y}]$.

#### Filter
分为 Linear Filters 和 Non-linear Filters，线性的 filter 可以用卷积来实现，非线性的则是如 threshold 来判定。
### Edge Detection
Edge就是延其方向像素强度显著变化，法线方向几乎没有变化。

#### Criteria of edge detection
首先是一些定义：
- True Positive：正确预测真正的edge
- False Positive：错误预测不存在的edge
- True Negative：正确预测不存在的edge
- False Negative：没有预测存在的edge

$$
Precision=\frac{TP}{TP+FP}, Recall=\frac{TP}{TP+FN}
$$

在image中如果直接用求出来的gradient会存在很大的noise，所以需要用filter来对其进行smooth。比较常见的是使用 Gaussian Filter。

#### Non-Maximum Suppression
对于grids上的每一个点q，求出来gradient一步上的相邻点p和r，如果q点的magnitude比p和q高，就保留q。
> 计算p和r的方法是bilinear interpolation

##### Bilinear interpolation
实际上是根据该点所处的正方形块内对$x$和$y$分别做线性插值求出来的近似值，用形式化的语言描述即：
我们假设左下左上右下右上分别为$Q_{1,1},Q_{1,2},Q_{2,1},Q_{2,2}$，那么
$$
\begin{aligned}
f(R_1)&= \frac{x_2-x}{x_2-x_1}f(Q_{1,1})+\frac{x-x_1}{x_2-x_1}f(Q_{2,1})\\
f(R_2)&= \frac{x_2-x}{x_2-x_1}f(Q_{1,2})+\frac{x-x_1}{x_2-x_1}f(Q_{2,2})\\
f(p) &= \frac{y_2-y}{y_2-y_1}f(R_1) + \frac{y-y_1}{y_2-y_1}f(R_2)
\end{aligned}
$$
##### Approximately NMS
我们直接通过gradient的方向来决定选取当前块周围8个块的值作为近似的bilinear interpolation。

#### Edge Linking
一定保留magnitude比$maxval$更大的pixel，一定删除magnitude比$minval$更小的pixel，这里这两个threshold可以通过0.3/0.1\*average magnitude来计算。

接下来考虑连接这些点，对于一个edge点，考虑他在垂直梯度方向上的相邻点，如果magnitude比minval大，那么就可以把这他们加入到edge点集中，

我们使用的是Canny Edge Detector，不同的高斯函数对应的$\sigma$会有不同的效果，$\sigma$比较大会让edge比较blur，但是能减少噪声；$\sigma$比较小则会让edge探测中的噪声比较多

### Line Fitting
我们如果想要找到image之中的一条直线，直接使用edge detection效果并不一定好，因为：
- 存在遮挡
- 不一定是直线
- 存在很多条线
这个时候就需要用到数学方法计算，也就是最小二乘法。
#### Least Square Method
直线的方程为$ax+by+d=0$,那么距离就是$dis=\frac{|ax_i+by_i+d|}{\sqrt{a^2+b^2}}$.
所以$\sum dis^2$可以近似于$\sum (ax_i+by_i+d)^2$.  写成矩阵的形式就是
$$
A=\begin{bmatrix}x_1&y_1&1\\&\cdots\\x_n&y_n&1\end{bmatrix},h=\begin{bmatrix}a\\b\\d\end{bmatrix}
$$
$$
\sum dis^2=||Ah||
$$
此时我们为了避免零解的出现，约束$||h||=1$.

接下来采用SVD分解来解决这个问题：
$$
\begin{aligned}
A&=UDV^T \\
Ah&=UDV^Th\\
V&=\begin{bmatrix}c_1&c_2&c_3\end{bmatrix},h=\alpha_1c_1+\alpha_2c_2+\alpha_3c_3.\\
V^Th&=\begin{bmatrix}\alpha_1\\\alpha_2\\\alpha_3\end{bmatrix}\\
||Ah||&=(\alpha_1\lambda_1)^2 + (\alpha_2\lambda_2)^2 + (\alpha_3\lambda_3)^2
\end{aligned}
$$
此时我们发现最好令$h=c_3$。

#### RANSAC
SVD分解计算方法对于outlier很敏感，所以需要ransac来降低外点的影响。

1. 选取若干个groups，每个group里有构建平面的最少点数
2. 计算平面方程和threshold对应的inlier数。
3. 找到最多的inlier对应的所有点，再通过SVD计算答案。

需要采样多少groups呢？
> fraction of inliers: w
> points of defining hypothesis: n
> groups count: k

Least Success Ratio: $1-(1-w^n)^k$.

#### Hough Transform
另一种Voting的方式，用来处理outlier比较多的情况。
我们从 original space 转换到 parameter space，这个时候就从可以通过最hot的parameter区域找到我们需要的答案。
但是鲁棒性不强。

### Keypoint Detector
Keypoint 就是image中的关键点，而 keypoint 的特征有：
- Repeatability: detect the same point independently in both images
- Saliency: interesting points
- Accurate localization
- Quantity: sufficient number

这里我们以Corner举例，则在这个区域的image gradient有两个比较显著的方向。
#### Harris Corner
对于两个windows $(x_0,y_0)$ 和 $(x_0+u,y_0+v)$，这两个图像存在差异$E=\sum_{x,y\in N}[I(x+u,y+v)-I(x,y)]^2$.
$$
\begin{aligned}
E&=\sum_{x,y\in N}[I(x+u,y+v)-I(x,y)]^2 \\
&=\sum_{x,y}w'_{x_0,y_0}(x,y)D_{u,v}(x,y)\\
&=\sum_{x,y}w(x-x_0,y-y_0)D_{u,v}(x,y)\\
&=\sum_{x,y}w(x_0-x,y_0-y)D_{u,v}(x,y)\\
&=w*D_{u,v}
\end{aligned}
$$
接下来对$D$做近似，有：
$$
\begin{aligned}
I[x+u,y+v]-I[x,y]&\approx I_xu+I_yv\\
D_{u,v}(x,y)&=(I_xu+I_yv)^2\\
&=\begin{bmatrix}u &v\end{bmatrix}\begin{bmatrix}I_x^2&I_xI_y\\I_xI_y&I_y^2\end{bmatrix}\begin{bmatrix}u \\v\end{bmatrix}\\
w*D_{u,v}&=\begin{bmatrix}u &v\end{bmatrix}w*\begin{bmatrix}I_x^2&I_xI_y\\I_xI_y&I_y^2\end{bmatrix}\begin{bmatrix}u \\v\end{bmatrix}\\
M&=w*\begin{bmatrix}I_x^2&I_xI_y\\I_xI_y&I_y^2\end{bmatrix}\\
&=\begin{bmatrix}w*I_x^2&w*I_xI_y\\w*I_xI_y&w*I_y^2\end{bmatrix}\\
&=Q\begin{bmatrix}\lambda_1 & 0\\0&\lambda_2\end{bmatrix}Q^T\\
E=w*D_{u,v}&=\begin{bmatrix}u'&v'\end{bmatrix}\begin{bmatrix}\lambda_1 & 0\\0&\lambda_2\end{bmatrix}\begin{bmatrix}u'\\v'\end{bmatrix}\\
&=\lambda_1u'^2+\lambda_2v'^2
\end{aligned}
$$
此时用一个 approximate 计算：
$\theta = \det(M) - \alpha Tr(M)^2 - t$,其中$\alpha,t$是阈值。$\alpha=\frac{1}{2}(k+\frac{1}{k})^2,t=\frac{b^2}{2}$,$k,b$是关于$\lambda_1,\lambda_2$的阈值。
最后注意在$\theta$的保留上仍然需要引入非最大值抑制，让求出来的值进一步筛选。

#### Equivariance and Invariance：
等变：$F(T(X))=S(F(X))$.
不变:$F(T(X))=F(X)$，属于等变的特殊例子。

高斯Filter具有旋转等变，而rectangle window就没有。

对于 Harris Corner Detector 有，具有旋转和平移等变性，但是在缩放上并非不变的。
## Machine Learning
现在我们诉说的是*Learning Method*。
### Feature
feature 就是用来描述图片的局部情况，具有平移旋转不变性（invariance）。
Model则是基于features来计算 / 预测 我们需要的结果。
### Example
这里以 Digit 为例，来简述该过程。
#### Prepare Data
这里我们用 MNIST - 一个标注好了的数据集 来作为我们的dataset($X\in\mathbb{R}^{28*28},y\in {0,1}$)
#### Build Model
先 flatten 把整个图片变成一个列向量，然后$z=h(x)=\theta^Tx$
用 sigmoid 函数来作为从$[-\infty,\infty]$到$(0,1)$的概率映射，$g(z)=\frac{1}{1+e^{-z}}$
#### Loss Function
我们用极大似然估计来计算Loss函数，考虑
$$
\begin{aligned}
p(y|x;\theta)&=h(\theta)^y(1-h(\theta))^{1-y}\\
p(Y|X;\theta)&=\prod h(\theta)^y(1-h(\theta))^{1-y}\\
\log p(Y|X;\theta)&=\sum y\log h(\theta) + (1-y)\log(1-h(\theta))\\
\mathcal{L}(\theta)&=-\log p(Y|X;\theta) \\
&=-\sum y\log h(\theta) + (1-y)\log(1-h(\theta))
\end{aligned}
$$
#### Training
通过梯度下降(GD)的办法来优化参数：$\theta:=\theta-\alpha\nabla_\theta\mathcal{L}(\theta)$
**Batch Gradient Descent vs. Stochastic Gradient Descent**
Batch Gradient Descent 把所有数据作为训练数据来计算梯度（也就是说一个batch就是all data）
Stochastic 采用 Randomly sample N pairs as a batch from the training data and then compute the average gradient from them, 很明显更快，也可以从局部最优解中逃脱出来。
#### Testing
需要通过test来eval model的表现。

### Multilayer Perceptron
如果只有一个线性层和一个激活层（sigmoid）无法处理不能直接通过线性变换来区分的分类任务，所以引入MLP。
多层线性层，每个线性层后面搭配一个Non-Linear的激活层，这样子的梯度下降需要通过链式法则来计算梯度值。

激活函数比较常用的有：Sigmoid($\frac{1}{1+e^{-x}}$)，tanh($\tanh(x)$)，ReLU($\max(0,x)$)，Leaky ReLU($\max(0.1x,x)$)，Max(直接取$\max$)和ELU($x / \alpha(e^x-1)$)。**表现最好的是ReLU！**

关于神经网络层数的计算不能计算Input Layer（也就是说就是中间层+输出层），FC（Fully Connected Layer）表示全连接，当前层和下一层的所有节点都有连接。
### Convolutional Neural Network(CNN)
我们发现如果直接 Flatten 输入会导致一些局部信息的丢失，所以考虑引入卷积。
对于一个$H*W*C$的图像，用一个$A*B*C$的卷积Filter会把他变成$(H-A+1)*(W-B+1)*1$的新图像，有多少个Filter新的图像的channel数就是多少。*注意卷积完要做ReLU，就像Linear（FC）变化之后也要ReLU一样*
#### Stride & Padding
 > Stride：表示每两个选取Filter之间的距离，每隔多少距离计算一次Conv.
 > Padding: 把整个图像的上下左右边界同时扩展多少。
 
我们设原图像的尺寸为$H*W$，卷积 Filter 的大小为$K*K$，Stride为$S$，Padding为$P$，那么：
新图像的尺寸为$(H+2P-K)/S+1, (W+2P-K)/S+1$。
一般来说我们选取P=1,K=3,S=1。

#### Pooling
池化，将输入规模可以缩小一半，一般有avgpooling、maxpooling和sumpooling。

一般来说，CNN的网络结构为：((CONV+ReLU)\*N+POOLING)\*M -> (FC+ReLU)\*K -> Softmax.

很明显CNN的网络需要的参数量比FC的更少，FC的表达能力比CNN更强，但是由于参数更多更难训练出来。

#### Data preparation
对于一个CNN而言，如果输入都是正/负，考虑在某个Linear层的时候所有的gradient都是大于0的，就会导致学习效率下降。
所以需要normalize，即$x=(x-mean)/std$.

#### Weight Initialization
首先一个错误的思路是把 Weight 初始全部都赋值为0，但是由于所有的weight都一样，最后传递回来的梯度也是一样的，就失去了不对称性。

接下来考虑给所有的 Weight 给一个很小的随机权重，$W=0.01 * rand(D,H)$，这样子mean是0，std=0.01.
*并不是说数字越小越好，对于小权重而言可能会导致backward的值很小减少Gradient Signal。*
但是这样子随着层数的叠加，$x$会变成越来越接近0的分布，导致gradient会都变成0. 如果给权重变大，则可能初始就进入饱和状态，也让梯度没有变化。
我们为了保证$y=x_1w_1+\cdots+x_kw_k$等式左右两端的方差不变，有：$Var(y)=\sum Var(x_i)Var(w_i)$，此时我们可以令$Std(w)=\frac{1}{\sqrt{n}}$.  - **Xavier Initialization**

如果我们把激活函数从tanh变成ReLU，发现还是会让右端的方差减半，此时我们需要改变一下：$Std(w)=\frac{2}{\sqrt{n}}$. - **He Initialization**
#### Optimizer
为了防止陷入鞍点 / 局部最小值，我们可以采用SGD+Momentum的方法，也就是把gradient descent和上一次的结果做一个带权求和。$v_{t+1}=\rho v_t + \nabla f(x_t)$
Adam：可以理解为带Momentum的RMSProp。
$$
\begin{aligned}
m &= beta1*m + (1-beta1)*dx\\
mt &= m / (1-beta1**t)\\
v &= beta2*v + (1-beta2)*(dx^2)\\
vt &= v / (1-beta2^t)\\
x &-= lr * mt / (np.sqrt(vt) + eps)
\end{aligned}
$$

Learning Rate方面，训练的一开始我们需要比较大的Lr，而随着训练的进行就需要降低Lr来保证能够收敛到最低点。
*如果Batch的规模扩大了N倍，那么Lr也要相对应的扩大N倍*
#### UnderFitting
##### BatchNorm
一般插入在FC/Conv和非线性层中间，学习平均值$\beta$和标准差$\gamma$，把input变成一个正态分布的output。
Train ：
$$
\begin{aligned}
\mu_j&=\frac{1}{N}\sum_{i}x_{i,j}\\
\sigma_j&=\frac{1}{N}\sum_{i}(x_{i,j}-\mu_j)^2\\
\hat{x}_{i,j}&=\frac{x_{i,j}-\mu_j}{\sqrt{\sigma_j+\varepsilon}}\\
\hat{y}_{i,j}&=\gamma\hat{x}_{i,j}+\beta
\end{aligned}
$$
Eval:
$$
\begin{aligned}
\mu_{rms}&=\rho\mu_{rms}+(1-\rho)\mu_i\\
\sigma_{rms}&=\rho\sigma_{rms}+(1-\rho)\sigma_i\\
\hat{x}_{i,j}&=\frac{x_{i,j}-\mu_{rms,j}}{\sqrt{\sigma_{rms,j}+\varepsilon}}\\
\hat{y}_{i,j}&=\gamma\hat{x}_{i,j}+\beta
\end{aligned}
$$
此时我们可以把整个CNN的神经网络进一步的强化，变成：((CONV+BN+ReLU)\*N+POOLING)\*M -> (FC+BN+ReLU)\*K -> FC-> Softmax. 注意最后一个Block不能加BN了，因为不需要变成一个正态分布的形式。

**BatchNorm有效的原因不在于减少内部协变量转移，而有可能是平滑Loss Landscape，简化优化函数。**

Cons：训练和测试的目标不一致，可能导致表现差。
注意，Batchnorm是对每个Batch起作用，然后作用范围是C个Channel的H\*W层。

还存在别的norm工具，如Layer Norm，Instance Norm，Group Norm。

##### ResNet
我们发现Deeper的模型表现反而没有Shallower的模型好，因为更难被optimizer。如何让深层网络表现的至少和浅层网络一样好呢？使用残差。
对于一个Residual Block，Y(Output) = X(Input) + F(X). F(X) 的结构可以理解为 Conv+ReLU+Conv 层。
会给Gradient BP增加一个bypass，这种Skip Link能够降低Loss Landscape的Chaotic程度。
#### Overfitting
##### Early Stopping
在 Val 上一旦出现 acc 下降就立即停止，或者记录val上best acc对应的参数模型。
##### Data Augmentation
对于数据层面多收集数据的cost比较大，所以我们采用数据增强的方法
- Position：旋转，平移等
- Color：亮度调整，对比度调整，色彩调整等
可以用来：
- 降低Overfitting
- 增加模型泛化能力
- 解决分类不平衡
但是增强太强了会失去信息，太弱了会无效。
##### Regularization
对于Loss增加一个正则化的代价，防止模型过于复杂。
常用的有L1，L2和Elastic net（L1+L2）

Batchnorm也是正则化的一种，能够限制模型的表达能力（有了BN就可以不用Dropout了）。
##### Dropout
在Train的时候随机丢弃某些层的若干节点，为了防止某些特征的co-adaptive。
*注意：在Train的时候Dropout了要在最后Val的时候Scale回来*

一般来说只有在FC才需要用正则化和Dropout。
### Classification
做分类任务可以有参数和没参数：
- Non-parameteric：KNN（k近邻）
- Parameteric：CNN等
#### K Nearest Neighbor
距离的计算可以用L1 distance（曼哈顿距离），L2 distance（欧几里得距离）来计算，但是这些都对图像的变换非常敏感。
#### CNN
##### Loss Function
这里比较常见的方法是：Softmax + Cross-Entropy Loss。
*也有SVM loss，$L_i=\sum_{j\neq y_i}\max(0,s_j-s_{y_i}+1)$，但是现在已经不常用*。

Softmax函数$\sigma^\beta(z)_i=\frac{e^{\beta z_i}}{\sum_j e^{\beta z_j}}$，一般来说取$\beta=1$,如果$\beta \rightarrow \infty$，Softmax就变成了argmax，对于两个值的Softmax和sigmoid函数等价。

考虑如何计算两个Predict Vector之间的距离，采用Kullback-Leibler divergence。
$$
\begin{aligned}
D_{KL}(P||Q)&=\sum_{x\in \mathcal{X}}P(x)\log (\frac{P(x)}{Q(x)})\\
&=\sum_{x\in \mathcal{X}}P(x)\log P(x) - \sum_{x\in \mathcal{X}}P(x)\log Q(x)\\
&=H(P,Q)-H(P)
\end{aligned}
$$
我们这里引入把P作为ground-truth vector，此时$H(P)$就是一个常量。那么可以把Loss写成交叉熵的形式：
$\mathcal{L}_{CE}=H(P,Q)=-\sum_{x\in\mathcal{X}}P(x)\log Q(x)$

##### Receptive Field
感知域，可以理解为一个点能够与原图多少个临近点相关。
这可以解释为什么CNN的网络结构要用比较小的Filter以及深层网络，为了保证Receptive Field大小差不多，我们可以选择多用几个小Filter的Conv层，而不是直接用一个比较大的Filter的Conv层。
这样子可以减小参数量，例如两个3\*3的表达力等价于一个5\*5的表达力，但是参数:$3*3*2=18 < 5*5=25$。

##### ResNet
ResNet创造性的把Layer层数提升到了152层，同时把分类错误率降低到了5%以内（比人类做得更好）
每进入一个新的Stage（由多个Block组成）就把dimension / 2.
ResNET中把一个3\*3的Conv改成了两个1\*1和一个3\*3的Conv，能够减小参数的开销和内存存储。BottleNeck：对信息做出抽象和提炼，却更加的有效（152层效果好）。这里做的是减小channel数，最后再用1\*1的还原。

Beyond ResNet：
- DenseNet: 增加bypass。
- MobileNets: 在边缘端用有限的算力来进行学习。
- Neural architecture search：通过强化学习学习网络结构。 （NAS）

### Segmentation
Semantic Instance Segmentation： 既要同语义，还要有个体区分。

Granularity：颗粒度，segmentation之间存在颗粒度的差异。

grounding： 定位Segmentation的位置。

- Semantic Segmentation是一个Dense Label Prediction / per-pixel classification problem。
- Classification是一个Global Prediction。

如果单纯的使用Convolution，由于input和output同维，所以中间不能够进行resolution减小，导致开销会变得非常的大。
#### Encoder-Decoder
Auto-Encoder：BottleNeck的模型，用encoder将input维度变小，decoder将它回到output的维度。
要求中间的维度比信息维度更大，不然会出现不可逆转的信息缺失（irreversible info loss）

Upsampling：从小的resolution到大的resolution：
- Unpooling：把大格子内的所有点的值都赋值为小resolution对应的值。
- Maxpooling：只记录一个格中具有最大值，然后把resolution扩大。
- 采用Transposed Convolution，实际上就是把小的矩阵当作权重，给空间中对应的位置加上一个值, 做的是卷积的逆操作。
> 可以采用矩阵的方法来实现，考虑卷积相当于是$\vec{x}*\vec{a}=X\vec{a},\vec{x}*^T\vec{a}=X^T\vec{a}$

Advantage of Bottleneck：
- Lower memory cost
- Larger receptive field and thus better global context
#### UNet
但是比较小的bottleneck中只能存储大致的位置信息，还需要在里面存储`Global context`和`Per-pixel spatial information`
UNet：将原图copy and crop到网络右侧，将得到的bottleneck的信息作为论据来划分原图。
Skip link: 
- Assist final segmentation 
- Avoid memorization

#### Evaluation
如果直接使用pixel-accuracy，你有可能什么都不做就达到比较不错的准确率（如预测蓝天）。
Intersection of Union: $IoU = \frac{\text{target} \land \text{prediction}}{\text{targe} \lor \text{prediction}}$，此时对应的Loss应该修改为：
$$
\begin{aligned}
I(X)&=\sum_{v\in V}X_v*Y_v\\
U(X)&=\sum_{v\in V}(X_v+Y_v-X_v*Y_v)\\
IoU&=\frac{I(X)}{U(X)}\\
L_{IoU}&=1-IoU=1-\frac{I(X)}{U(X)}
\end{aligned}
$$
然后对于每一个类别算一个IoU，把他们取mean作为最终的评测标准 - $mIoU \le 1$。


## 3D Vision

### From 2D to 3D
2D Image Representation: Regular grid: $H\times W\times 3$

2D->3D: 有explicit和implicit的区别。
- Stereo images: 通过 Disparity 来实现3D表达。
- Multiview images: 多视角，如昆虫。
- Panoramic images: 全景图，将多视角缝合在一张图内。

获取视觉的设备：
- RGB camera： RGB图像
- Depth camera： RGB Depth图像
- LiDAR： 点云 Point Cloud

3D Image Representations:
- Regular form:
	- Multi-view images
	- Depth
	- Volumetric
- Irregular form:
	- Surface mesh: 表面网格，图形学用的多
	- Point cloud
	- Implicit representation: $x^2+y^2+z^2=1$

### Camera Model
Pinhole Camera: 小孔(Aperture)成像, PinHole 和 Image Plane 的距离叫做f（焦距），然后存在等式关系：
$$
\begin{aligned}
P(x,y,z)&\rightarrow P'(x',y') \\
x' &= x\frac{f}{z} = f\frac{x}{z}\\
y' &= y\frac{f}{z} = f\frac{y}{z}
\end{aligned}
$$

在这个3维坐标轴内，z代表深度（可以理解为和小孔的距离）。
二维坐标轴内原点是小孔到成像平面的投影点。

问题在于aperture的大小需要非常小，但是随着孔径大小减小，光亮就会减小，这就是trade-off。

现在我们把小孔变成一个透镜 lens，让光线全部通过一个折射投影到相纸平面，对于不同距离的存在不同的对焦距离。

> 椭形畸变(Radial Distortion)：在镜头边缘会出现畸变，靠近棱镜的**近轴折射**则不会
- 完美的棱镜：No distortion
- Pin cushion: 会比实际物品更大一点（外弯）
- Barrel: 比实际物体小（内弯）

相机的参数：
- Intrinsics（内参）：相机制造后就不会改变的参数。
- Extrinsics（外参）：在世界坐标系下的相机位置。

#### Intrinsics
转换需要：
1. Off-set： Pixel的坐标起始点不一定是焦点垂直的对应点。
2. From metric to pixels: 进行单位的转换。

**Transformation:**
$P=(x,y,z)\rightarrow P'=(\alpha\frac{x}{z} + c_x, \beta\frac{y}{z} + c_y)$
其中$(c_x,c_y)$是焦点垂直点的坐标，$\alpha$和$\beta$是相机的内参$\alpha=fk, \beta=fl$。

这样子的一个变化不是一个线性变化(因为要divide z)，采用齐次坐标系可以变成线性变化。
接下来用矩阵表示这个trans：
##### Homogeneous Coordinate System
$E\rightarrow H: (x,y)\rightarrow (x,y,1) | (x,y,z)\rightarrow (x,y,z,1)$
$H\rightarrow E: (x,y,w)\rightarrow (x/w,y/w) | (x,y,z,w)\rightarrow (x/w,y/w,z/w)$

通过齐次坐标可以用矩阵来表示这个transformation:
$$
\begin{aligned}
P'_h = \begin{bmatrix}\alpha x + zc_x\\\beta y + zc_y\\z\end{bmatrix} = \begin{bmatrix}\alpha& 0&c_x&0 \\0&\beta&c_y&0\\0&0&1&0\end{bmatrix}\begin{bmatrix}x\\y\\z\\1\end{bmatrix} = K\begin{bmatrix}I&0\end{bmatrix}P_h
\end{aligned}
$$
> 如果image plane存在一个倾斜角也可以通过矩阵来表示（不考）

$$
\begin{aligned}
P' = \begin{bmatrix}\alpha&-\alpha\cot\theta&c_x&0\\0&\frac{\beta}{\sin\theta}&c_y&0\\0&0&1&0\end{bmatrix}\begin{bmatrix}x\\y\\z\\1\end{bmatrix}
\end{aligned}
$$
这个$K$就是相机的内参，一共有5个自由度。

#### Extrinsics

现在之前所运用的$P=(x,y,z)$都是相机坐标内的，但是我们只能给出世界坐标系下的坐标，存在一个Rotation和Translation的过程：

$$
P_c = \begin{bmatrix}R&T\\0&1\end{bmatrix}\begin{bmatrix}x\\y\\z\\1\end{bmatrix}
$$
然后我们带入整个推导得到：
$$
P_h' = K[I\ \ \ 0]P_h = K\begin{bmatrix}R&T\end{bmatrix}P_w=MP_w
$$
这样就把相机的内参和外参都运用到了坐标转换中。
考虑把$M$矩阵写成$\begin{bmatrix}m_1\\m_2\\m_3\end{bmatrix}$的形式，有：最终image plane上的坐标应该是$(\frac{m_1P_w}{m_3P_w},\frac{m_2P_w}{m_3P_w})$.

Weak Projective Camera：弱投影相机，相机距离物体的距离为常数，可以不需要齐次坐标就能够使用线性变换。
虽然比较的简单，但是失去了近大远小的感觉。最终相机平面上的坐标应该是$(m_1P_w,m_2P_w)$.

Orthographic Projection: 正交投影，不改变长度特征。$x'=x, y'=y$

### Camera Calibration
Goal: 通过多个点得到相机的内参和外参。
已知$P_1,\dots, P_n$在某个参考坐标系下的坐标，以及在对应image上的pixel坐标，要求相机的参数。

考虑相机参数$M$自由度，内参有5个自由度（含倾斜），外参旋转矩阵3个自由度，Translation 3个自由度，一共是$5+3+3=11$个自由度。

那么一个点对$(P_i, p_i)$可以构成两个等式$u_i(m_3P_i)-m_1P_i=0, v_i(m_3P_i)-m_2P_i=0$，我们至少需要6个对应点才能够求解。
$Pm=0 \rightarrow P=UDV^T$，然后根据SVD可以求解出来M，再把M标准化成最后一行是0，1的形式就可以求解参数了。
P矩阵的构成为
$$
P=\begin{bmatrix}P_1^T&0&-u_1P_1^T\\
				0&P_1^T&-v_1P_1^T\\
				&\cdots&\\
				P_n^T&0&-u_nP_n^T\\
				0&P_n^T&-v_nP_n^T\end{bmatrix}=U_{2n\times12}D_{12\times12}V_{12\times12}
$$
M是$V$的最后一列变成$3\times4$矩阵$\hat M=[\hat A\ \ \hat b]$，
然后计算：
$$
\begin{aligned}
\rho &= \frac{\pm1}{||\hat a_3||} \\
c_x &= \rho^2 \hat a_1\cdot \hat a_3 \\
c_y &= \rho^2 \hat a_2\cdot \hat a_3 \\
\cos\theta &= \frac{(\hat a_2\times \hat a_3)\cdot (\hat a_1\times \hat a_3)}{|\hat a_2\times \hat a_3|\cdot |\hat a_1\times \hat a_3|} \\
\alpha &= \rho^2 |\hat a_1\times \hat a_3|\sin\theta \\
\beta &= \rho^2 |\hat a_2\times \hat a_3|\sin\theta \\
r_1 &= \frac{\hat a_2\times \hat a_3}{|\hat a_2\times \hat a_3|} \\
r_3 &= \frac{\pm\hat a_3}{|\hat a_3|}\\
r_2 &= r_3\times r_1\\
T &= \rho K^{-1}\hat b
\end{aligned}
$$

*感觉这一串推导还是很有难度的，不知道会不会考*

相机的K中的$\alpha,\beta$决定field of view（视长角）
为什么需要3个棋盘（或者至少2个不平行的棋盘）：对于只有一个棋盘，存在深度和大小的多义性。
一般来说我们需要做相片畸变的还原，计算三维坐标，最后再进行参数的计算。

出现Reproduction Errors的原因有很多种，比较常见的有：
1. 输入数据错误（标定板尺寸错误）、标定板制造不精确（尺寸不准、不平整）
2. 图像质量问题（模糊、噪声、光照变化）
3. 相机模型不合适（未考虑高阶畸变）
需要对于具体的问题多分析。

### 3D model representation
#### Depth image
2.5D的图像, $H*W*1$，最后一维的1深度。
- ray depth: 激光雷达返回的深度（实际上是长度）
- Z depth: 深度图中代表的深度

##### Depth back-projection:
将深度图中的$(u,v,z)$重新得到$(x,y,z)$.$(u,v)=(\alpha\frac{x}{z}+c_x,\beta\frac{y}{z}+c_y) \rightarrow x=z(u-c_x)/\alpha,y=z(v-c_y)/\beta$.
2.5D的含义则是由于需要得知相机$K$才能够得到三维空间坐标系下的坐标。

##### Depth Sensor
Stereo Sensors：双目传感器，可以通过双目视差来计算物体对应的距离：
$u-u' = \frac{B\cdot f}{z}$，我们称这叫做 disparity. 
$\frac{u'+B-u}{B}=\frac{z-f}{z}\rightarrow \frac{u-u'}{B}=\frac{f}{z}$
<img src="Depth-Sensor.png" width="600">


左眼和右眼都能够看到的field叫做co-visible，只有单目能看到的对应的depth是 non-valued 的。

寻找correspondence是非常困难的事情，因为可能存在：
- specularity / transparency (Non-Lambertian surface)
- occlusions, repetition
- textureless surface
本质是因为会影响features的寻找，最好还是在diffuse surface。

**Structural Light**
红外线(Infrared)来产生texture, 如果到了太阳下就会不准确，对于黑色的物体一样会存在误差。
对于 Specular 的物体，无法产生有效的 depth 值；对于 Transparent 的物体，无法产生正确的 depth 值。

#### Mesh
A piece-wise Linear Surface Representation。
如何表述triangle mesh：用 Vertex, Edge, Face 来表示，其中每个 Face 都是 Triangle。

存储
STL：triangle用vertex index indice来存储，本质是一个List，注意按照逆时针顺序存储，保证法向量方向正确（指向外）

Compute Mesh Geodesic Distance：
- Naive：直接求graph上的最短路径，这样子没有考虑从surface中间穿过的路径。
- A fast and more accurate approximation: Fast Marching method
- Exact geodesic distance: MMP method
#### Point Cloud
本质上是point set，因为点的内部顺序无关。
点云并非是 surface ， 而是二维流形：Mesh Surface + Sampling

##### Uniform Sampling
在Triangle中采样，可以采用:
- 把三角形翻折成平行四边形然后最后将点对称回来的做法实现均匀采样. $x = (1 − a_1)v_1 + (1 − a_2)v_2 + (a_1 + a_2)v_3, a_1,a_2 \sim \text{Uniform}(0,1)$
- $x=(1-\sqrt{r_1})v_1+\sqrt{r_1}(1-r_2)v_2+\sqrt{r_1}r_2v_3, r_1,r_2\sim U(0,1)$

##### Farthest Point Sampling (FPS)
目标是让两两点之间的距离和最大。
1. 首先也需要进行Uniform Sampling，要求首先采样超过需求目标的点数（over sample）。
2. 初始随机加入一个点，之后每次贪心的加入离当前点集距离和最远的点，直到达到$k$个点。

这样子如果原始点集很大的话复杂度就会比较高，所以我们先采用一次Uniform Sample缩小点集规模。

##### Distance Metrics
- Chamfer Distance：$d_{CD}(S1,S2)=\sum_x\min_y ||x-y|| + \sum_y\min_x ||x-y||$，对Sample不敏感
- Earth Mover's Distance: $d_{EMD}(S_1,S_2)=\min_{\phi:S_1\rightarrow S_2}\sum_{x}||x-\phi(x)||$，要求$\phi$是个双射，对Sample很敏感
### 3D Network
#### PointNet
对于点云的网络，由于点的顺序对于实际构成是没有影响的，所以我们需要具有 Permutation Invariance 的函数。
如果排序的话，考虑新加入一个点会对整个input造成translation的影响，所以不能采用排序的方法。

Symmetric Function：$f(x_1,x_2,\cdots,x_n)\equiv f(x_{\pi_1},x_{\pi_2},\cdots,x_{\pi_n})$.
为了保证orderless，我们采用Pointnet的结构，首先把每一个点的3维坐标变成一个C Channels的feature，然后用maxpool把这n个点的综合在一起，就变成了一个C维的Feature。然后再对这个做classification就好了。

如果要做Segmentation的话，可以把我们算出来的$n*64$的叠加一个C维的feature，然后这就是每一个点的local feature+global feature（实际上和Unet的最顶层是一样的做法）

Robustness：发现训练出来的特征选取点都是框架点（Critical Points），所以删除点/加入点只要不是框架点就不会影响结果。

Limitation：没有局部信息，无Local context，基于absolute coordinate。

#### PointNet++
考虑到PointNet没有提取到局部特征，所以考虑采取 sampling+grouping 的方法。
##### Classification
![[pointnet++_cla.png]]
新增一个set abstraction的过程：
1. 采用sample ball采取若干个小圆球，球的中心点为central point，然后将坐标变成relative coordinate。central points的选取通过FPS来选择。
2. 对于每一个小的Ball做pointnet提取得到C个Features，变成新的点集（点的数量减少，特征变多）。
最后变成只有一个点，再做pointnet就和以前一样即可。

##### Segmentation
考虑如何在点数减少之后UpConvolution回到原有点数目，可以采用Unet的方法，通过Skip Link把原本的features继承，然后对于新出现的维度的特征，采用3-interpolate的方法把最近的3个点的inverse distance weighted average累加。
![[pointnet++_seg.png]]

##### More
Pointnet是各向同性的，Conv是各向异性。

#### SparseConvNet
把 point cloud 转换成 surface voxel(此时Voxel的resolution比较大，但是点比较小，所以只储存occupied voxel)，然后用Sparse Conv来做卷积。
比较适用于大尺度下的预测（大尺度的激光雷达，小尺度还是用点云比较合适）。

### Object Detection
#### Task
- Single object: Location + Classification
- Multi Object: 如果采用sliding window的话，计算量太大了。
> Rooted mean squared loss(RMSE): $\sqrt{\frac{1}{N}\sum\Delta_i^2}$, 在0的时候没有梯度。
#### RCNN
首先用Region of Interest（RoI），然后对每一个region做一个CNN得到features，对这些做SVM和Bound Box Reg得到detection结果。

#### Fast RCNN
考虑到对于每一个RoI都存储一个CNN的参数比较的麻烦，所以可以先提取features然后对feature做一个RoI，将得到的region做crop后用同一个CNN再抽象，最后通过Linear得到Box offset，Linear+softmax得到classification.

这里对RoI做Crop用到的是**RoI Pooling**
##### RoI Pool
- 首先把边框Snap到网格上
- 对这个grid cells做 max pooling 缩小规模到2\*2或者7\*7。

#### Faster RCNN
现在的问题在于Fast RCNN的开销主要是在Region Proposal，无法做到Tracking by detection.
所以引入Region Proposal Network（RPN）
##### Region proposal network
目的：Predict the RoI.
在feature上采用sliding window的办法扫描，因为resolution很低所以开销不高。
对于每一个点有不同大小的window，我们称每一个不同的叫做一个anchor box，对这些存在：
- Object or not
- Box transforms（4种不同的corrections）

所以现在存在K\*20\*15个Objectness，4K\*20\*15个Correction.
这里我们只取objectness的top 300左右。

所以Faster RCNN也叫做two stage detector：
1. First stage: Use backbone to extract features, Use RPN to generate ~ 300 proposals.
2. Second stage: For each proposal, predict class label and bbox refinement, Perform confidence thresholding to remove low-confidence bbox predictions, Perform non-maximal suppression (NMS) for deduplication.

**YOLO: You only look once(Single-stage Detector)**
大粒度的，直接在原图上划分成7\*7的网格，然后对每一个中心点用B=3个anchor box，同时算objectness和classification（把背景也作为一类）。
快，但是效果差。

##### Non-maximum Suppression
找到置信度最高的bbox，把和他IoU高于threshold的都remove，然后再接着去剩下的最高的置信度bbox。

#### Evaluation
用Average Precision来判断，对于每一个类的bounding box，找到 top n 的 classification scores, 然后计算precision（用threshold）和recall曲线。
$AP = \frac{1}{11}\sum_{Recall_i} Precision(Recall_i)$
最终一般用不同threshold的AP mean(mAP)，或者干脆用0.50的AP来作为metric。
### Instance Segmentation
Multi Object的segmentation。

#### Mask RCNN
在每一个RoI后面加入一个Mask的binary的网络层就可以。
最大的contribution在于提出了RoI Align

##### RoI Align
对于每一个提取出来的RoI区域，不采用Snap的方法，而是对于每一个区域sampling一些点，然后用双线性插值的方法来计算这些点的值，最后做max pooling。

### 3D Object Detection and Instance Segmentation
#### Frustum PointNet
- Frustum Proposal
- 3D Instance Segmentation
- Amodal 3D Box Estimation
#### Deep Sliding Shape
在3D空间下做Sliding Window，

#### Monocular Camera
类似于Faster RCNN，只是在3D空间下，所以引入的也是3D的Proposals
#### VoteNet
点做Vote来找到表面点，然后做一个Cluster得到Bounding Box。

## Sequential Data
### RNN(Recurrent)
循环使用的神经网络，$h_t=f_W(h_{t-1},x_t),y_t = f_{W_{hy}}(h_t)$，可以理解为时序电路，用来处理 Sequential Data.

#### Vanilla RNN
> vanilla是香草的意思，在该语境下代表basic。

$$
\begin{aligned}
h_t &= \tanh(W_{hh}h_{t-1} + W_{xh}x_t) \\
y_t &= W_{hy}h_t
\end{aligned}
$$
注意Loss在BP中，高层的Loss都会回流到第一层.

**Vanishing Gradient:**
$$
\begin{aligned}
\frac{\partial h_t}{\partial h_{t-1}} &= \tanh'(W_{hh}h_{t-1}+W_{xh}x_t)W_{hh} \\
\frac{\partial L_t}{\partial W} &=\frac{\partial L_t}{\partial h_t}\frac{\partial h_t}{\partial h_{t-1}}...\frac{\partial h_1}{\partial W} \\
&=\frac{\partial L_t}{\partial h_t}\frac{\partial h_1}{\partial W}\prod_i W_{hh}\tanh'(W_{hh}h_{i-1}+W_{xh}x_i)
\end{aligned}
$$

由于tanh导数的值在(0,1)之间，就会导致距离比较远的Loss对于当前参数的影响接近于0，进一步削弱Long-term effect.

对于$\prod_i W_{hh}^t$，如果过大可以截断，但是太小了就没有办法，只能改变 RNN 结构

#### Truncated Backpropagation
在计算Backpropagation的时候只反传一部分的上文的Loss，这一段的窗口大小叫做sequence length:$\Delta T$.
*长文本的生成仍然是当前的问题。*

#### Embedding Layer
考虑对于One hot vector做hidden layer的转换操作，实际上只能够利用到W的某一列。
所以我们可以多增加一层embedding层，这个甚至可以是pretrained word embedding matrix。

#### Sampling Strategies
##### Greedy sampling
always takes the highest prob.问题在于每一次输出的都会是固定的内容
##### Weighted sampling
sample the next token according to the predicted probability distribution，可能会出现wrong token
##### Search
首先我们考虑要求一个length为$T$的最大概率结果，可以变成:
$p(y|x)=p(y_1|x)p(y_2|y_1,x)\cdots p(y_t|y_1,y_2,\cdots,y_{t-1},x)=\prod_{i=1}^Tp(y_i|y_1,...,y_{i-1},x)$

- Exhaustive Search：直接搜索全空间，复杂度为$O(V^T)$
- Beam Search: 每一个step仅保留$k$个最优的结果，实现每一步都是在$k^2$之中寻找$k$个最优解。k被称作the beam size

### LSTM

对于RNN中，本来存在$h, x, W$，现在我们把$W\begin{bmatrix}h\\x\end{bmatrix}$过一个$\begin{bmatrix}\sigma\\\sigma\\\sigma\\\tanh\end{bmatrix}$，形成$\begin{bmatrix}i\\f\\o\\g\end{bmatrix}$
- i: Input gate, whether to write
- f: Forget gate, whether to erase
- o: Output gate, how much to reveal
- g: Gate gate, how much to write

然后$c_t=f\odot c_{t-1} + i\odot g, h_t = o\odot \tanh(c_t)$，这个符号表示按位乘法。

维度可以理解为$(4h \times 2h)(2h\times 1) = 4h\times 1$

LSTM提供了一个类似ResNet的Skip Link方法。

### GRU(Gated Recurrent Unit)
$$
\begin{aligned}
r_t&=\sigma(W_{xr}x_t+W_{hr}h_{t-1}+b_r)\\
z_t&=\sigma(W_{xz}x_t+W_{hz}h_{t-1}+b_z)\\
\hat{h}_t &= \tanh(W_{xh}x_t+W_{hh}(r_t\odot h_{t-1}) + b_h)\\
h_t&=z_t\odot h_{t-1} + (1-z_t)\odot \hat{h}_t\\
\end{aligned}
$$

### Image Captioning 
可以先用CNN从图片中提取关键信息再过RNN，也可以用两个网络训练然后合并得到结果（VQA）

### Attention 
对于我们知道的input x得到的h，用网络$f_{att}(s_t, h_i)$得到对应的weight，做一个softmax后把h加权得到context $c_t$。$s$的获得则是由$s_t=g_U(y_{t-1},s_{t-1},c)$。
#### Cross-Attention Layer
Inputs: $Q,X,W_K,W_V$

Outputs:
- Keys: $K=XW_K$
- Values: $V=XW_V$
- Similarities: $E=QK^T / \sqrt{D_Q}$
- Attention Weight: $A=\text{softmax}(E, dim=1)$。对列求softmax。
- Output: $Y=AV$
#### Self-Attention Layer
只需要把上面的$Q$变成$Q=XW_Q$即可。

但是这个具有permutation equivariant，所以可以在Q上面加入一个order的信息来表示第几个。
##### Masked ...
或者我们在算 similarities 的时候，可以只看前面的（将未来的E全部设置成-Inf），这样子就可以预测未来。
##### Multiheaded ...
有H个不同的Self-Attention，把他们生成的结果综合在一起。最后增加一个$W_O$的矩阵来把output变成和input同dimension。

### Comparison
- RNN：好处在于复杂度低，坏处在于无并行。
- Convolution：好处在于可并行，坏处在于对于长序列需要存储很多数据
- Self-Attention：好处：长序列很好，高并行度，只有4个矩阵乘法。坏处在于计算复杂度开销很高。

### Transformer
Input: X
Output: Y

- X过一个Self-Attention层到A
- A过Layer Norm到B
- B过MLP和Skip Link到C
- C过一个Layer Norm到Y

*Layer Norm:*
对于每一个不同的token.
$$
\begin{aligned}
\mu_i &= (\sum_j h_{i,j}) / D \\
\delta &= (\sum_j (h_{i,j} - \mu_i)^2 / D)^\frac{1}{2} \\
z_i &= (h_i - \mu_i) / \delta_i \\
y_i &= \gamma \times z_i + \beta
\end{aligned}
$$

#### Language Model
需要在首和尾添加 Embedding matrix， 把单词转换成feature的维度(V\*D)。

#### Vision Transformers(ViT)
把图片分割成若干个小的patches，flatten之后通过linear层变成D维feature

#### Tweaking Transformer
##### Pre-Norm Transformer
把最后一层的Layer Norm变到第一层，训练可以变得更加的稳定。

##### RMS Norm
把Layer Norm改成RMS Norm
$y_i = \frac{x_i}{RMS(x)} * \gamma_i$
$RMS(x) = \sqrt{\varepsilon + \frac{1}{N}\sum_i x_i^2}$

##### SwiGLU MLP
本来的MLP是$W_1[D\times 4D],W_2[4D\times D]$,$Y=\delta(XW_1)W_2$
更新之后是：
$$
\begin{aligned}
&W_1,W_2,W_3[D\times H]\\ 
&Y=(\delta(XW_1)\odot XW_2)W_3
\end{aligned}
$$
这样子并不可以减少参数量，$H=\frac{8D}{3}$，主要是能够增加稳定性(more stable)。

##### MoE(Mixture of Experts)
MLP的部分更新成$E$个不同的MLP，但是每一个Token只过A个MLP，称为active experts.

## Generative Models
### Fully Visible Belief Network (FVBN)
考虑$p(x)=p(x_1,x_2,\dots,x_n)=\prod_ip(x_i|x_1,x_2,\dots,x_{i-1})$
然后就可以用PixelRNN或者PixelCNN来计算$p(x)$的值。
但是这个算法生成的很慢。
### VAE
考虑autoencoder的过程是从x encoder到特征z，然后再从特征z decoder到x'。
用全概率公式拆分$p(x)=\frac{p(x,z)}{p(z|x)}=\frac{p(z)p(x|z)}{p(z|x)}$
此时我们钦定$Z\sim N(0,I)$，那么问题就变成了求解$p(x|z)$和$p(z|x)$
我们通过一个分布$q(z|x)$来近似$p(z|x)$，假设这些分布都是正态分布，就训练出来他们的均值和标准差。
$L=-\log p_\theta(x)$，下文假设L是$\log p_\theta(x)$，然后要最大化L。
![[Pasted image 20250618002824.png]]
这个时候就取前两部分作为Loss，称为ELBO。
在计算的时候$E_z[\log p_\theta(x^{(i)}|z)]$需要用到蒙特卡洛估计来近似。

为什么叫variational？因为实际上是对$p_\theta$和$q_\phi$这两个函数求极值，但是我们也可以理解为对它的参数求解。

### GAN(Generative Adversarial Networks)
输入一个随机噪声z，通过一个Generator网络得到一张图片，再通过Discriminator网络判断这张图片是真/假。

训练的目的在于让Generator网络得到一张尽可能不被判断为假的图片，以及让Discriminator网络尽可能把一张图片判断的准确。

#### Generator Gradient Ascent
这里我们发现对生成网络做Gradient Descent会在初期很缓慢，所以我们直接做Ascent。
$\max_{\theta_g}E_{z\sim p(z)}\log(D_{\theta_d}(G_{\theta_g}(z)))$
训练的时候可以做k步Discriminator的调优，然后再做一步generator的调优。这里k的取值可以为1或者大于1的值。
注意，在generator的网络和一般的CNN网络不一样
- 把pooling层用卷积层替代
- 采用batchnorm归一化
- 删除全连接层
- 在generator中，除了最后一层用tanh，其它层都用ReLU激活
- 在discriminator中，都用LeakyReLU激活。

#### Quantitative Measurement
$FID(r,g)=|\mu_r-\mu_g|^2 + Tr(\sum_r+\sum_g-2\sqrt{\sum_r\sum_g})$
FID越小，两张图之间的差异越小。

GAN还发现可以用噪声的向量运算得到想要的图片（smiling women - natural women + natural men = smiling men）。

缺点在于无法得到准确的$p(x)$值，训练会不稳定（可能出现Mode Collapse的情况）
### Diffusion Model
考虑VAE只用了一个正态分布来拟合，假设它应该是$T$个高斯分布的累加。
这样子再去训练得到的VAE模型等于对噪声不断加噪和去噪的过程，称为diffusion model。

三者的比较：
- VAE不能生成高质量图片
- GAN不能够有好的模式覆盖度
- Diffusion Model没有快的采样时间（运行开销大）。