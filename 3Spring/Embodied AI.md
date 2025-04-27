## Robotics 

Kinematics VS Dynamics:
- 运动学只考虑物体的运动，而不考虑如何通过force来实现这一过程。
- 动力学则需要从力和力矩出发，计算如何实现。

### Rigid Transformation

Link：连杆，可以理解为机器人的手臂部分（刚体）
Joints：关节，连接连杆的部分，具有灵活度。
DoF：The degree of freedom (DoF), 表明了这个系统的独立参数个数。

$$
\begin{aligned}
o_s\rightarrow o_b:\\
o_b &= o_s + t_{s\rightarrow b}\\
[x_b^s,y_b^s,z_b^s]&= R_{s\rightarrow b}[x_s^s,y_s^s,z_s^s]

\\
\\
t_{s\rightarrow b}^s &= o_b-o_s = o_{b}^s\\
R_{s\rightarrow b}^s &= [x_b^s,y_b^s,z_b^s]
\end{aligned}
$$
那么实际上$p$的坐标变化(Coordination Transformation)就是$p^s = Rp^b + t$，在欧几里得空间内不是线性变化，所以我们放到齐次坐标系内，发现他就是一个线性变化。

这个时候引入一个矩阵T来表示：
$$
\begin{aligned}T = \begin{bmatrix}R&t\\0&1\end{bmatrix}\end{aligned}
$$
那么有:$x^1 = T_{1\rightarrow 2}^1 \dots T_{n-1\rightarrow n}^{n-1}x^{n} = T_{1\rightarrow n}^1x^n$.

### Multi-Link Rigid-Body Geometry

Joint type一般分为Revolute(R)型和Prismatic(P)型，表示旋转和拉伸。还有其他的包括：Helical joint(和螺丝一样可以旋扭)，Ball and socket joint(球形关节，自由度为3).

Base link(root link)：Regarded as the “fixed” reference，作为robot的第0个link。
end-effector link: 末端Link，如夹爪等。

存在两种表示robot状态的方法：
1. Joint Space：通过描述在Joint Space中的变化来求解在目标位置
2. Cartesian Space: 直接描述在笛卡尔坐标系中的位置，需要反过来求解Joint Space中的参数。
#### Forward Kinematics
前向运动学，给你每个Link和Joint的Pose和参数$\theta$，要你计算最后的Gripper的pose

#### Inverse Kinematics
给定最终的Gripper Pose，要求计算参数$\theta$。

形象化地说: $T = T_{s\rightarrow t}(\theta)$,FK就是给定函数$T_{s\rightarrow t}$和$\theta$求$T_{target}$，IK是给定函数$T_{s\rightarrow t}$和$T_{target}$求$\theta$

> Why 6 or 7 DoF arms? 因为要做到transformation需要R和T一共6个自由度，如果要做抓取任务的话还得分析一个gripper的width自由度，一共就是7个。

Pieper准则：如果一个6-DoF的arm具有如下性质之一：
- 三个连续旋转关节的轴线相交于一点。
- 三个连续旋转关节的轴线平行。
那么他就具有IK的解（但是这个只是充分条件而非必要）

结合上面关于Robot状态的representations，发现：
- FK就是Joint Space中变化的解释
- IK就是Cartesian Space中变化的反推

### SO(3) and SE(3)

$\mathbb{SO}(3)$:3d Special Orthogonal Group,表示旋转的空间。其中的矩阵$R$满足约束:$RR^T=I, \det(R)=1$
$\mathbb{SE}(3)$:3d Special Euclidean Group, 表示刚体变换的空间，其中矩阵$T = \begin{bmatrix}R &t\\0&1\end{bmatrix}$，要求$R\in \mathbb{SO}(3)$

关于$\mathbb{SO}(3)$中的矩阵的表达存在多种方式，我们需要找到
#### Euler angle
$R = R_x(\alpha)R_y(\beta)R_z(\gamma)$，分别用在三个轴上的旋转乘起来表示最终的旋转.

Inspection：存在Not unique的情况，而且当其中某一个angle为90度的时候，出现**Gimbal Lock**。这个时候另外两个角就只有1个自由度了。

虽然欧拉角可以表示所有旋转矩阵，但是存在不唯一性和无法表达*变化*

#### Angle-Axis
通过给定转轴$\hat \omega$和转角$\theta$来确定一个旋转。
$[a]:=\begin{bmatrix}0&-a_3&a_2\\a_3&0&-a_1\\-a_2&a_1&0\end{bmatrix}, a\times b = [a]b$

通过Rodrigues formula可以知道：
$Rot = I + [\hat\omega]\sin\theta + [\hat\omega]^2 (1 − \cos\theta)$，然后我们用泰勒公式把theta展开，同时由$[\hat\omega]^3=-[\hat\omega]$可知：
$Rot(\hat\omega,\theta)=e^{[\hat\omega]\theta}$
我们也称$\vec \theta=\hat\omega \theta$为旋转向量（rotation vector / exponential coordinate），也就是模长为旋转角度的转轴向量。
很明显$(\hat\omega,\theta)$和$(-\hat\omega, -\theta)$是同一个旋转，所以我们固定在$(0,\pi)$上每一个旋转矩阵唯一对应一个Angle-Axis表示：
$$
\begin{aligned}
\theta &= \arccos\frac{1}{2}[\text{tr}(R) -1]\\
[\hat\omega] &= \frac{1}{2\sin\theta}(R-R^T)
\end{aligned}
$$
通过这个表达我们可以求出来两个旋转矩阵的距离：
$$
\begin{aligned}
R_2&=R_2R_1^TR_1=(R_2R_1^T)R_1\\
dis(R1,R2)&=\theta(R_2R_1^T)=\arccos\frac{1}{2}[tr(R_2R_1^T)-1]
\end{aligned}
$$
#### Quaternion
四元数，$q = (w, \vec{v})=w+xi+yj+zk$
$||q||^2=w^2 + \vec{v}^T\vec{v}, q* = (w, -\vec{v}), q^{-1}=\frac{q*}{||q||}$.
对于$q1=(w_1, \vec{v_1}), q2=(w_2, \vec{v_2})$,有：
$q1q2 = (w_1w_2 - \vec{v_1}^T\vec{v_2}, w_1\vec{v_2}+w_2\vec{v_1}+\vec{v_1}\times\vec{v_2})$
unit quaternion： $||q||=1$,有3个自由度。

##### exponential coordinate to quaternion
quaternion和exponential coordinate：
$$
\begin{aligned}
q &= (\cos(\frac{\theta}{2}), \sin\frac{\theta}{2}\vec{\omega}) \\
\theta &= 2\arccos w\\

&\begin{eqnarray}
		\vec{\omega}=\left\{
		\begin{aligned}
			&\frac{1}{\sin\frac{\theta}{2}}\vec{v} &\theta\neq 0\\
			&0&\theta=0
		\end{aligned}
		\right.
\end{eqnarray}
\end{aligned}
$$
要把一个向量$x$旋转quaternion对应的角度，应该是
1. 将x变成quaternion$(0, x)$.
2. $x'=qxq^{-1}$
3. $x'$的vector部就是旋转之后的向量。

*比较需要注意的是quaternion在不同的库中的顺序不一样*

##### rotation matrix to quaternion
$$
\begin{aligned}
R &= \begin{bmatrix}
1-2y^2-2z^2&2xy-2zw&2xz+2yw\\
2xy+2zw&1-2x^2-2z^2&2yz-2xw\\
2xz-2yw&2yz+2xw&1-2x^2-2y^2
\end{bmatrix}
\\
tr(R)&=4w^2-1 \\
w&=\frac{\sqrt{tr(R)+1}}{2}, x=\frac{R_{32}-R_{23}}{4w},y=\frac{R_{13}-R_{31}}{4w},z=\frac{R_{21}-R_{12}}{4w}
\end{aligned}
$$

distance between quaternion(仍然根据之前转换到Axis-Angle的角度计算): $\theta = 2\arccos(|q_1\cdot q_2|)$

##### slerp
考虑两个quaternion在旋转的时候求中间时刻的quaternion值，有：
$v_t=\omega_1v_1+\omega_2v_2=\frac{\sin(1-t)\theta}{\sin\theta}v_1+\frac{\sin t\theta}{\sin\theta}v_2$,代入成quaternion有：
夹角$\theta=\arccos (q1\cdot q2), q_t = \frac{\sin (1-t)\theta q_1 + \sin t\theta q_2}{\sin\theta}$
比较需要注意的是如果$q1,q2$的夹角大于$\frac{\pi}{2}$，可以选择把$q2$改成$-q2$，再计算slerp。

##### Uniform Sampling in SO(3)
在$\mathbb{SO}(3)$中对rotation matrix均匀采样等价于在$\mathbb{S}(3)$中对quaternion均匀采样，因为他们的距离是线性关系的。
实际上就是均匀采样4个数然后归一化，再把它转换成矩阵的形式。

### Motion Planning

$\mathcal{C}-space$，是所有状态的子集：
$\mathcal{C}_{free}$:valid states
$\mathcal{C}_{obs}$:obstacle states
Motion planning实际上就是给定$\mathcal{C}_{free}$，然后给定起始点$q_{start}$和$q_{end}$，让你规划一条最优action路径。

如何判断robot是否发生碰撞？将机器臂的collision mesh用forward kinematics求出来，判断和obstacles的mesh是否相交。
但是mesh的复杂度太大了，**所以我们改成用球来覆盖机器臂mesh**，这样可以减少计算的复杂度。

*Alternative: 用Approximately Convex Decomposition来近似Mesh，可以用比较少的cluster数来近似一个物体。*

#### Probabilistic Roadmap Method (PRM)
在Space中随机选择在$\mathcal{C}_{free}$中的N个点，然后对于这N个点求$k$近邻，将合法的边加入图中，最后在新构成的图中用Dijkstra来求解最短路。
但是uniform sampling无法处理Narrow Passages的情况，所以我们有：
- Gaussian Sampling：对于随机采样的$q_1$，用高斯分布$N(q1, \sigma^2)$选取$q_2$。如果$q_1 \in \mathcal{C}_{free},q_2\notin \mathcal{C}_{free}$，就把$q1$加入点集中。这样子采样出来的是在边界附近的点。
- Bridge Sampling: 随机采样$q_1$,用高斯分布采样$q_2$，如果$q_1\notin \mathcal{C}_{free},q_2\notin \mathcal{C}_{free}$，就将$q_3=\frac{q_1+q_2}{2}$加入点集。这样子采样的点位于原本空间的桥上。
#### Rapidly-exploring Random Trees (RRT)
每一次采样随机选择空间中的一个点或者以$q_{goal}$作为目标，选择已有点集内的距离$q_{target}$中最近的点作为$q_{near}$，然后在这个向量上移动一段距离，如果这一条边和新的点在$\mathcal{C}_{free}$中，就可以把他们加入点集，最后再用dijksra求解。
**RRT-connect：$q_{start}$和$q_{goal}$中同时扩展树。**

shortcutting：通过在已有路径上随便选取两个点，直接把他们两个点相连，判断这个路径是否能够被简化。
优化的做法可以是不断的shortcutting，然后不断的运行RRT，再加上shortcutting，判断新的曲线能否更优。
### Control System
- 开环控制(FeedForward,FF)：直接由input得到output。
- 闭环控制(Feedback,FB): 由input不断通过循环修正error得到output。
也有FF+FB的控制器。

Effectiveness:
- minimize the steady state error
- minimize oscillations around the steady state
- reach steady state fast

error function:$\theta_e = \theta_d - \theta$,$\theta_d$表示desired state，$\theta$表示当前状态。
Steady-State Error:$e_{ss}=\lim_{t\rightarrow \infty}\theta_e(t)$
#### PID Controller
proportional controler: Controller的返回值为$P=K_p\theta_e(t)$:
一般对状态的影响为一阶导，若$\dot\theta_d(t)=C$，则：
$$
\begin{aligned}
\dot{\theta}(t)&=K_p\theta_e(t)\\
\dot{\theta}_d(t)-\dot\theta_e(t)&=K_p\theta_e(t)\\
\theta_e(t)&=\frac{c}{K_p}+(\theta_e(0)-\frac{c}{K_p})e^{-K_pt}
\end{aligned}
$$
如果影响变成二阶导，我们引入Integral Signal:$I=K_i\int_0^t\theta_e(\tau)d\tau$.
那么我们就能得到PI Controller:
$PI = K_p\theta_e(t)+K_i\int_0^t\theta_e(\tau)d\tau$

PI控制器可以消除目标速率的影响。

接下来考虑对误差做缓冲，引入Derivative Signal:$D=P_d\frac{d}{dt}\theta_e(t)$
这个时候就构成了PID Controller:
$$
PID = K_p\theta_e(t)+K_i\int_0^t\theta_e(\tau)d\tau+K_d\frac{d}{dt}\theta_e(t)
$$
Tuning PID非常经验的，比较常用的是PD（如果我们目标是s，那么P控制的是位置，D控制的就是速度）。
## Vision and Grasping

抓取分为开环抓取（在收到视觉信息后直接抓取最后判断是否成功）和闭环抓取（在操作中还需要根据反馈信息调整）
在开环抓取中：
- 如果对于已知物体，可以采用姿态估计的办法
- 对于未知物体，则直接预测抓取位姿。

### 6D Pose Estimation
通过Object coordinate求出来camera coordinate:$(X',Y',Z') = R(X,Y,Z) + T$
应用：
- 6D Pose for Robotic Manipulation
- Augmented Reality（AR）
- 6D Pose for Human Object Interaction
#### Rotation Regression
如何预测这个旋转矩阵呢？可以用不同的rotation representations来预测：
- Rotation Matrix：9D
- Euler Angle：3D
- Axis-Angle：3D
- Quaternion：4D
但是采用下面3种方法预测的时候，都会出现奇异性和不连续性（Singularities and Discontinuity）
- Euler Angle：从$2\pi$到$0$的时候，旋转角度连续但是对应的自变量出现跳跃
- Axis-Angle：$\theta=\pi$的时候，当前轴和他的反方向对应同一个旋转，也会出现不连续。
- Quaternion：我们只取上半部分4维球，但是在球大圆上总会出现不连续的情况。

所以最后我们还是采用预测原始矩阵的办法，但是用6D的做法（只给前两个行向量），然后再用**施密特正交化**计算对应的旋转矩阵：
$$
\begin{aligned}
b_1 &= Norm(a_1) \\ 
b_2 &= Norm(a_2 - (b_1\cdot a_2)b_1) \\
b_3 &= b_1 \times b_2
\end{aligned}
$$
但是我们发现这样子每一个点对应的权重被天然划分了，不符合我们的要求，所以采用9D预测，通过SVD来计算旋转矩阵：$R=UDV^T$,要求$\det(D)=\det(U)\det(V)$
#### Rotation Fitting

先预测object coordinate，Goal：给定两组点要求用旋转和平移的方式把他们对齐。
现在$M$是ground-truth，$N$是object coordinate，那么要求的实际上是：
$\hat A = \text{argmax}_{A\in\mathbb{R}^{3\times 3}}||M^T-AN^T||_F^2,A^TA=I$，其中$||||_F$表示$\sqrt{trace(X^TX)}=\sqrt{\sum_{i,j}x_{i,j}^2}$
通过SVD可以得到:$M^TN=UDV^T, \hat A=UV^T$.注意如果要求$\hat A\in \mathbb{SO}(3)$，那么需要$D$的行列式为$\det UV^T$。

接下来我们结合$Rot$和$Trans$：
先把$M$和$N$全部减去mean，然后预测$R$，最后将$T=\overline{M}^T-\hat{R}\overline{N}^T$.

**注意SVD对outlier很敏感，所以要结合RANSAC使用**

#### Perspective-n-Point (PnP) Algorithm
如果知道相机的内参$K$，实际上我们可以知道对于image的pixel(u,v):
$sp_c=K[R|T]p_w$,然后用我们求解相机参数的办法求$T$和$R$就好了。

### Instance-Level 6D Object Pose Estimation
#### PoseCNN
输入是RGB/RGBD图，输出是已知实例的具体位姿。需要已知CAD模型
dataset：YCB objects
**通过quaternion来进行旋转回归，网络分为分类网络和回归网络（通过Rol+quaternion实现）**
#### Iterative Closet Point(ICP)
1. 将两个点云中心化，得到$\widetilde{P}$和$\widetilde{Q}$
2. 找到$\widetilde{P}$中每一个点对应的最近的$\widetilde{Q}$上的点构成曲线$P_{corr}$
3. 通过$P_{corr}$和$P$用SVD求出Rotation和Translation
4. 最后更新$P$作为答案
由于叫做iterative，所以需要迭代上述过程。

- 优点：
	- 简单
	- 初始状态好的情况下表现良好
- 缺点
	- 计算复杂
	- 没有对点云结构的考虑
	- 高度依赖初始状态的正确性
也诞生了变种：Point-to-Plane ICP（考虑目标点云的整体结构），Plane-to-Plane ICP（考虑了初始和目标点云的结构）、Fast and Robust ICP。

PoseCNN+ICP可以提升准确率，但是实例级别的预测仍然局限于实例个数，CAD模型。
### Category-Level 6D Object Pose Estimation

Goal：对于新兴事物也能够预测位姿，不需要使用CAD模型。
#### NOCS for 9D estimation
Work：Normalized Object Coordinate Space for Category-Level 6D Object Pose and Size Estimation

首先我们需要求出NOCS：
1. 对齐object的方向：比如说把杯子全部都转到杯口朝上，杯柄朝右
2. 把这个物品放到一个0坐标中心内。
3. 缩放尺寸把他放入一个$1*1*1$的立方体内。

那么Category-Level的Pose Estimation就是从NOCS坐标转换到相机坐标系。
输出应该是6D位姿+3D大小。

难点在于：
1. 类别内的object形状各异
2. 数据难以生成和标注。

针对数据方面可以用Mixed-reality data generation pipeline（如CAMERA:Context-Aware MixEd ReAlity），就能够生成这样子的数据，同时自动标注。
在NOCS网络的最后存在Pose Fitting层，采用Umeyama算法估计7D相似性，RANSAC去除异常值。

可以用来对透明的物体做pose estimation，从而进行抓取。
*如果能够抓取刚体，那么肯定能够抓取柔性物体（无非就是多用点力）*

#### GAPartNet
把物品分成小的part，对于零件来做位姿和抓取。

### Object Grasping
**Grasp Synthesis is a high-dimensional search or optimization problem to find gripper poses or joint configurations.**
抓取综合就是要分析夹抓位姿和关节的设置参数。

夹抓分为4DoF夹抓（3DoF的Position和1DoF的Orientation，即只能在z轴上转动）和6DoF夹抓（3DoF的Position和3DoF的Orientation）
#### Grasp and Closure
*从左到右表示条件的强弱，form grasp的条件非常强，需要很多附加信息*
Success Grasp <= force Grasp <= form Grasp.
在计算force grasp的时候，我们引入摩擦锥的概念(实际上就是对于这个点的摩擦因子对应的$\tan\theta \le \mu$计算出的$\theta$)，每个摩擦锥可以被近似成一个底面为6边形的锥体，然后列出一个$3*n/6*n$的矩阵，分别表示2D和3D的情况，此时满足force closure的要求为：
$$
\begin{aligned}
rank(F)&=3/6\\
Fk&=0
\end{aligned}
$$
#### Grasp Data Synthesis
Dataset分为：
- object dataset：without grasping annotation，ShapeNet、ModelNet、Objaverse-XL，需要再做标注的工作。
- Synthetic dataset：with grasping annotation，ACRONYM
- Real dataset：with grasping annotation, GraspNet-1Billion
这里重点介绍**GraspNet-1B**:
1. 通过点云来计算抓取点
2. 在抓取点上找到对应的抓取位姿
3. 进行碰撞检测

#### Grasp Detection
Visual Representation：
- Voxel Grid
- Point Cloud
- Image
测量的标准则是：
- Success rate
- Planning time
- Percent cleared
##### Volumetric Grasping Network: Real-time 6 DOF Grasp Detection in Clutter（VGN）
Input是Voxel Grid，用来检测6DoF的夹抓。

- 首先从TSDF的Voxel Grid到3种representations: quality（表示这个抓取点的抓取成功率）, orientation(用quaternion表示rotation)以及width（表示夹抓宽度）
- 处理完之后还需要用非极大值抑制来得到更好的结果
- 在合成数据的生成方面注意判断碰撞。
##### GS-Net
- Pre backbone Network: 将点云N\*3变成N\*(3+C)，具有$C$个feature channel。==注意这里卷积用了SparseConv的技术，可以快速的对点云这种内部很多空0的做卷积==
- Graspable Farthest Point Sampling：选取graspness score大于阈值的点。
- Graspable Probabilistic View Selection：选择抓取点上的最佳view
- Cylinder-Grouping：把每一个采样点在view上的圆柱对应的radius和height计算出来
- Grasp Generation from Candidates：对于不同的angle和depth对，返回score和对应夹抓的width。
#### Conditional Grasp Generative Model

##### DexGraspNet 2.0
灵巧手版本的GraspNet，仍然可以分成两个部分：Seed Point和Grasp Pose的分别采取。
主要是采用local feature，让他对于不同的抓取对象都能够有比较优秀的表现。
*我们在分析中发现决定抓取成功率最主要的因素还是数据集大小*
Contribution:
- Large-scale Benchmark for dex-grasp.
- an end-to-end generative grasp prediction pipeline
##### DREDS
主要介绍了的是domain randomization的内容。
- object layout / object material / background
- illumination / camera viewpoint

##### ASGrasp
我们发现对透明和反光的物体我们Image中的Depth不能够很好的表达这个物体的真实状况。
原理在于双目识别对于透明/反光的效果，然后生成真实的点云再去做抓取。
可以理解为DREDS+GraspNet.
#### Another Vision-based Approach: Affordance
##### Where2act
找到pull和push的可能接触物体（affordance）以及在未来他们可能进行的动作
##### VAT-Mart
先找到Affordance然后预测Trajectory Proposals

### Conclusion
Vision-based 的开环操作可以：
- 预测物体位姿
- 预测抓取位姿
- 预测affordance
但是缺点在于需要一些预设（主要是在于策略）
## Policy Learning

### Imitation Learning
Behavior Cloning是模仿学习的一种，但是由于distribution drift，可能导致我们的真实情况和专家本身预设的不同。

#### DAgger: Dataset Aggregation
通过不断的与环境交互，然后在线询问来更新数据集。
这样子能够跨大数据集大小，从而让可能出现的偏差在新的数据集中出现。
询问得到Action Labels的方法：
- Human
- Teacher Policy
- Optimal Solution

#### More
但是在拟合的时候可能出现的问题：
1. 并非马尔可夫行为：当前决策可能和历史信息有关，此时可以采用LSTM（Long-Short Term Memory）进行有效的拟合。
2. multimodal behavior：解决方法有很多种，改为预测action的分布。
	- output mixture of Gaussians：通过预测action的分布而不是直接预测action. $\pi = \prod \mu_i\mathcal{N}(\rho, \sigma^2)$
	- Latent variable models：隐变量来预测action的分布。
	- diffusion models：添加噪声来还原action。
	- Autoregressive discretization：通过把action分解求解每一个小动作$a_{t,i}$的分布乘起来作为最后$a_t$的分布。

Multi-Task learning:
利用所有的数据，在计算reward的时候引入一个goal的变量参考。
**但是会存在另一个distribution drift：goal。如果goal从来没有见过也会出现偏移。**

### Reinforcement Learning
与监督学习不同的是，强化学习给出的是reward，我们无法得到对于给定input的ground-truth output，只能知道每一步决策对应的奖惩。
这里给出一些Notations:
 - $A$:action space
 - $S$:State space
 - $r$:Reward function
 - $T$:transition operator($T(i,j,k)=p(s_{t+1}=i|s_t=j,a_t=k)$)
 - $O$:Observation space

这个时候动作构成了一个集合$\tau = (s_1,a_1,\cdots,s_t,a_t)$,这个集合发生的概率我们可以写成$p_\theta(\tau)=p(s_1)\prod\pi_\theta(a_t|s_t)p(s_{t+1}|a_t,s_t)$，其中$\theta$是我们关于action的参数。
### Policy Gradient
Learning的分类：
- Online RL: 与环境会进行交互，得到实际的值
	- On-policy: 只能根据当前的 policy. 
	- Off-policy: Policy 可以是当前的，也可以是之前任意一个 policy，会将这些数据存储在一个buffer D中。 
- Offline Policy: 对环境的感知源于给定的输入 input. 
	- Behavior Cloning：全是通过expert demonstration来，没有与环境的交互，也不需要reward。
	- Off-policy RL：先把所有的policy与环境交互得到buffer D，然后通过这个里面的数据训练得到模型$\pi$。

首先我们知道 RL 的reward和函数为$J(\theta) = E_{\tau \sim p_\theta(\tau)}[\sum_{t=1}^T r(s_t, a_t)]$
接下来我们定义
$$
\begin{aligned}
r(\tau) &= \sum_{t}r(s_t, a_t)  \\
J(\theta) &= E_{\tau \sim p_\theta(\tau)}[r(\tau)] = \int p_\theta(\tau)r(\tau)d\tau \\
\nabla_\theta J(\theta) &= \int \nabla_\theta p_\theta(\tau)r(\tau)dr = \int p_\theta(\tau)\nabla_\theta \log p_\theta(\tau)r(\tau)dr = E_{\tau\sim p_\theta(\tau)}[\nabla_\theta \log p_\theta(\tau)r(\tau)] \\
p_\theta(\tau) &= p(s_1) \prod_{t}\pi(s_t,a_t)p(s_{t+1}|s_t,a_t) \\
\log p_\theta(\tau) &= \log p(s_1)+\sum_{t}(\log \pi_\theta(s_t,a_t) + \log p(s_{t+1}|s_t,a_t)) \\
\nabla_\theta \log p_\theta(\tau) &= \sum_{t}\nabla_\theta \log\pi(s_t,a_t) \\
\nabla_\theta J(\theta) &= E_{\tau \sim p_\theta(\tau)}[(\sum_{t}\nabla_\theta \log\pi(s_t,a_t))(\sum_{t}r(s_t, a_t))] 
\end{aligned}
$$

Policy Gradient(REINFORCE) 和 Maximum Likelihood 的区别：
$$
\begin{aligned}
\nabla_\theta J(\theta) &= E_{\tau \sim p_\theta(\tau)}[(\sum_{t}\nabla_\theta \log\pi(s_t,a_t))(\sum_{t}r(s_t, a_t))]  \\
\nabla_\theta J_{ML}(\theta) &= E_{\tau \sim p_\theta(\tau)}[\sum_{t}\nabla_\theta \log\pi(s_t,a_t)]
\end{aligned}
$$
REINFORCE 实际上是给每一个梯度增加了一个关于 reward 的权重。
#### Reducing variance
但是由于trajectory $\pi$是高维空间下的值，可能会出现很大的variance.
##### Reward to go
在计算最后的 reward 权重的时候，对于 t 时刻的action，只有 $t' > t$ 的 reward 是有效的。“未来不为过去负责。”
$$
\nabla_\theta  J(\theta)=\frac{1}{N}\sum_{i=1}^N\sum_{t=1}^T\nabla_\theta \log \pi_\theta(a_{i,t}|s_{i,t})(\sum_{t'=t}^Tr(s_{i,t'},a_{i,t'}))
$$
##### Baseline

给 reward 全部都减去一个 baseline, 通过对其积分发现不影响我们对于 gradient 的计算。
$$
\begin{aligned}
\nabla_\theta J(\theta)&=\frac{1}{N}\sum_{i=1}^N\nabla_\theta \log p_\theta(\tau)[r(\tau)-b] \\
E[\nabla_\theta \log p_\theta(\tau)b] &=\int p_\theta(\tau) b\nabla_\theta \log p_\theta(\tau)d\tau\\
&=\int b\nabla_\theta p_\theta(\tau)d\tau\\
&=b\nabla_\theta1=0
\end{aligned}
$$

不同的 baseline 对应相同的均值，但是是不同的方差 variance. 
#### Actor-Critic
##### Algorithm

在Policy gradient的reward to go中，我们的权重函数是吧不准确的，我们改成分布的均值之后能够减少variance
现在我们引入新的几个变量：
- Q Function: Q(s, a) 是 total reward from taking action $a_t$ at $s_t$. $Q^\pi (s_t,a_t)=\sum_{t'=t}^T E_{\pi_\theta} [r(s_{t'},a_{t'})|s_t,a_t]$
- V Function: V(s) 是 total reward from $s_t$. $V^\pi (s_t)=E_{a_t\sim \pi(a_t|s_t)}[Q^\pi(s_t,a_t)]$
- A Function: A(s, a) = Q(s,a) - V(s). $A^\pi (s_t,a_t)=Q^\pi (s_t,a_t) - V^\pi(s_t) = r(s_t,a_t) + V^\pi(s_t+1)-V^\pi(s_t)$
等于我们将 baseline 定义成 V(s). 
通过一系列**数学推导**我们可以把 A(s, a) 变成只和 r(s, a) 和 V(s) 相关的函数，接下来问题就变成了如何求解 V(a). *通过神经网络实现*
我们训练的目标一步一步随着数学推导而变化成了$V^\pi(s_t)$.

对于$V^\pi(s_t)$我们近似用$y_{t}=\sum_{t'=t}^T r(s_t',a_t')=r(s_t,a_t)+V^\pi_\phi(s_t',a_t')$来代替，这个时候就可以用监督学习的办法回归出$(s_{i,t},y_{i,t})$得到模型$\phi$.


Actor: $\pi(a_t|s_t) \leftrightarrow a_t = f_{\pi}(s_t)$. Taking action，用来训练action的网络
Critic: $V^{\pi}_{\phi}(s_t)$. Evaluating value of action. $L(\phi)=\frac{1}{2}\sum_i ||\hat{V}^\pi_\phi(s_i)-y_i||^2$

在这之上还有一个小trick，因为如果$T\to \infty$的话，计算出来的$V^\pi_\phi$可能很大，我们可以给$y_{i,t}$的计算加入一个$\gamma$值作为discount factors.
此时$y_{i,t}=r(s_t,a_t)+\gamma V^\pi_\phi(s_{t'},a_{t'})$,而我们计算$A_{s,t}=y_{i,t}-V^\pi(s_t)=r(s_t,a_t)+\gamma V^\pi_\phi(s_{t'},a_{t'})-V^\pi(s_t)$.
一般$\gamma$取0.99最佳。
**采用这个方法更新Policy Gradient的时候需要注意：**
应该采用
$$
\nabla_\theta J(\theta)=\frac{1}{N}\sum_{i=1}^n
\sum_{t=1}^T\nabla_\theta \log \pi_\theta(a_{i,t}|s_{i,t})(\sum_{t'=t}^T \gamma^{t'-t}r(s_{i,t'}|a_{i,t'}))$$
因为如果$\gamma$提取到前面的话相当于多计算了一些无关的状态。

- Batch actor-critic: 首先得到整个轨迹 batch， 然后再去更新policy。
- Online actor-critic: 每走一步action，都更新一次policy，但是需要多个步骤并行。（一般采用这个）

##### Design Decisions

有两种方法：
- 分别用两个预测$\pi_\theta(a|s)$和$V^\pi_\phi(s)$,好处是简单稳定
- 用同一个网络同时预测$\pi_\theta(a|s)$和$V^\pi_\phi(s)$,好处是能够利用shared features。

##### Improvement

但是在计算gradient的时候，采用actor-critic虽然能够减少variance，但是not unbiased；采用最原始的baseline，虽然无偏，但是会有更高的variance。

于是我们可以在计算Gradient对应的reward weight的时候，用Action模型跑出来一部分，然后再用我们的$\hat V$来预测。即：
$$
\nabla_\theta J(\theta) = \frac{1}{N}\sum_{i=1}^N\sum_{t=1}^T\nabla_\theta \log \pi_\theta(a_{i, t} | s_{i,t}))((\sum_{t'=t}^T\gamma^{t'-t}r(s_{i,t',a_{i,t'}}))-\hat{V}_\phi^\pi(s_{i,t}))
$$
由此我们引入n-step returns，也就是把后面 $n$ steps 直接action出来，剩下的通过预测 critic 来得到。
此时$\hat{A}_n^\pi(s_t,a_t)=\sum_{t'=t}^{t+n}\gamma^{t'-t}r(s_{t'},a_{t'})-\hat{V}_\phi^\pi(s_t)+\gamma^n\hat{V}^\pi_\phi(s_{t+n})$
如果我们求一个比较极端的情况，对所有$n$的取值求平均，那么实际上可以推导得出来和 discount 会具有同样的效应。

### Off-Policy RL

