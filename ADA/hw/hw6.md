# 算分第6次作业

郭劲豪 2200013146@stu.pku.edu.cn

--------------

## 第六章

### 6.2

我们设第$i$种咖啡豆使用$x_i$千克，那么有：
$$
\min w = 20x_1+28x_2+18x_3 \\
x_1 \le 500 \\
x_2 \le 600 \\
x_3 \le 400 \\
0.075x_1 + 0.085x_2 + 0.06x_3 \ge 75 \\
0.086x_1 + 0.088x_2 + 0.075x_3 \ge 80 \\
x_1,x_2,x_3\ge 0
$$


### 6.4（5）

 <img src="E:\University\2Spring\算分\微信图片_20240401211150.jpg" alt="微信图片_20240401211150" style="zoom:33%;" />

画出图后可知，不存在可行解。

### 6.6

#### (1)

<img src="E:\University\2Spring\算分\微信图片_20240401211203.jpg" alt="微信图片_20240401211203" style="zoom:33%;" />

画图后可知，最优解为$x_1=5,x_2=4.5,z=2x_1+x_2=14.5$

#### (2)

标准形为：
$$
\begin{aligned}
&\min z=-2x_1-x_2\\
\text{s.t  } &-x_1+2x_2+x_3=4
\\
&x_1 + x_4 = 5
\\
&x_1, x_2, x_3, x_4\ge 0
\end{aligned}
$$
所以$A=\begin{bmatrix}-1\ 2\ 1\ 0 \\\ \ \  1 \ 0\ 0\ 1\end{bmatrix}$，所有的基：

- $B_1=(P_1,P_2)$，$x_1=5,x_2=4.5,x_3=0,x_4=0,z=-14.5$，是可行基，对应点(5,4.5)。
- $B_2=(P_1,P_3)$，$x_1=5,x_2=0,x_3=9,x_4=0,z=-10$，是可行基，对应点(5,0)。
- $B_3=(P_1,P_4)$，$x_1=-4,x_2=0,x_3=0,x_4=9$，是基，不是可行解.
- $B_4=(P_2,P_3)$，不是基。
- $B_5=(P_2,P_4)$，$x_1=0,x_2=2,x_3=0,x_4=5,z=-2$，是可行基，对应点(0,2)。
- $B_6=(P_3,P_4)$，$x_1=0,x_2=0,x_3=4,x_4=5,z=0$，是可行基，对应点(0,0)。

### 6.8

<img src="C:\Users\MLEAutoMaton\AppData\Roaming\Typora\typora-user-images\image-20240401233404908.png" alt="image-20240401233404908" style="zoom:33%;" />

由画图可知(**修正图中标点为(4,6)**)，得出来的解为$x_1=4,x_2=6$，此时$z=x_1+2x_2$取最大值$16$.

考虑单纯形法的计算过程：

先写成标准形：
$$
\begin{aligned}
\min\ \ \  &z=-x_1-2x_2\\
&-x_1+x_2+x_3 =4\\
&x_2+x_4 = 6\\
&x_1+x_2+x_5 = 10\\
&x_1-x_2+x_6 = 4\\
&\forall i=1,2,...,6 &x_i \ge 0
\end{aligned}
$$
<img src="C:\Users\MLEAutoMaton\AppData\Roaming\Typora\typora-user-images\image-20240402000032743.png" alt="image-20240402000032743" style="zoom:33%;" />

所以可以得到基本可行解对应的点为：(0,0),(0,4),(2,6),(4,6).

### 6.10(1)

先化成标准形：
$$
\begin{aligned}
\min\ \ \  &-2x_1+x_2-x_3 \\
&2x_1+x_2+x_4=10\\
&-4x_1-2x_2+3x_3+x_5 = 10\\
&x_1-2x_2+x_3+x_6=14 \\
x_j \ge 0, &j=1,2,3,4,5,6
\end{aligned}
$$
然后用单纯形法计算，表格如下：

<img src="C:\Users\MLEAutoMaton\AppData\Roaming\Typora\typora-user-images\image-20240402002024449.png" alt="image-20240402002024449" style="zoom:33%;" />

最后当$x_1=\frac{24}{5},x_2=\frac{2}{5},x_3=10$的时候，$z=-\frac{96}{5}$。

### 6.13

考虑设$x_1=\alpha,x_3=0$，那么$x_2=\alpha+2,x_4=3\alpha+10$.

然后此时$z=z_0=x_1-x_2=-2$，所以有无穷多个最优解。

### 6.14

对偶问题为：
$$
\min 6y_1 - 5y_2 -4y_3 \\
\begin{aligned}
\text{s.t.} &y_1-y_2+2y_3 \ge 3 \\
&y_1+2y_2+y_3 \ge -2 \\
&-y_1-y_2-3y_3 \ge 1 \\
&-y_1+y_3= 4 \\
&y_1,y_2 \ge 0,y_3任意
\end{aligned}
$$
