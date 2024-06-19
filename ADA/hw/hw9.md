# 算分第9次作业

郭劲豪 2200013146@stu.pku.edu.cn
--------------

## 9.5
### (1)
设我们判断图$(V,E)$中是否有至少有$k$个点的独立集，我们设$(V,\overline{E})$为原图的补图，对于两个点$u,v\in A$，若$A$是原图独立集$\Leftrightarrow$$(u,v)\notin E\Leftrightarrow (u,v)\in \overline{E}$，所以原图中的一个独立集$A$等价于补图中的一个团。
而构建补图是$O(|E|^2)$的，所以是一个多项式时间变换。
### (2)
我们判断图$(V,E)$中有至多$k$个点的顶点覆盖，设该集合为$A$，$\overline{A} = V - A$，$\forall (u,v) \in E, u\in A \text{ or } v\in A\Leftrightarrow \forall u',v'\in A',(u',v')\notin E$. 所以在多项式时间内可以将VC转换为独立集，再根据(1)的结论不难推出VC可以多项式时间变换内转换为团。
### (3)

我们不妨假设当前判断图$G=(V,E)$是否存在HC，那么我们构造一个长度为$|V|$的环作为图H。

- 若原图存在哈密顿回路$\Leftrightarrow$存在一个排列$p$使得$\forall (p_i,p_{i+1})\in E,(p_n,p_1)\in E$。那么我们构造$f:V\rightarrow V$，其中$f(i)=p_i$即可。

- 若$H$是$G$的子图同构，那么也就是说存在$f:V\rightarrow V$使得$(f(i),f(i+1))\in E $，那么我们只需要令$p=f$然后将其作为哈密顿回路的路径即可。

上述就是HC到子图同构的多项式时间变换。

### (4)

我们对于一个团问题：$G=(V,E)$，至少有$K$个节点，我们构造实例：

限制条件为$x_i+x_j<2$，其中$i\ne j,(i,j)\notin E$。

对于一个点集的子集$V'$，用$x$向量来表示选取状况:$x_i = \begin{cases}0, i\in V'\\1, i\notin V' \\\end{cases}$，$D=K$	。

- 若存在至少$K$个节点的一个团，那么很明显满足限制条件，这就是0-1整数规划的一个解。
- 如果$x$是0-1整数规划的一个解，我们只需要将$x_i=1$的元素全部放入团中即可。此时对于$x_i=x_j=1$的点，很明显$(i,j)\in E$，所以这就是一个团。

上述就是团到0-1整数规划的多项式时间变换。

## 9.10
证明： 考虑$\Pi' \in \text{NPC}$，有：$\forall \Pi'' \in \text{NP-Hard}, \Pi'' \le_p \Pi'$，又因为$\Pi' \le_p \Pi$，由传递性可知：$\forall \Pi'' \in \text{NP-Hard}, \Pi'' \le_p \Pi$

所以$\Pi \in \text{NP-Hard}$ , 又因为$\Pi \in NP$， 所以$\Pi \in \text{NPC}$

## 9.14
易证，划分问题$\in \text{NP}$。下证 子集和 $\le_p$ 划分问题：

考虑一个子集和问题$Y = \{A=\{x_1,x_2,\dots,x_n\},N\}$，如果$\sum_{i\in T}x_i = N$，我们不妨设$s=\sum_{i=1}^nx_i$，有$\sum_{i\notin T}  = s- N$
1. $s-N > N$，我们插入一个新的元素$x_{n+1} = s-2\times N$，在这个新的集合$U=\{x_1,x_2,\dots,x_n,x_{n+1}\}$上做划分问题得到集合$T'$和$U-'T$，不妨设$x_{n+1}\in T',\sum_{i\in T'}x_i - x_{n+1}= s-N - (s-2*N) = N$，也就是对应一个和为$N$的解。
2. $s-N < N$，我们插入一个新的元素$x_{n+1} = 2\times N -s$，类似情况1可以推出一个子集和的解。
3. $s-N = N$，我们在原问题给出的集合上做划分问题等价于对其做子集和。

所以综上 子集和 $\le_p$ 划分问题， 可以得出划分问题$\in \text{NPC}$

## 9.15
给出一个单射$f:V_1\rightarrow V$，我们对于$E_1$中的每一条边都去判断$(f(u),f(v))\in E$即可，这样子的算法复杂度是$|V_1|^2$，所以子图同构是NP的。又由9.5(3)可知，$\text{HC} \le_p$子图同构。所以我们可以得出：

子图同构是$NPC$的。

