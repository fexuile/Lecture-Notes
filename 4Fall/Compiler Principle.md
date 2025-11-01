# 前言
编译器 VS 解释器：
- 编译器是将源程序转换成可执行程序（先转换后执行）
- 解释器是在转换源程序的同时执行程序（边转换边执行）

编译器的组织结构：前端（从源程序到中间代码），后端（从中间代码到机器指令）
遍（Pass）：对源程序或者中间代码从头到尾扫描一次完成任务，这过程叫做pass。

# 词法分析
词法分析器：读入源程序，输出token序列（包括名字+类型）
token类别包括
- 保留字
- 标识符
- 字面常数
- 运算符
- 分解符

如何识别词法单元：
1. 状态转换图：**优先匹配长度最长的，长度相同的情况下匹配最先的**
2. LEX生成工具：lex源程序->C文件->可执行文件，匹配规则和状态转换图相同。

有限自动机：是识别器，只能对每个可能的输入串简单地回答 yes / no
DFA：确定的有限自动机，对于一个状态的一个输入只有可能有一个输出和后继状态。
NFA：不确定的有限自动机，对于一个状态一个输入应该是转移到状态的一个子集。NFA可以转移$\varepsilon$。
**DFA的表达能力与NFA是等价的，DFA是特殊的NFA**

从正则表达式到NFA：通过一个定式可以得到：
- |就走上下两条
- +就连在一起
- \*就采用一个循环的来表示
然后将正则表达式一一还原即可。

从NFA到DFA：通过子集构造法得到，每一步都走到$\varepsilon$闭包中。

DFA的化简：首先将集合分为终结态和非终结态，然后依次划分出与集合内转移不相同的，直到无法再划分。

Lex的做法：先将NFA合并，然后变成DFA化简，确定终结态。
> 注意：
> 1. 在实际操作中遍历DFA不应该是到达终结态就停止，而是一直读入字符直到达到不可能终结态，然后回退查找最长的。
> 2. 关于遍历DFA还是NFA，NFA的理由是DFA状态太多，DFA的理由是NFA复杂度不可控。

# 语法分析
识别符号串S是否为某个合法的语法单元。
种类具有：通用语法分析器、自顶向下（从根节点出发）和自底向上（从叶节点出发）。
后两种语法分析器**通常总是从左到右、逐个扫描词法单元**

## 文法
$G=(V_T,V_N,S,P)$，$V_T$表示终结符号集，$V_N$非终结符号符号集,$P$是产生式$\alpha\rightarrow \beta$，$S$是开始符号

## 上下文无关文法
Context-Free Grammar(CFG)
所有产生式的左边只有一个非终结符号，即$A-\rightarrow \beta$

直接推导: $\gamma\alpha\delta \Rightarrow \gamma\beta\delta$，倒过来叫做直接规约。
对于一个直接推导序列$a_0\Rightarrow a_1\Rightarrow\dots\Rightarrow a_n$，叫做$a_0\Rightarrow^+ a_n$，根据和正则表达式类似的方法可以定义$\Rightarrow^*$

### 语法分析树
句型：$S\Rightarrow^* \alpha$就称$\alpha$是$S$的句型。
句子：不包含非终结符的句型叫做句子。
语言：$L(G)$，文法$G$所有句子的集合。

可以通过推导序列 / 规约序列画出语法分析树。

### 二义性
1. 一个句子的结构可能不唯一
2. 一个句子的对应的分析树可能不唯一
如果一个文法中存在某个句子有两棵分析树, 那么该句子是二义性的。
如果一个文法存在二义性句子，那么就称这个文法是二义性文法

### 证明L(G)和L相同
证明文法生成的语言：
1. 首先证明L(G)属于L（按照推导序列长度进行数学归纳）
2. 然后证明L属于L(G) （按照符号串的长度来构造推导序列）

### 上下文无关文法和正则表达式
上下文无关文法比正则表达式的能力更强。
存在无法用正则表达式表达的上下文无关文法：$\{a^nb^n|n>0\}$，可以通过状态机的状态数证明；但是所有的正则表达式都可以用上下文无关文法表示，等价于NFA的每一个状态都是文法中的一个非终结符。
NFA接受一个句子实际上是文法推导出该句子的过程。
### 文法分类
- 任意文法（0型）：$\gamma\alpha\delta \rightarrow \gamma\beta\delta$
- 上下文有关文法（1型）：$\gamma A\delta\rightarrow \gamma\beta\delta$
- 上下文无关文法（2型）：$A\rightarrow \beta$
- 正则文法（3型）：
	- （右线性）：$A\rightarrow aB, A\rightarrow a$
	- （左线性）：$A\rightarrow Ba, A\rightarrow a$

## 文法设计
### 消除二义性
举例：为了保证“else和最近未匹配的then匹配”，在文法中引入"matched_stmt"

### 消除左递归
**直接左递归**：
$$
\begin{aligned}
A\rightarrow Aa|b
\end{aligned}
$$
可以变成
$$
\begin{aligned}
A\rightarrow bA' \\
A'\rightarrow aA' | \varepsilon \\
\end{aligned}
$$

**间接左递归：** 考虑将生成式带入，然后变成直接左递归再消除。

消除所有左递归的算法：将非终结符按照一定顺序排列，然后对于第$i$个终结符，每个形如$A_i \rightarrow A_jr(j<i)$的规则展开替换，然后消除其中的左递归。
> 不同的顺序可能导致不同的结果，但是他们本质是相同的。

**提取左公因子：**
$$
A\rightarrow \alpha\beta_1 | \alpha\beta_2
$$
变成
$$
\begin{aligned}
A\rightarrow &\alpha B \\
B\rightarrow &\beta_1 | \beta_2
\end{aligned}
$$

## 自顶向下的语法分析
自顶向下分析是从文法的开始符号出发，试构造出一个最左推导，从左至右匹配输入的单词串。
考虑当前到达非终结符$A$，且句子匹配到的字符为$a$，$A\rightarrow \alpha_1 |\dots| \alpha_k$，只有一个推导的首字符是$a$，所以可以直接构造最左推导。否则需要带回溯的进行尝试。

特点：
1. 带预测
2. 试探过程（需要回溯）
3. 可以通过编程实现，但是带回溯因为时间效率低应用少。

### 如何保证没有回溯
考虑寻找First和Follow集合：
#### First
$First(X)$表示产生式$X$推导出来的语句的第一个字符。
1. 考虑对于单个符号如何计算：
	- 终结符X,First(X)=X
	- 非终结符，存在推导式$X\rightarrow Y_1Y_2\dots Y_k$，需要考虑添加$First(Y_i)$和$\varepsilon$。
	- 非终结符，存在$X\rightarrow \varepsilon$。把$\varepsilon$放入First中。
2. 对于产生式右端$X_1X_2\dots X_n$，考虑依次枚举保证前面均为$\varepsilon$就可以添加当前的First。

#### Follow
$Follow(X)$表示紧跟在X后面的终结符。
1. 将\$符号添加到Follow(S)中
2. 迭代直到所有Follow集合都不变：
	- 考虑$A\rightarrow \alpha B\beta$，$First(\beta)$的所有非$\varepsilon$符号都属于Follow(B)
	- 考虑$A\rightarrow \alpha B$或$A\rightarrow \alpha B\beta$且$First(\beta)$中有$\varepsilon$，那么$Follow(A)$中所有符号均属于$Follow(B)$

### LL(1)文法
> LL(k)表示从左到右扫描（L），最左推导（L），往前看k个字符

对于文法中任意产生式：$A\rightarrow \alpha|\beta$，有：
- $First(\alpha)\cap First(\beta)=\emptyset$
- 若$\varepsilon\in First(\beta)$，有：$Follow(A)\cap First(\alpha)=\emptyset$，反之亦然

#### 预测分析表：
对于任何产生式$A\rightarrow \alpha$:
- 对于First($\alpha$)中的每个终结符号a，将$A\rightarrow \alpha$加入到 M[A,a] 中。
- 如果$\varepsilon$在First(A)中，那么对于Follow(A)中的每个符号b，将将$A\rightarrow\alpha$加入到 M[A,b] 中。

然后就可以根据预测分析表来进行语法分析了。

#### 非LL(1)文法
二义性 / 左递归都不是，也有不存在LL(k)文法的语言。
*左递归文法不适合自顶向下分析！*

### 错误恢复
目的是在一次分析中找到更多的语法错误，所以需要对错误进行恢复以进行之后的分析。
#### 恐慌模式
忽略输入中的一些符号，直到出现由设计者选定的某个同步词法单元
一般来说同步集合可以选择First(A)和Follow(A)中的所有符号。
遇到error忽略输入，遇到synch弹出非终结符。
#### 短语层次的恢复
在error处插入错误处理函数，需要确保不会无限循环。 

## 自底向上的语法分析
为一个输入串构造语法分析树的过程。本质上是“移进-归约”分析。
- LR：最大的可以构造出移进-归约语法分析器的语法类

实际上就是从一个句子到起始符号S的过程。

句柄：实际上就是最右推导中一个推导过程的右部分。
整个分析的过程就是不断地将输入的内容移进栈中，然后归约，就能够构造出一颗语法分析树。

### 移进-归约分析中的冲突
分为
- 归约 / 归约冲突：不知道应该使用什么归约
- 移入 / 归约冲突：不知道什么时候应该开始归约而不是继续移入字符。

### LR(0)分析
LR(k)分析，是一种常用的自底向上分析。
L指从左向右扫描输入符号串，R指的构造最右推导，k表示往前看的字符

#### LR(k) item
$A\rightarrow\alpha\cdot\beta, \gamma$是他的一个项，表示已经读完$\alpha$了，还没读$\beta$，$\gamma$用来判断句柄。
对于LR(0)而言，通常省略$\gamma$，可以将项分为：移进/待归约/归约/接受

#### Closure
如何求解Closure(I)：
1. 将I中的所有项加入到Closure中
2. 迭代直到不再变化：若$A\rightarrow \alpha\cdot B\beta \in I$，且$B\rightarrow \gamma\in P$，那么将$B\rightarrow \cdot\gamma$加入。
Closure的意义在于

内核项:$S'\rightarrow \cdot S$和所有分割点不在最左侧的项。
#### Goto
$Goto(I, X)$：项集$\{A\rightarrow \alpha X\cdot\beta|A\rightarrow \alpha\cdot X\beta \in I\}$的闭包。
实际上对应着移入了X后能够到哪些项集。

有了Closure和Goto函数就可以构造出来一个LR(0)的自动机，然后就可以进行LR(0)的移入归约。

LR(0)自动机需要注意的是在归约的时候退回的数目应该是推导式右端的符号数，也就是说$T\rightarrow T*F\cdot$就需要弹栈3次。
#### LR分析表(SLR分析表)
由action[s,a]和goto[s,s']构成。
action[s,a]可以表示归约$r_j$，也可以表示移进然后转移$s_j$.
goto对应的就是读入的如果是一个非终结符应该怎么移动。

判断移进/归约和归约/归约冲突：
考虑当前项集内有：$X\rightarrow \alpha\cdot b\beta, A\rightarrow \alpha\cdot, B\rightarrow \beta\cdot$，考虑下一个字符是a，则：
1. a=b,移入
2. $a\in Follow(A)$，归约A
3. $b\in Follow(B)$，归约B

活前缀（可行前缀）：某个右句型的前缀，且没有越过该句型的句柄的右端。
有效项：如果存在$S\rightarrow \alpha A\omega\rightarrow \alpha\beta_1\beta_2\omega$，那么就称项$A\rightarrow \beta_1\cdot\beta_2$对$\alpha\beta_1$有效。

问题在于如果follow(A)中的集合可能移进可能归约，那么就依旧无法处理冲突。

### LR(1)分析
在产生式后面增加1个向前搜索符号，考虑对于一个归约串$A\rightarrow \alpha\cdot, a_1a_2\dots a_k$，只有输入字符和向前搜索符号匹配才能够归约。
LR(1)有效项：存在推导$S\rightarrow \delta A\omega\rightarrow \delta\alpha\beta\omega$，则称$[A\rightarrow \alpha\cdot\beta, a]$对活前缀$\delta\alpha$有效，其中$\omega$为$\varepsilon$且a为\$，或$a\in First(\omega)$。

构造方法：
**Closure：**
设I是G的一个LR(1)项集，closure(I)是从I出发用下面三个规则构造的项集：
1. 每一个I中的项都属于closure(I)。 
2. 若项 $[A\rightarrow \alpha\cdot B\beta, a]$ 属于closure(I) 且 $B\rightarrow \gamma\in P$, 则对任何$b\in First(\beta a)$, 把 $[B→\cdot\gamma, b]$ 加到 closure(I)中。 
3. 重复执行(2)直到closure(I)不再增大为止。
**GOTO：** 与LR(0)相同

### LALR文法
**同心集：** 如果两个LR(1)项集去掉搜索符之后是相同的, 则称这两个项集具有相同的核心(core)。
合并同心集（合并搜索符串）后，像构造LR(1)分析表一样构造出的LR分析表称作是LALR(1)分析表
实际上可以理解为后面的搜索字符可以不唯一的LR(1)文法，也可以理解为增加了搜索字符的LR(0)项集。

## 总结
对于LL文法，不存在项这个概念，而是单纯只需要求first和follow即可。
对于LR文法，首先需要求出项集，然后考虑 归约/移进冲突 和 归约/归约冲突。
- 对于LR(0)自动机，如果不存在冲突就是LR(0)文法，否则如果可以通过follow集合判断，就是SLR文法。
- 对于LR(1)自动机，如果可以通过缩小到同心集然后也不存在冲突，那么就是LALR文法，否则如果没有冲突就是LR(1)文法。
总结来说$LL(k) \subset LR(k)$，然后$LR(0) \subset SLR \subset LALR \subset LR(1)$。

## 二义性语法
1. 可以通过优先级和结合性来对二义性文法使用LR分析
2. 也可以通过强制移入的办法来消除悬空-else二义性文法。

# 语法制导翻译