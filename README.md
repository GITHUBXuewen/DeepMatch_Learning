# DeepMatch
https://github.com/shenweichen/DeepMatch/blob/master/examples/colab_MovieLen1M_YoutubeDNN.ipynb


召回算法总结分为传统的算法和深度学习

推荐系统算法中有两个根本性的思路，一是用户和条目表征，二是特征交叉。在传统算法中，隐语义模型从用户和条目表征(Embedding, 或称为“嵌入”)出发，以用户表征向量与条目表征向量的乘(内积)表示用户对条目的喜好。而因子分解机(FM)则致力于解决特征交叉问题。在上一篇中的AutoRec则是首个使用深度学习从用户和条目表征的角度解决问题的方案。当然，其结构过于简单，没有应对特征交叉的能力，模型本身的非线性能力也不足。而NeuralCF[1]是第一个深度学习下的同时处理这两个问题的模型。

（1）传统模型名称
协同过滤
矩阵分解
逻辑回归
FM
FFM
GBDT+LR
LS-PLM


（2）深度学习模型
总结起来，有七个演变方向：
    a.改变神经网络的复杂程度：增加深度神经网络的层数和结构复杂度。
    b.丰富特征交叉方式：改变特征向量的交叉方式，如NeuralCF，PNN(Product-based Neural Network)。
    c.组合模型：组合两种不同特点、优势互补的网络，主要是指Wide&Deep及其后续各种改进模型如Deep&Cross、DeepFM等。
    d.FM模型的深度学习演化：对FM模型的各种改进，包括NFM(Neural Factorization Machine)使用神经网络提升FM二阶交叉部分的特征交叉能力、FNN(Factorization-machine supported Neural Network)利用FM的结果进行网络初始化、AFM(Attention neural Factorization Machine)在FM中引入注意力机制。 
    e.引入注意力机制：主要包括上述的AFM和DIN(Deep Interest Network， 深度兴趣网络)模型融合序列模型：使用序列模型模拟用户行为或用户兴趣的演化趋势，如DIEN(Deep Interest Evolution Network，深度兴趣进化网络)
    f.结合强化学习：主要是为了模型的在线学习和实时更新，包括DRN(Deep Reinforcement Learning Network, 深度强化学习网络)

AutoRec ：是首个使用深度学习从用户和条目表征的角度解决问题的方案。当然，其结构过于简单，没有应对特征交叉的能力
Deep Crossing   ：emb+多层网络   
NeuralCF   ：多层神经网络+输出层  代替矩阵分解的内积，加强模型表达能力  https://zhuanlan.zhihu.com/p/160158270 优：特征交叉组合  缺点 ：未使用特征信息
PNN：是将depp&crossing中的stack层换成乘积层=线性操作+乘积操作 平均池化   这对所有特征进行无差别的交叉、忽略了原始特征中包含的价值信息


Wide&Deep和FM 尝试使得特征交叉方式更加高效
wide&deep :CF/LR的记忆功能（共现频率）和泛化能力； 把单输入层的wide部分和经过多层干之际的Deep部分连接起来，一起输入最终的输出层。 wide（记忆能力）：善于处理大量稀疏的ID类特征，Deep：表达能力强、进行深层的特征交叉。  最终利用LR 将wide（交叉积cross）和Deep部分结合起来统一的模型。

FNN：用FM的意向量完成Embeddding层的初始化，并没有对神经网络结构进行调整。在wide&deep中的结构中，在进行embeding之前是有FM的权重进行emb参数的初始化，从而加速了embedding层的速度。在训练FM过程中，并没有对特征与进行划分，但是FNN模型中，特征被划分为特征域。因此每个特征域具有对应的embedding层并且每个特征与embedding的维度都应与FM隐向量维度保持一致。

DeepFM：2017年由哈工大&华为提出，使用FM替换Wide&Deep的Wide部分，加强浅层网络组合特征的能力。DeepFM的改进目的和Deep&Cross的目的是一致的，只是采用的手段不同。

XDeepFM：

NFM：将FM的神经网络化的尝试，FM、FFM都是二阶特征交叉

以上的方式已经穷尽了所有特征工程的可能，以后局限性非常小。
之后尝试注意力机制在推荐模型中的应用

AFM： deep部分，在NFM中不同的特征域的特征Embedding向量经过特征交叉池化层的交叉，将各个交叉特征向量进行“加和”，输入最后有剁成NN组成的输出层。
但是这个特征交叉池化层是一视同仁地交叉，消除了大量有价值的信息。AFM在两两特征交叉层和池化层之间引入注意力机制是通过在特征价差层和最终输出层之间加入注意力网络实现的。

DIN ： 用户特征和广告特征使用“激活单元”计算Vads的权重 ,求和  g( Vu-i  , Vads )  * Vads = Vuser
       DIN 输入层+embedding层+连接层+多层全连接NN + 输出层的整体结构

之前的注意力机制就是在数学 求和或者平均 变成了 加权求和或者加权平均。这与时间无关、与序列无关

（序列）
DIEN：使用DIN网络结构的升级了兴趣进化网络，使用序列模型模拟用户兴趣的进化过程，考虑时间（对最近购买行为的影响）和序列（购买趋势，转移率）的影响。 兴趣进化网络从下到上：行为序列层+兴趣抽取层（GRU）+（注意力机制的AUGRU）兴趣进化

MIND: 

（强化学习）
DRN:


--------------向量化---------------------
word2vec: 利用词序列，CBOW\Skip-Gram   负采样只计算采样的误差而不需要计算所有词的误差 所以提升训练速度，层级softmax不常用
item2vec ：word2vec用于推荐系统，利用历史行为序列
DSSM：物品塔+用户塔  其中物品塔：由之前的用户行为序列生成的one-hot序列 变为 包含多特征的、全面的物品特征向量

item2vec 只能利用序列型数据，卖你对大量的网络话数据--》Graph Embedding

DeepWalk：随机游走：随机选择起始点，得到物品序列+word2vec
Node2Vec：同质性（DFS深度优先遍历） 和 结构性（BFS宽度优先遍历）  ，通过节点之间的跳转概率控制BFS和DFS倾向性
EGES：graph embeding with information，基本思想是在Deepwalk生成的Graph Embedding基础上引入节点的其他属性信息作为冷启动时的补充信息，因此一个节点可能拥有多个Embedding向量，EGES将这多个向量加权平均后输入softmax层，学习出每个Embedding的权重

上述这些GraphEmbedding方法可以认为都是直推式学习(transductive learning), 即在固定的图上学习每个节点的Embedding，每次学习只考虑当前数据，不能适应图结构经常变化的情况，因为图结构发生变化后需要重新学习全图Embedding。而归纳学习(inductive learning)是学习在图上生成节点Embedding的方法，这一类中目前已经落地的是Pinsage算法：

GraphSage & PinSage：2017年斯坦福发表GraphSage论文，2018年斯坦福和Pinterest公司合作落地PinSage。GraphSage的核心是学习如何聚合节点的邻居特征生成当前节点的信息，学习到这样一个聚合函数之后，不管图结构和图信息如何变化，都可以通过当前已知各个节点的特征和邻居关系，生成节点的embedding。GraphSage算法主要由两个操作组成：Sample采样和Aggregate聚合。采样是为了避免全图计算，每次只计算部分节点的Embedding。聚合操作则是学习一个聚合函数。关于GraphSage和PinSage的详细讲解，

