# DeepMatch
https://github.com/shenweichen/DeepMatch/blob/master/examples/colab_MovieLen1M_YoutubeDNN.ipynb
召回算法总结分为传统的算法和深度学习

------------------step1 :--------总结篇-推荐算法总结-------------------
https://yuancl.github.io/2019/03/26/rs/%E6%80%BB%E7%BB%93%E7%AF%87-%E6%8E%A8%E8%8D%90%E7%AE%97%E6%B3%95%E6%80%BB%E7%BB%93/

-------------step2:--------深度学习推荐--------------------


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
AutoRec ：是首个使用深度学习从用户和条目表征的角度解决问题的方案。当然，其结构过于简单，没有应对特征交叉的能力
Deep Crossing   ：emb+多层网络   
NeuralCF   ：多层神经网络+输出层  代替矩阵分解的内积，加强模型表达能力  https://zhuanlan.zhihu.com/p/160158270 优：特征交叉组合  缺点 ：未使用特征信息
PNN：是将depp&crossing中的stack层换成乘积层=线性操作+乘积操作 平均池化   这对所有特征进行无差别的交叉、忽略了原始特征中包含的价值信息


Wide&Deep和FM 尝试使得特征交叉方式更加高效
wide&deep :CF/LR的记忆功能（共现频率）和泛化能力； 把单输入层的wide部分和经过多层干之际的Deep部分连接起来，一起输入最终的输出层。 wide（记忆能力）：善于处理大量稀疏的ID类特征，Deep：表达能力强、进行深层的特征交叉。  最终利用LR 将wide（交叉积cross）和Deep部分结合起来统一的模型。

FNN：用FM的意向量完成Embeddding层的初始化，并没有对神经网络结构进行调整。在wide&deep中的结构中，在进行embeding之前是有FM的权重进行emb参数的初始化，从而加速了embedding层的速度。在训练FM过程中，并没有对特征与进行划分，但是FNN模型中，特征被划分为特征域。因此每个特征域具有对应的embedding层并且每个特征与embedding的维度都应与FM隐向量维度保持一致。

DeepFM：2017年由哈工大&华为提出，使用FM替换Wide&Deep的Wide部分，加强浅层网络组合特征的能力。DeepFM的改进目的和Deep&Cross的目的是一致的，只是采用的手段不同。
NFM：将FM的神经网络化的尝试，FM、FFM都是二阶特征交叉

以上的方式已经穷尽了所有特征工程的可能，以后局限性非常小。
之后尝试注意力机制在推荐模型中的应用

AFM： deep部分，在NFM中不同的特征域的特征Embedding向量经过特征交叉池化层的交叉，将各个交叉特征向量进行“加和”，输入最后有剁成NN组成的输出层。
但是这个特征交叉池化层是一视同仁地交叉，消除了大量有价值的信息。AFM引入注意力机制是通过在特征价差层和最终输出层之间加入注意力网络实现的。
DIN ： 用户特征和广告特征使用“激活单元”计算Vads的权重 ,求和  g( Vu-i  , Vads )  * Vads = Vuser

之前的注意力机制就是在数学 求和或者平均 变成了 加权求和或者加权平均。这与时间无关、与序列无关

DIEN：DIN的升级，使用序列模型模拟用户兴趣的进化过程，考虑时间（对最近购买行为的影响）和序列（购买趋势，转移率）的影响



-------------step3:--------文哥学习笔记--------------------
https://www.jianshu.com/u/c5df9e229a67       推荐系统遇上深度学习
浅梦的学习笔记 资料汇总   https://zhuanlan.zhihu.com/p/270918998 
召回匹配 <https://github.com/shenweichen/AlgoNotes#%E5%9B%BE%E7%AE%97%E6%B3%95> 
图算法  https://github.com/shenweichen/AlgoNotes#%E5%9B%BE%E7%AE%97%E6%B3%95
GraphEmbedding <https://github.com/shenweichen/GraphEmbedding> 
Graph Neural Network <https://github.com/shenweichen/GraphNeuralNetwork> 


