一、前言<br><br>自然语言是人类智慧的结晶，自然语言处理是人工智能中最为困难的问题之一，而对自然语言处理的研究也是充满魅力和挑战的。 通过经典的斯坦福cs224n教程，让我们一起和自然语言处理共舞！也希望大家能够在NLP领域有所成就！<br><br><br>二、先修知识（学习的过程中可以遇到问题后再复习）<br><br>了解python基础知识了解高等数学、概率论、线性代数知识了解基础机器学习算法：梯度下降、线性回归、逻辑回归、Softmax、SVM、PAC（先修课程斯坦福cs229 或者周志华西瓜书）具有英语4级水平（深度学习学习材料、论文基本都是英文，一定要阅读英文原文，进步和提高的速度会加快！！！！）以上知识要求内容可在最下方的知识工具中查找<br><br><br>三、每周学习时间安排<br><br>每周具体学习时间划分为4个部分:<br><br>1部分安排周一到周二2部分安排在周四到周五3部分安排在周日4部分作业是本周任何时候空余时间周日晚上提交作业运行截图周三、周六休息^_^<br><br><br>（以下的部分链接在手机端无法正常显示，请复制链接到电脑浏览器打开）
<br><br><br><br>﻿课程资料：
<br><br>课程主页：&nbsp;https://web.stanford.edu/class/cs224n&nbsp;/<br><br>中文笔记：&nbsp;http://www.hankcs.com/nlp/cs224n-introduction-to-nlp-and-deep-learning.html

http://www.hankcs.com/tag/cs224n/<br><br>课程视频：&nbsp;https://www.bilibili.com/video/av30326868/?spm_id_from=333&nbsp;.788.videocard.0<br><br>实验环境推荐使用Linux或者Mac系统，以下环境搭建方法皆适用:<br><br>· Docker环境配置：&nbsp;https://github.com/ufoym/deepo<br><br>· 本地环境配置：&nbsp;https://github.com/learning511/cs224n-learning-camp/blob/master/environment.md<br><br>注册一个github账号：github.com<br><br>后续发布的一些project和exercise会在这个github下：<br><br>&nbsp;https://github.com/learning511/cs224n-learning-camp<br><br>重要的一些资源：<br><br>深度学习斯坦福教程：&nbsp;http://deeplearning.stanford.edu/wiki/index.php/UFLDL%E6%95%99%E7%A8%8B<br><br>廖雪峰python3教程：&nbsp;https://www.liaoxuefeng.com/article/001432619295115c918a094d8954bd493037b03d27bf9a9000<br><br>github教程：&nbsp;https://www.liaoxuefeng.com/wiki/0013739516305929606dd18361248578c67b8067c8c017b000<br><br>莫烦机器学习教程：&nbsp;http://morvanzhou.github.io/tutorials&nbsp;/<br><br>深度学习经典论文：&nbsp;https://github.com/floodsung/Deep-Learning-Papers-Reading-Roadmap<br><br>斯坦福cs229代码(机器学习算法python徒手实现)：&nbsp;https://github.com/nsoojin/coursera-ml-py<br><br>本人博客：&nbsp;https://blog.csdn.net/dukuku5038/article/details/82253966<br><br>知识工具<br><br>为了让大家逐渐适应英文阅读，复习材料我们有中英两个版本，但是推荐大家读英文<br><br>数学工具<br><br>斯坦福资料：<br><br>线性代数（链接地址：&nbsp;http://web.stanford.edu/class/cs224n/readings/cs229-linalg.pdf&nbsp;）概率论（链接地址：&nbsp;http://101.96.10.44/web.stanford.edu/class/cs224n/readings/cs229-prob.pdf&nbsp;）凸函数优化（链接地址：&nbsp;http://101.96.10.43/web.stanford.edu/class/cs224n/readings/cs229-cvxopt.pdf&nbsp;）随机梯度下降算法（链接地址：&nbsp;http://cs231n.github.io/optimization-1&nbsp;/）<br><br>中文资料：<br><br>机器学习中的数学基本知识（链接地址：&nbsp;https://www.cnblogs.com/steven-yang/p/6348112.html&nbsp;）统计学习方法（链接地址：&nbsp;http://vdisk.weibo.com/s/vfFpMc1YgPOr&nbsp;）大学数学课本（从故纸堆里翻出来^_^）<br><br>编程工具<br><br>斯坦福资料：<br><br>Python复习（链接地址：&nbsp;http://web.stanford.edu/class/cs224n/lectures/python-review.pdf&nbsp;）TensorFlow教程（链接地址：&nbsp;https://github.com/open-source-for-science/TensorFlow-Course#why-use-tensorflow&nbsp;）<br><br>中文资料：<br><br>廖雪峰python3教程（链接地址：&nbsp;https://www.liaoxuefeng.com/article/001432619295115c918a094d8954bd493037b03d27bf9a9000&nbsp;）莫烦TensorFlow教程（链接地址：&nbsp;https://morvanzhou.github.io/tutorials/machine-learning/tensorflow&nbsp;/）

作业参考答案：http://www.hankcs.com/nlp/cs224n-assignment-1.html
达观杯：https://github.com/MLjian/TextClassificationImplement
# 第一周

##  <br>第1部分学习任务：
<br><br>（1）观看自然语言处理课学习绪论，了解深度学习的概括和应用案例以及训练营后续的一些学习安排<br><br>学习时长：10/23—10/28<br><br>绪论视频地址：&nbsp;https://m.weike.fm/lecture/10194068<br><br>（2）自然语言处理和深度学习简介，观看课件lecture01、视频1、学习笔记<br><br>学习时长：10/23<br><br>课件:&nbsp;lecture01（链接地址：&nbsp;https://github.com/learning511/cs224n-learning-camp/blob/master/lecture-notes/cs224n-2017-lecture1.pdf&nbsp;）观看视频1（链接地址：&nbsp;https://www.bilibili.com/video/av30326868/?spm_id_from=333&nbsp;.788.videocard.0）学习笔记：自然语言处理与深度学习简介（链接地址：&nbsp;http://www.hankcs.com/nlp/cs224n-introduction-to-nlp-and-deep-learning.html&nbsp;）

## <br>第2部分学习任务：
<br>（1）词的向量表示1，观看课件lecture02、视频2、学习笔记<br>学习时长：10/25—10/26<br>课件: lecture02（链接地址： https://github.com/learning511/cs224n-learning-camp/blob/master/lecture-notes/cs224n-2017-lecture2.pdf ）<br>观看视频2（链接地址： https://www.bilibili.com/video/av30326868/?p=2 ）<br>学习笔记：wordvecotor（链接地址： http://www.hankcs.com/nlp/word-vector-representations-word2vec.html ）<br>


## <br>第3部分学习任务：
<br>（1）论文导读：一个简单但很难超越的Sentence Embedding基线方法<br>学习时长：10/28<br>论文原文: paper（链接地址： https://github.com/learning511/cs224n-learning-camp/blob/master/paper/A%20Simple%20but%20Tough-to-beat%20Baseline%20for%20Sentence%20Embeddings.pdf ）<br>论文分析: highlight（链接地址： https://github.com/learning511/cs224n-learning-camp/blob/master/paper/highlight/cs224n-2017-lecture1-highlight.pdf ）<br>论文笔记：Sentence Embedding（链接地址： http://www.hankcs.com/nlp/cs224n-sentence-embeddings.html ）<br>

## <br>第4部分作业：
Assignment 1.1-1.2（链接地址： https://github.com/learning511/cs224n-learning-camp/blob/master/Assignmnet.md ）<br>1.1 Softmax 算法<br>1.2 Neural Network Basics 神经网络基础实现

# 第二周

## 第1部分学习任务：
<br>（1）高级词向量表示：word2vec 2，观看课件lecture03、视频3、学习笔记<br>学习时长：10/29—10/30<br>课件: lecture03（链接地址： https://github.com/learning511/cs224n-learning-camp/blob/master/lecture-notes/cs224n-2017-lecture3.pdf ）<br>观看视频3（链接地址： https://www.bilibili.com/video/av30326868/?p=3 ）<br>学习笔记：word2vec 2（链接地址： http://www.hankcs.com/nlp/cs224n-advanced-word-vector-representations.html ）<br>


## <br>第2部分学习任务：
<br>（1）Word Window分类与神经网络，观看课件lecture04、视频4、学习笔记<br>学习时长：11/1—11/2<br>课件: lecture04（链接地址： https://github.com/learning511/cs224n-learning-camp/blob/master/lecture-notes/cs224n-2017-lecture4.pdf ）<br>观看视频4（链接地址： https://www.bilibili.com/video/av30326868/?p=4 ）<br>学习笔记：Word Window分类与神经网络（链接地址： http://www.hankcs.com/nlp/cs224n-word-window-classification-and-neural-networks.html ）



## <br>第3部分学习任务：
<br>（1）论文导读：词语义项的线性代数结构与词义消歧<br>学习时长：11/4<br>论文原文: paper（链接地址： https://github.com/learning511/cs224n-learning-camp/blob/master/paper/Linear%20Algebraic%20Structure%20of%20Word%20Senses%2C%20with%20Applications%20to%20Polysemy.pdf ）<br>论文分析: highlight（链接地址： https://github.com/learning511/cs224n-learning-camp/blob/master/paper/highlight/cs224n-2017-lecture2-highlight.pdf ）<br>论文笔记：Sentence Embedding（链接地址： http://www.hankcs.com/nlp/cs224n-word-senses.html ）

## <br>第4部分作业：
Assignment 1.3-1.4（链接地址： https://github.com/learning511/cs224n-learning-camp/blob/master/Assignmnet.md ）<br>1.3 word2vec 实现<br>1.4 Sentiment Analysis 情绪分析

# 第三周
(因为中间跳过一周，11/4-11/10，所以本周从11/12开始)
## 第1部分学习任务：
（1）反向传播与项目指导：Backpropagation and Project Advice，观看课件lecture05、视频5、学习笔记<br>学习时长：11/12—11/13<br>课件: lecture05（链接地址： https://github.com/learning511/cs224n-learning-camp/blob/master/lecture-notes/cs224n-2017-lecture5.pdf ）<br>观看视频5（链接地址： https://www.bilibili.com/video/av30326868/?p=5 ）<br>学习笔记：反向传播与项目指导（链接地址： http://www.hankcs.com/nlp/cs224n-backpropagation-and-project-advice.html ）


## 第2部分学习任务：
<br>（1）依赖解析：Dependency Parsing，观看课件lecture06、视频6、学习笔记<br>学习时长：11/15—11/16<br>课件: lecture06（链接地址： https://github.com/learning511/cs224n-learning-camp/blob/master/lecture-notes/cs224n-2017-lecture3.pdf ）<br>观看视频6（链接地址： https://www.bilibili.com/video/av30326868/?p=6 ）<br>学习笔记：句法分析和依赖解析（链接地址： http://www.hankcs.com/nlp/cs224n-dependency-parsing.html ）

## 第3部分学习内容
<br>（1）论文导读：高效文本分类<br>学习时长：11/18<br>论文原文: paper（链接地址： https://github.com/learning511/cs224n-learning-camp/blob/master/paper/Bag%20of%20Tricks%20for%20Efficient%20Text%20Classification.pdf ）<br>论文分析: highlight（链接地址： https://github.com/learning511/cs224n-learning-camp/blob/master/paper/highlight/cs224n-2017-lecture3-highlight.pdf ）<br>论文笔记：高效文本分类（链接地址： http://www.hankcs.com/nlp/cs224n-bag-of-tricks-for-efficient-text-classification.html ）

## <br>第4部分作业：
Assignment 2.2（链接地址： https://github.com/learning511/cs224n-learning-camp/blob/master/Assignmnet.md ）<br> Neural Transition-Based Dependency Parsing 基于神经网络的依赖分析
# 第四周
## 第1部分学习任务：
（1）TensorFlow入门，观看课件lecture07、视频、学习笔记<br>学习时长：11/19—11/20<br>课件: lecture07（链接地址： https://github.com/learning511/cs224n-learning-camp/blob/master/lecture-notes/cs224n-2017-lecture7-tensorflow.pdf ）<br>观看视频7（链接地址： https://www.bilibili.com/video/av30326868/?p=7 ）<br>学习笔记：TensorFlow（链接地址： http://www.hankcs.com/nlp/cs224n-tensorflow.html ）
## 第2部分学习任务：
第2部分学习任务：<br>（1）RNN和语言模型，观看课件lecture08、视频、学习笔记<br>学习时长：11/22—11/23<br>课件: lecture08（链接地址： https://github.com/learning511/cs224n-learning-camp/blob/master/lecture-notes/cs224n-2017-lecture8.pdf ）<br>观看视频8（链接地址： https://www.bilibili.com/video/av30326868/?p=8 ）<br>学习笔记：TensorFlow（链接地址： http://www.hankcs.com/nlp/cs224n-rnn-and-language-models.html ）
## 第3部分学习任务：
（1）论文导读：词嵌入对传统方法的启发<br>学习时长：11/25<br>论文原文: paper（链接地址： https://github.com/learning511/cs224n-learning-camp/blob/master/paper/Impoving%20distributional%20similarly%20with%20lessons%20learned%20from%20word%20embeddings.pdf ）<br>论文分析: highlight（链接地址： https://github.com/learning511/cs224n-learning-camp/blob/master/paper/highlight/cs224n-2017-lecture4-highlight.pdf ）<br>论文笔记：词嵌入对传统方法的启发（链接地址： http://www.hankcs.com/nlp/cs224n-improve-word-embeddings.html ）

## 第4部分作业：
Assignment 2.1，2.2（链接地址： https://github.com/learning511/cs224n-learning-camp/blob/master/Assignmnet.md ）<br>2.1Tensorflow Softmax 基于TensorFlow的softmax分类<br>2.2 Neural Transition-Based Dependency Parsing 基于神经网络的依赖分析

# 第五周
## 第1部分学习任务：
（1）高级LSTM及GRU：LSTM and GRU，观看课件lecture09、视频、学习笔记<br><br>学习时长：11/26—11/27<br><br>课件:&nbsp;lecture09（链接地址：&nbsp;https://github.com/learning511/cs224n-learning-camp/blob/master/lecture-notes/cs224n-2017-lecture9.pdf&nbsp;）观看视频9（链接地址：&nbsp;https://www.bilibili.com/video/av30326868/?p=9&nbsp;）学习笔记：高级LSTM及GRU（链接地址：&nbsp;http://www.hankcs.com/nlp/cs224n-mt-lstm-gru.html&nbsp;）
## 第2部分学习任务：
（1）期中复习，观看课件和视频、回顾上一阶段学习的知识<br>学习时长：11/29—11/30<br>课件: （链接地址： https://github.com/learning511/cs224n-learning-camp/blob/master/lecture-notes/cs224n-midterm-review.pdf ）<br>观看视频（链接地址： https://www.youtube.com/watch?v=2DYxT4OMAmw&list=PL3FW7Lu3i5Jsnh1rnUwq_TcylNr7EkRe6&index=10 ）
## 第3部分学习任务：
（1）论文导读：基于转移的神经网络句法分析的结构化训练<br><br>学习时长：12/2<br><br>论文原文:&nbsp;paper（链接地址：&nbsp;https://github.com/learning511/cs224n-learning-camp/blob/master/paper/Structured%20Training%20for%20Neural%20Network%20Transition-Based%20Parsing.pdf&nbsp;）论文分析:&nbsp;highlight（链接地址：&nbsp;https://github.com/learning511/cs224n-learning-camp/blob/master/paper/highlight/cs224n-2017-lecture6-highlight.pdf&nbsp;）论文笔记：基于神经网络句法分析的结构化训练（链接地址：&nbsp;http://www.hankcs.com/nlp/cs224n-syntaxnet.html&nbsp;）
## 第4部分作业：
Assignment 2.3（链接地址：&nbsp;https://github.com/learning511/cs224n-learning-camp/blob/master/Assignmnet.md&nbsp;）<br>Recurrent Neural Networks: Language Modeling 循环神经网络语言建模

# 第六周
## 第1部分学习任务：
（1）机器翻译、序列到序列、注意力模型，观看课件lecture10、视频、学习笔记<br>学习时长：12/3—12/4<br>课件: lecture10（链接地址： https://github.com/learning511/cs224n-learning-camp/blob/master/lecture-notes/cs224n-2017-lecture10.pdf ）<br>观看视频9（链接地址： https://www.bilibili.com/video/av30326868/?p=10 ）<br>学习笔记：机器翻译、序列到序列、注意力模型（链接地址： http://www.hankcs.com/nlp/cs224n-9-nmt-models-with-attention.html ）


## 第2部分学习任务：
（1）GRU和NMT的进阶，观看课件lecture11、视频、学习笔记<br>学习时长：12/6—12/7<br>课件: lecture11（链接地址： https://github.com/learning511/cs224n-learning-camp/blob/master/lecture-notes/cs224n-2017-lecture11.pdf ）<br>观看视频9（链接地址： https://www.bilibili.com/video/av30326868/?p=11 ）<br>学习笔记：GRU和NMT的进阶（链接地址： http://www.hankcs.com/nlp/cs224n-gru-nmt.html ）

## 第3部分学习任务：
（1）论文导读：谷歌的多语种神经网络翻译系统<br>学习时长：12/9<br>论文原文: paper（链接地址： https://github.com/learning511/cs224n-learning-camp/blob/master/paper/Google%E2%80%99s%20Multilingual%20Neural%20Machine%20Translation%20System_%20Enabling%20Zero-Shot%20Translation.pdf ）<br>论文分析: highlight（链接地址： https://github.com/learning511/cs224n-learning-camp/blob/master/paper/highlight/cs224n-2017-lecture8-highlight.pdf ）<br>论文笔记：基于神经网络句法分析的结构化训练（链接地址： http://www.hankcs.com/nlp/cs224n-google-nmt.html ）
## 第4部分作业：
Assignment 3.1（链接地址： https://github.com/learning511/cs224n-learning-camp/blob/master/Assignmnet.md ）<br>A window into named entity recognition（NER）基于窗口模式的名称识别


# 第七周
## 第1部分学习任务：
（1）语音识别的end-to-end模型，观看课件lecture12、视频、学习笔记<br>学习时长：12/10—12/11<br>课件: lecture12（链接地址： https://github.com/learning511/cs224n-learning-camp/blob/master/lecture-notes/cs224n-2017-lecture12.pdf ）<br>观看视频12（链接地址： https://www.bilibili.com/video/av30326868/?p=12 ）<br>学习笔记：语音识别的end-to-end模型（链接地址： http://www.hankcs.com/nlp/cs224n-end-to-end-asr.html ）

## 第2部分学习任务：
（1）卷积神经网络:CNN，观看课件lecture13、视频、学习笔记<br>学习时长：12/13—12/14<br>课件: lecture13（链接地址： https://github.com/learning511/cs224n-learning-camp/blob/master/lecture-notes/cs224n-2017-lecture13.pdf ）<br>观看视频13（链接地址： https://www.bilibili.com/video/av30326868/?p=13 ）<br>学习笔记：卷积神经网络（链接地址： http://www.hankcs.com/nlp/cs224n-convolutional-neural-networks.html ）

## 第3部分学习任务：
（1）论文导读：读唇术<br>学习时长：12/16<br>论文原文: paper（链接地址： https://github.com/learning511/cs224n-learning-camp/blob/master/paper/Lip%20Reading%20Sentences%20in%20the%20Wild.pdf ）<br>论文分析: highlight（链接地址： https://github.com/learning511/cs224n-learning-camp/blob/master/paper/highlight/cs224n-2017-lecture9-highlight.pdf ）<br>论文笔记：读唇术（链接地址： http://www.hankcs.com/nlp/cs224n-lip-reading.html ）

## 第4部分作业：
Assignment 3.2（链接地址： https://github.com/learning511/cs224n-learning-camp/blob/master/Assignmnet.md ）<br>Recurrent neural nets for named entity recognition(NER) 基于RNN的名称识别

# 第八周
## 第1部分学习任务：
（1）Tree RNN与短语句法分析，观看课件lecture14、视频、学习笔记<br>学习时长：12/17—12/18<br>课件: lecture14（链接地址： https://github.com/learning511/cs224n-learning-camp/blob/master/lecture-notes/cs224n-2017-lecture14.pdf ）<br>观看视频14（链接地址： https://www.bilibili.com/video/av30326868/?p=14 ）<br>学习笔记：Tree RNN与短语句法分析（链接地址： http://www.hankcs.com/nlp/cs224n-tree-recursive-neural-networks-and-constituency-parsing.html ）

## 第2部分学习任务：
（1）指代消解，观看课件lecture15、视频、学习笔记<br>学习时长：12/20—12/21<br>课件: lecture15（链接地址： https://github.com/learning511/cs224n-learning-camp/blob/master/lecture-notes/cs224n-2017-lecture15.pdf ）<br>观看视频15（链接地址： https://www.bilibili.com/video/av30326868/?p=15 ）<br>学习笔记：指代消解（链接地址： http://www.hankcs.com/nlp/cs224n-coreference-resolution.html ）

## 第3部分学习任务：
（1）论文导读：谷歌的多语种神经网络翻译系统<br>学习时长：12/23<br>论文原文: paper（链接地址： https://github.com/learning511/cs224n-learning-camp/blob/master/paper/Character-Aware%20Neural%20Language%20Models.pdf ）<br>论文分析: highlight（链接地址： https://github.com/learning511/cs224n-learning-camp/blob/master/paper/highlight/cs224n-2017-lecture10-highlight.pdf ）<br>论文笔记：Character-Aware神经网络语言模型（链接地址： http://www.hankcs.com/nlp/cs224n-character-aware-neural-language-models.html ）
## 第4部分作业：
Assignment 3.3（链接地址： https://github.com/learning511/cs224n-learning-camp/blob/master/Assignmnet.md ）<br>3Grooving with GRUs(（NER）基于GRU的名称识别

# 第九周
## 第1部分学习任务：
DMN与问答系统，观看课件lecture16、视频、学习笔记<br><br>学习时长：12/24—12/25<br><br>课件:&nbsp;lecture16（链接地址：&nbsp;https://github.com/learning511/cs224n-learning-camp/blob/master/lecture-notes/cs224n-2017-lecture16.pdf&nbsp;）观看视频16（链接地址：&nbsp;https://www.bilibili.com/video/av30326868/?p=16&nbsp;）学习笔记：DMN与问答系统（链接地址：&nbsp;http://www.hankcs.com/nlp/cs224n-dmn-question-answering.html&nbsp;）

## 第2部分学习任务：
NLP存在的问题与未来的架构，观看课件lecture17、视频、学习笔记<br>学习时长：12/27—12/28<br>课件: lecture17（链接地址： https://github.com/learning511/cs224n-learning-camp/blob/master/lecture-notes/cs224n-2017-lecture17.pdf ）<br>观看视频17（链接地址： https://www.bilibili.com/video/av30326868/?p=17 ）<br>学习笔记：指代消解（链接地址： http://www.hankcs.com/nlp/cs224n-nlp-issues-architectures.html ）

## 第3部分学习任务：
（1）论文导读：谷歌的多语种神经网络翻译系统<br>学习时长：12/30<br>论文原文: paper（链接地址： https://github.com/learning511/cs224n-learning-camp/blob/master/paper/Learning%20Program%20Embeddings%20to%20Propagate%20Feedback%20on%20Student%20Code.pdf ）<br>论文分析: highlight（链接地址： https://github.com/learning511/cs224n-learning-camp/blob/master/paper/highlight/cs224n-2017-lecture12-highlight.pdf ）<br>论文笔记：学习代码中的语义（链接地址： http://www.hankcs.com/nlp/cs224n-program-embeddings.html ）
## 第4部分作业：
Assignment 3.3（链接地址： https://github.com/learning511/cs224n-learning-camp/blob/master/Assignmnet.md ）<br>3Grooving with GRUs(（NER）基于GRU的名称识别

# 第十周
## 第1部分学习任务：
挑战深度学习与自然语言处理的极限，观看课件lecture18、视频、学习笔记<br><br>学习时长：12/31—1/1<br><br>课件:&nbsp;lecture18（链接地址：&nbsp;https://github.com/learning511/cs224n-learning-camp/blob/master/lecture-notes/cs224n-2017-lecture18.pdf&nbsp;）观看视频18（链接地址：&nbsp;https://www.bilibili.com/video/av30326868/?p=18&nbsp;）学习笔记：挑战深度学习与自然语言处理的极限（链接地址：&nbsp;http://www.hankcs.com/nlp/cs224n-tackling-the-limits-of-dl-for-nlp.html&nbsp;）

## 第2、3部分学习任务：
（1）论文导读：neural-turing-machines<br>学习时长：1/3—1/6<br>论文原文: paper（ https://github.com/learning511/cs224n-learning-camp/blob/master/paper/Deep%20Reinforcement%20Learning%20for%20Dialogue%20Generation.pdf ）<br>论文分析: highlight（ https://github.com/learning511/cs224n-learning-camp/blob/master/paper/highlight/cs224n-2017-lecture14-highlight.pdf ）<br>论文笔记：neural-turing-machines（ http://www.hankcs.com/nlp/cs224n-neural-turing-machines.html ）<br><br>(2)论文导读： 深度强化学习用于对话生成<br>学习时长：1/3—1/6<br>论文原文: paper（ https://github.com/learning511/cs224n-learning-camp/blob/master/paper/Deep%20Reinforcement%20Learning%20for%20Dialogue%20Generation.pdf ）<br>论文分析: highlight（ https://github.com/learning511/cs224n-learning-camp/blob/master/paper/highlight/cs224n-2017-lecture11-highlight.pdf ）<br>论文笔记：深度强化学习用于对话生成（ http://www.hankcs.com/nlp/cs224n-deep-reinforcement-learning-for-dialogue-generation.html ）<br><br><br> 

# 第十一周
二、分节学习内容<br><br>（1）论文导读：图像对话<br><br>学习时长：1/7—1/13<br><br>论文原文:&nbsp;paper（&nbsp;https://github.com/learning511/cs224n-learning-camp/blob/master/paper/highlight/cs224n-2017-lecture5-highlight.pdf&nbsp;）论文分析:&nbsp;highlight（&nbsp;https://github.com/learning511/cs224n-learning-camp/blob/master/paper/Visual%20Dialog.pdf&nbsp;）论文笔记：图像对话（&nbsp;http://www.hankcs.com/nlp/cs224n-visual-dialog.html&nbsp;）<br><br>（2）比赛复盘：对之前的比赛进行回顾<br><br>（3）课程总结：输出自己的笔记内容

---

# 达观杯比赛
## 1.观看达观杯NLP算法大赛报名指导PDF和入门指导视频
<br>学习时长：10/25—11/4<br>零基础1小时完成一场AI比赛<br>达观杯文本智能挑战赛入门指导（视频在下方，如果不清楚也可以去荔枝微课看 https://m.weike.fm/lecture/10195400 ，密码是011220）<br> <br>[02零基础1小时完成一场AI比赛.pdf](http://p2.dcsapi.com/c2ZtYnVqd2ZRYnVpJD4kMzEyOTAyMzA0MjBOVWh5TmtOeU9rRnhOa0ozT3tGeS9pdW5tJCckd2pmeCQ+JDMxMjkwMjMwNDIwTlVoeU5rTnlPa0Z4TmtKM097RnkvaXVubSQnJHVqbmYkPiQyNjU3OjY5MzEzMTY4JCckdXpxZiQ+JDI1)<br>2018.10.22 <br>03 达观杯文本智能挑战赛.mp4 2018.10.22


## 2.观看达观杯NLP算法大赛进阶指导视频
学习时长：11/19—12/2<br>达观杯文本智能挑战赛进阶指导（视频在下方，如果不清楚也可以去荔枝微课看 https://m.weike.fm/lecture/10726829 ，密码是011220）<br>04达观杯之文本分类任务解析与代码使用(进阶指导).mp4 2018.11.18



