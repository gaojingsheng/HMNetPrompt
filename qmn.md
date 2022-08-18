# 英文会议摘要模型

## 背景介绍

会议摘要(meeting summarization)任务属于多人长对话场景下的文本摘要；就是在对话形式的会议记录基础上经过加工、整理出来的一种记叙性和介绍性的正式化文本。包括会议的基本情况、主要精神及中心内容。

## 模型整体结构

此摘要模型基于微软在论文《A Hierarchical Network for Abstractive Meeting Summarization with Cross-Domain Pretraining》中提出的HMNet模型构建的，总体依然可以分为字符等级编码器（Word-level Encoder, W-Encoder）,话语等级编码器（Utterance-level Encoder, U-Encoder），解码器（Decoder）三部分，在此基础上添加融合对话角色标签的融合器层以及融合主题分布信息的主题分割模块，均位于话语等级编码器部分。模型整体结构如下：

<img src="C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20220224121028169.png" alt="image-20220224121028169" style="zoom:150%;" />

## 模型相关代码介绍

模型相关代码全部保存在HMNet文件夹下面，下面分别介绍各部分的功能。

### DataLoader文件夹：

微软独特设计的InfinibatchLoader方式，这个建议不要修改

### ExampleRawData文件夹：

存放适应本模型的预训练以及微调输入数据

format.py:刘浩编写的如何将原始对话数据转换为模型能够接收的标准格式代码

meeting_summarization文件夹：存放已经处理好了的各项数据

——AMI/ICSI_proprec文件夹：原HMNet使用的下游微调数据集

——AMI/ICSI_topic文件夹：添加了主题分割标注的下游微调数据集

——Mediasum/CNNDM_proprec文件夹：预训练所涉及到的数据集

——role_dict_ms.json：角色标签的查询字典，用于制造角色向量

——ernie_config.json：ernie的融合器层里面的超参数配置

### Utils文件夹：

模型其余各模块所需要的配置文件保存在这里，外面的那些.py文件建议不要修改

HMNet文件夹：保存各种导入预训练以及微调数据方式的DataLoader方法，极为重要！！

——InfinibatchLoader.py：适用于正常进行摘要生成任务场景，可以在CNN/DM以及AMI/ICSI数据集上施行。

——InfinibatchLoader_topicseg.py：适用于正常进行摘要生成任务场景，添加了主题分割信息的导入；适用于主题分割任务场景，可以用来训练主题分割模块。

——InfinibatchLoader_rouge_topicseg.py：适用GSG预训练任务中使用Rouge方法抽取原文关键句的预训练场景

——InfinibatchLoader_textrank_topicseg.py：适用GSG预训练任务中使用Textrank方法抽取原文关键句的预训练场景

——InfinibatchLoader_mmr_topicseg.py：适用GSG预训练任务中使用MMR方法抽取原文关键句的预训练场景

——sumar文件夹：解释Textrank方法以及MMR方法中如何抽取原文关键句

### Models文件夹：

Networks文件夹：这里存放着不同情况下的模型，有_cpu后缀的适用于GPU不够用，只能在CPU上跑的情况（推理的时候经常会发生这种情况）

——MeetingNet_Transformer_ernie.py：我们的模型结构

——MeetingNet_Transformer_ernie_topicseg.py：在基础模型结构上添加主题分割向量

——MeetingNet_Transformer_ernie_topicseg_all.py：在基础模型结构上添加主题分割向量,添加注意力约束模块

——TopicsegNet_Transformer_ernie.py：主题分割模块，适用于主题分割任务场景，生成主题分割标签

——crf.py：主题分割模块中的crf层

——Transformer.py：存放着组成模型的每个基础模块，Attention,LN，FFN层等等

——Layers.py：各层使用的dropout算法

Criteria文件夹：存放模型使用的Loss

Optimizers文件夹：存放模型使用的Optimizers策略

knowledge_bert文件夹：描述了如何构建ERNIE中的融合器层

Trainer文件夹：这里存放着如何将模型、Loss、Optimizer模块组合在一起训练或者推理的.py文件

——HMNetTrainer.py：适用于下游摘要场景

——HMNetTrainer_pretrain：适用于预训练（GSG+dRA）场景

——TopicsegTrainer.py：适用于主题分割任务场景，用来训练主题分割模块

——Tasks.py：相当于一个转换器，将Utils/HMNet文件夹下面不同的DataLoader方法与Evaluation方法组合

### Evaluation文件夹：

Rouge_1,Rouge_2,Rouge_L,Rouge_SU4等模型评价指标在这里定义

### ExampleConf文件夹：

保存模型启动的sh文件，以及模型训练与测试的结果

### ExampleInitModel文件夹：

——transfo-xl-wt103文件夹：存放着使用的Tokenizer方法，具体是使用Transformer-XL的

——Pretrained文件夹：预训练得到的最好ckpt文件

——AMI_Finetuned文件夹：在AMI数据集上微调得到的最好ckpt文件

——ICSI_Finetuned文件夹：在ICSI数据集上微调得到的最好ckpt文件

## 运行命令

预训练命令（以在CNN/DM上预训练伪摘要任务为例）

```
CUDA_VISIBLE_DEVICES="0,1" mpirun -np 2 --allow-run-as-root python PyLearn.py train conf_hmnet_pretrain_ernie_cnndm --pretraining
```

AMI微调训练命令(如果OOM就调小GRADIENT_ACCUMULATE_STEP参数，实在不行就使用CPU)

```
CUDA_VISIBLE_DEVICES="0,1" mpirun -np 2 --allow-run-as-root python PyLearn.py train ExampleConf/conf_hmnet_ernie_finetune_AMI
```

ICSI微调训练命令(如果OOM就调小GRADIENT_ACCUMULATE_STEP参数，实在不行就使用CPU)

```
CUDA_VISIBLE_DEVICES="0,1" mpirun -np 2 --allow-run-as-root python PyLearn.py train ExampleConf/conf_hmnet_ernie_finetune_ICSI
```

AMI测试命令

```
python PyLearn.py evaluate ExampleConf/conf_eval_ernie_AMI --no_cuda
```

ICSI测试命令

```
python PyLearn.py evaluate ExampleConf/conf_eval_ernie_ICSI --no_cuda
```

## 模型DEMO

为了上手方便，API采用网页+命令行方式构建

运行python demo.py，复制公共地址到阅览器中

打开后界面如下：

![image-20220312100149253](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20220312100149253.png)

输入对话文本，选择对应的模型，右边会出现生成的摘要。

