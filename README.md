# CAIL_2019

CAIL-CCL-2019人工智能大赛-相似案例匹配

## Requirement

Python 3.x

Keras 2.x

sklearn

numpy

gensim

## Files
- callbacks 回调函数swa
- data目录（保存处理后的数据，目前已gitignore）
- layers 定义神经网络层
- log 打印当前模型每一轮迭代的准确率
- data_preprocess 原始数据处理成词向量
- main 模型的训练与测试
- models 存储不同的子模型

## Run

代码运行如下：

- 模型训练：

  main 中的train = True，然后models选择合适的models名称，比如esim
  ```python main.py```

- 模型测试: 采用公开测试集产生测试结果（用于测试代码的正确性和了解模型效果）：

  main 中的train = True，overwrite = False，然后models选择合适的models名称，比如esim, 如果此时已经有了已训练好的模型，会load并输出结果
  ```python main.py```

- 模型提交：(模型将结果保存在output文件夹下)：

  main 中的train = False，此时会调用服务器测试数据进行输出
  ```python main.py```

## 参数说明
1. train: True训练、False测试(提交时必须改为False)
2. level: 'word'词级、'word_char'字词级
3. model_name: 'siamese_cnn'、'esim'、'dpcnn'
4. 这里的main已经对多个模型交叉验证并把输出结果集成

## 结果
## level1 初赛
|model         |mode        |level      |word_embedding   |fold     |Acc    |
|--------------|:----------:|:---------:|:---------------:|:-------:|:-----:|
|siamese_cnn   |margin_loss |word       |word2vec_300dim  |10       |88.04  |
|esim          |margin_loss |word       |word2vec_300dim  |10       |79.45  |
|bert          |margin_loss |char       |word2vec_300dim  |1        |76.99  |
|dpcnn         |margin_loss |word       |word2vec_300dim  |10       |70.55  |

## level2 复赛
|model         |mode        |level      |word_embedding   |fold     |offline|online |
|--------------|:----------:|:---------:|:---------------:|:-------:|:-----:|:-----:|
|esim          |margin_loss |word       |word2vec_300dim  |10       |79.45  |66.47  |
