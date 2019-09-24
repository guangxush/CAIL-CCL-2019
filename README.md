# CAIL_2019
## 参数说明
1. train: True训练、False测试(提交时必须改为False)
2. level: 'word'词级、'word_char'字词级
3. model_name: 'siamese_cnn'、'esim'、'att_cnn'
## 结果
|model         |mode        |level      |word_embedding   |fold     |Acc    |
|--------------|:----------:|:---------:|:---------------:|:-------:|:-----:|
|siamese_cnn   |margin_loss |word       |word2vec_300dim  |10       |88.04  |
|esim          |margin_loss |word       |word2vec_300dim  |10       |79.45  |
|bert          |margin_loss |char       |word2vec_300dim  |1        |76.99  |
|dpcnn         |margin_loss |word       |word2vec_300dim  |10       |70.55  |

## level2
|model         |mode        |level      |word_embedding   |fold     |offline|online |
|--------------|:----------:|:---------:|:---------------:|:-------:|:-----:|:-----:|
|preattcnn     |margin_loss |word       |word2vec_300dim  |10       |67.23  |65.27  |
|esim          |margin_loss |word       |word2vec_300dim  |10       |79.45  |66.47  |
