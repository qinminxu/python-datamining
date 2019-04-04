

```python
train_contents=[]
train_labels=[]
test_contents=[]
test_labels=[]
#  导入文件
import os
import io
start=os.listdir(r'text_classification-master/train')
for item in start:
    test_path='text_classification-master/test/'+item+'/'
    train_path='text_classification-master/train/'+item+'/'
    for file in os.listdir(test_path):
        with open(test_path+file,encoding="GBK") as f:
            test_contents.append(f.readline())
            #print(test_contents)
            test_labels.append(item)
    for file in os.listdir(train_path):
        with open(train_path+file,encoding='gb18030', errors='ignore') as f:
            train_contents.append(f.readline())
            train_labels.append(item)
print(len(train_contents),len(test_contents))
 
# 导入stop word
import jieba
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB  
stop_words = [line.strip() for line in io.open('text_classification-master/stop/stopword.txt','r',encoding='UTF-8').readlines()]
 
# 分词方式使用jieba,计算单词的权重
tf = TfidfVectorizer(tokenizer=jieba.cut,stop_words=stop_words, max_df=0.5)
train_features = tf.fit_transform(train_contents)
print(train_features.shape)
 
#模块 4：生成朴素贝叶斯分类器
# 多项式贝叶斯分类器
clf = MultinomialNB(alpha=0.001).fit(train_features, train_labels)
 
#模块 5：使用生成的分类器做预测
test_tf = TfidfVectorizer(tokenizer=jieba.cut,stop_words=stop_words, max_df=0.5, vocabulary=tf.vocabulary_)
test_features=test_tf.fit_transform(test_contents)
 
print(test_features.shape)
predicted_labels=clf.predict(test_features)
print(metrics.accuracy_score(test_labels, predicted_labels))
```

    3306 200
    

    D:\Anaconda3\lib\site-packages\sklearn\feature_extraction\text.py:301: UserWarning: Your stop_words may be inconsistent with your preprocessing. Tokenizing the stop words generated tokens [',', 'a', 'ain', 'aren', 'c', 'couldn', 'd', 'didn', 'doesn', 'don', 'hadn', 'hasn', 'haven', 'i', 'isn', 'll', 'm', 'mon', 's', 'shouldn', 't', 've', 'wasn', 'weren', 'won', 'wouldn', '下', '不可', '专', '东方', '中', '买', '使', '信', '倒', '储能', '儿', '前', '只', '唷', '啪', '喔', '图木', '外', '大面儿', '天', '好', '密', '尔', '市', '年', '愿', '手', '抗拒', '拉', '拟', '敞开', '新', '无', '样', '次', '毫无保留', '漫', '特', '理', '皆', '目前为止', '种', '美', '舒克市', '草', '设', '话', '说', '赶早', '赶晚', '达', '限', '项', '高', '龙', '\ufeff'] not in stop_words.
      'stop_words.' % sorted(inconsistent))
    

    (3306, 24581)
    

    D:\Anaconda3\lib\site-packages\sklearn\feature_extraction\text.py:301: UserWarning: Your stop_words may be inconsistent with your preprocessing. Tokenizing the stop words generated tokens [',', 'a', 'ain', 'aren', 'c', 'couldn', 'd', 'didn', 'doesn', 'don', 'hadn', 'hasn', 'haven', 'i', 'isn', 'll', 'm', 'mon', 's', 'shouldn', 't', 've', 'wasn', 'weren', 'won', 'wouldn', '下', '不可', '专', '东方', '中', '买', '使', '信', '倒', '储能', '儿', '前', '只', '唷', '啪', '喔', '图木', '外', '大面儿', '天', '好', '密', '尔', '市', '年', '愿', '手', '抗拒', '拉', '拟', '敞开', '新', '无', '样', '次', '毫无保留', '漫', '特', '理', '皆', '目前为止', '种', '美', '舒克市', '草', '设', '话', '说', '赶早', '赶晚', '达', '限', '项', '高', '龙', '\ufeff'] not in stop_words.
      'stop_words.' % sorted(inconsistent))
    

    (200, 24581)
    0.925
    
