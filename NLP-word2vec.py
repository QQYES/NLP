# coding=utf-8

import codecs

import gensim
import jieba
import numpy as np

novels = ["书剑恩仇录",
          "天龙八部",
          "碧血剑",
          "越女剑",
          "飞狐外传",
          "侠客行",
          "射雕英雄传",
          "神雕侠侣",
          "连城诀",
          "鸳鸯刀",
          "倚天屠龙记",
          "白马啸西风",
          "笑傲江湖",
          "雪山飞狐",
          "鹿鼎记"]

sentences = []

for novel in novels:
    print("处理：{}".format(novel))
    with codecs.open('novels/{}.txt'.format(novel), encoding="utf8") as f:
        sentences += [list(jieba.cut(line.strip())) for line in f]

model = gensim.models.Word2Vec(sentences,
                               size=100,
                               window=5,
                               min_count=5,
                               workers=4)
for k, s in model.most_similar(positive=["乔峰", "萧峰"]):
    print(k, s)

for k, s in model.most_similar(positive=["阿朱"]):
    print(k, s)

for k, s in model.most_similar(positive=["丐帮"]):
    print(k, s)

for k, s in model.most_similar(positive=["降龙十八掌"]):
    print(k, s)


def find_relationship(a, b, c):
    """
    返回 d
    a与b的关系，跟c与d的关系一样
    """
    d, _ = model.most_similar(positive=[c, b], negative=[a])[0]
    print("给定“{}”与“{}”，“{}”和“{}”有类似的关系".format(a, b, c, d))


find_relationship("段誉", "段公子", "乔峰")

# 情侣对
find_relationship("郭靖", "黄蓉", "杨过")
# 岳父女婿
find_relationship("令狐冲", "任我行", "郭靖")
# 非情侣
find_relationship("郭靖", "华筝", "杨过")

# 韦小宝
find_relationship("杨过", "小龙女", "韦小宝")
find_relationship("令狐冲", "盈盈", "韦小宝")
find_relationship("张无忌", "赵敏", "韦小宝")

find_relationship("郭靖", "降龙十八掌", "黄蓉")
find_relationship("武当", "张三丰", "少林")
find_relationship("任我行", "魔教", "令狐冲")

all_names = np.array(filter(lambda c: c in model, novel_names["天龙八部"]))
word_vectors = np.array(map(lambda c: model[c], all_names))

from sklearn.cluster import KMeans

N = 3

label = KMeans(N).fit(word_vectors).labels_

for c in range(N):
    print("\n类别{}：".format(c + 1))
    for idx, name in enumerate(all_names[label == c]):
        print(name, )
        if idx % 10 == 9:
            print()
    print()


import scipy.cluster.hierarchy as sch

Y = sch.linkage(word_vectors, method="ward")

_, ax = plt.subplots(figsize=(10, 40))

Z = sch.dendrogram(Y, orientation='right')
idx = Z['leaves']

ax.set_xticks([])
ax.set_yticklabels(all_names[idx],
                   fontproperties=font_yahei_consolas)
ax.set_frame_on(False)

plt.show()
