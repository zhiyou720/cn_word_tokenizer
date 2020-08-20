# cn_word_tokenizer
中文分词，jieba、thu、hanlp三种方式，停用词，用户词典，词性过滤等制作token

### 使用

#### 分词并标注词性

```python
from hanLP_tokenizer import hanLP_tokenize
from jieba_tokenizer import jieba_tokenize
from thu_tokenizer import thu_tokenize

_text = '今天又给大家送福利啦,现在加入腾讯汽车会员，享中石油、中石化实名充值九折卡，最高100万的驾乘意外险、违章代缴6折优惠等更多！'

jieba_token = jieba_tokenize(_text)
hanLP_token = hanLP_tokenize(_text)
thu_token = thu_tokenize(_text)

print('jieba result:')
print(jieba_token)

print('hanlp result:')
print(hanLP_token)

print('thu result:')
print(thu_token)

_text = '从君士坦丁堡陷落到地中海海洋混战，一直到威尼斯《海洋霸权》的史诗，其实这三本书的角度和视角并不一致，' \
        '但是总有一种主题和精神气质贯穿始终。基督文明和伊斯兰文明在地中海的碰撞，献上了交织着血与泪的史诗。'


jieba_token = jieba_tokenize(_text)
hanLP_token = hanLP_tokenize(_text)
thu_token = thu_tokenize(_text)

print('jieba result:')
print(jieba_token)

print('hanlp result:')
print(hanLP_token)

print('thu result:')
print(thu_token)

```

结果

```
jieba result:
[('今天', '-'), ('又', '-'), ('给', '-'), ('大家', '-'), ('送', '-'), ('福利', 'ns'), ('啦', '-'), (',', '-'), ('现在', '-'), ('加入', '-'), ('腾讯', 'nz'), ('汽车', 'n'), ('会员', 'n'), ('，', '-'), ('享', '-'), ('中石油', 'x'), ('、', '-'), ('中石化', 'j'), ('实名', '-'), ('充值', 'n'), ('九折卡', 'x'), ('，', '-'), ('最高', '-'), ('100', '-'), ('万', '-'), ('的', '-'), ('驾乘', 'nz'), ('意外险', 'n'), ('、', '-'), ('违章', '-'), ('代缴', '-'), ('6', '-'), ('折', '-'), ('优惠', '-'), ('等', '-'), ('更', '-'), ('多', '-'), ('！', '-')]
hanlp result:
[('今天', '-'), ('又', '-'), ('给', '-'), ('大家', '-'), ('送', '-'), ('福利', 'NN'), ('啦', '-'), (',', '-'), ('现在', '-'), ('加入', '-'), ('腾讯', 'NR'), ('汽车', 'NN'), ('会员', 'NN'), ('，', '-'), ('享', '-'), ('中石油', 'NN'), ('、', '-'), ('中石化', 'NN'), ('实名', '-'), ('充值', 'NN'), ('九折卡', 'NN'), ('，', '-'), ('最高', '-'), ('100万', '-'), ('的', '-'), ('驾乘', '-'), ('意外险', 'NN'), ('、', '-'), ('违章', 'NN'), ('代', '-'), ('缴', '-'), ('6', '-'), ('折', '-'), ('优惠', '-'), ('等', '-'), ('更', '-'), ('多', '-'), ('！', '-')]
thu result:
[('今天', '-'), ('又', '-'), ('给', '-'), ('大家', '-'), ('送', '-'), ('福利', 'n'), ('啦', '-'), (',', '-'), ('现在', '-'), ('加入', '-'), ('腾讯', 'nz'), ('汽车', 'n'), ('会员', 'n'), ('，', '-'), ('享中', '-'), ('石油', 'n'), ('、', '-'), ('中石化', 'j'), ('实名', 'n'), ('充值', '-'), ('九折卡', 'uw'), ('，', '-'), ('最高', '-'), ('100万', '-'), ('的', '-'), ('驾乘意', '-'), ('外险', 'n'), ('、', '-'), ('违章', '-'), ('代', '-'), ('缴', '-'), ('6', '-'), ('折', '-'), ('优惠', '-'), ('等', '-'), ('更', '-'), ('多', '-'), ('！', '-')]
jieba result:
[('从', '-'), ('君士坦丁堡', 'nz'), ('陷落', '-'), ('到', '-'), ('地中海', 'ns'), ('海洋', 'ns'), ('混战', '-'), ('，', '-'), ('一直', '-'), ('到', '-'), ('威尼斯', '-'), ('《', '-'), ('海洋', 'ns'), ('霸权', 'n'), ('》', '-'), ('的', '-'), ('史诗', 'nr'), ('，', '-'), ('其实', '-'), ('这', '-'), ('三', '-'), ('本书', '-'), ('的', '-'), ('角度', 'n'), ('和', '-'), ('视角', 'n'), ('并', '-'), ('不', '-'), ('一致', '-'), ('，', '-'), ('但是', '-'), ('总', '-'), ('有', '-'), ('一种', '-'), ('主题', 'n'), ('和', '-'), ('精神', 'n'), ('气质', 'n'), ('贯穿', '-'), ('始终', '-'), ('。', '-'), ('基督', 'n'), ('文明', 'nr'), ('和', '-'), ('伊斯兰', 'ns'), ('文明', 'nr'), ('在', '-'), ('地中海', 'ns'), ('的', '-'), ('碰撞', '-'), ('，', '-'), ('献上', '-'), ('了', '-'), ('交织着', '-'), ('血', 'n'), ('与', '-'), ('泪', 'n'), ('的', '-'), ('史诗', 'nr'), ('。', '-')]
hanlp result:
[('从', '-'), ('君士坦丁堡', 'NR'), ('陷落', '-'), ('到', '-'), ('地中海', 'NR'), ('海洋', 'NN'), ('混战', 'NN'), ('，', '-'), ('一直', '-'), ('到', '-'), ('威尼斯', 'NR'), ('《', '-'), ('海洋', 'NN'), ('霸权', 'NN'), ('》', '-'), ('的', '-'), ('史诗', 'NN'), ('，', '-'), ('其实', '-'), ('这', '-'), ('三', '-'), ('本', '-'), ('书', '-'), ('的', '-'), ('角度', 'NN'), ('和', '-'), ('视角', 'NN'), ('并', '-'), ('不', '-'), ('一致', '-'), ('，', '-'), ('但是', '-'), ('总', '-'), ('有', '-'), ('一', '-'), ('种', '-'), ('主题', 'NN'), ('和', '-'), ('精神', 'NN'), ('气质', 'NN'), ('贯穿', '-'), ('始终', '-'), ('。', '-'), ('基督', 'NN'), ('文明', 'NN'), ('和', '-'), ('伊斯兰', 'NR'), ('文明', 'NN'), ('在', '-'), ('地中海', 'NR'), ('的', '-'), ('碰撞', 'NN'), ('，', '-'), ('献', '-'), ('上', '-'), ('了', '-'), ('交织', '-'), ('着', '-'), ('血', 'NN'), ('与', '-'), ('泪', 'NN'), ('的', '-'), ('史诗', 'NN'), ('。', '-')]
thu result:
[('从', '-'), ('君士坦丁堡', 'uw'), ('陷落', '-'), ('到', '-'), ('地中海', 'ns'), ('海洋', 'n'), ('混战', '-'), ('，', '-'), ('一直', '-'), ('到', '-'), ('威尼斯', 'ns'), ('《', '-'), ('海洋霸权', 'n'), ('》', '-'), ('的', '-'), ('史诗', 'n'), ('，', '-'), ('其实', '-'), ('这', '-'), ('三', '-'), ('本', '-'), ('书', '-'), ('的', '-'), ('角度', 'n'), ('和', '-'), ('视角', 'n'), ('并', '-'), ('不', '-'), ('一致', '-'), ('，', '-'), ('但是', '-'), ('总', '-'), ('有', '-'), ('一', '-'), ('种', '-'), ('主题', 'n'), ('和', '-'), ('精神', 'n'), ('气质', 'n'), ('贯穿', '-'), ('始终', '-'), ('。', '-'), ('基督', 'nz'), ('文明', 'n'), ('和', '-'), ('伊斯兰', 'nz'), ('文明', 'n'), ('在', '-'), ('地中海', 'ns'), ('的', '-'), ('碰撞', '-'), ('，', '-'), ('献', '-'), ('上', '-'), ('了', '-'), ('交织', '-'), ('着', '-'), ('血', 'n'), ('与', '-'), ('泪', 'n'), ('的', '-'), ('史诗', 'n'), ('。', '-')]

```

#### 过滤掉不需要的tag

```python
from tokenizer import TokenFilter
_text = '今天又给大家送福利啦,现在加入腾讯汽车会员，享中石油、中石化实名充值九折卡，最高100万的驾乘意外险、违章代缴6折优惠等更多！'
# 默认使用jieba分词
_ = TokenFilter()
__ = _.filtrate(_text)
print(__)
_text = '从君士坦丁堡陷落到地中海海洋混战，一直到威尼斯《海洋霸权》的史诗，其实这三本书的角度和视角并不一致，' \
        '但是总有一种主题和精神气质贯穿始终。基督文明和伊斯兰文明在地中海的碰撞，献上了交织着血与泪的史诗。'
_ = TokenFilter()
__ = _.filtrate(_text)
print(__)

```

结果

```
['福利', '腾讯', '汽车', '会员', '中石油', '中石化', '充值', '九折卡', '驾乘', '意外险']
['君士坦丁堡', '地中海', '海洋', '海洋', '霸权', '史诗', '角度', '视角', '主题', '精神', '气质', '基督', '文明', '伊斯兰', '文明', '地中海', '血', '泪', '史诗']
```


###注

使用清华模型需要[下载](conf/README.md)模型
