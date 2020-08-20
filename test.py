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
