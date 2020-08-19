from hanLP_tokenizer import hanLP_token_filter

_text = '吃饭，今天又给大家送福利啦,现在加入腾讯汽车会员，享中石油、中石化实名充值九折卡，最高100万的驾乘意外险、违章代缴6折优惠等更多！'

filter_tokens = hanLP_token_filter(_text)
print(_text)
print(filter_tokens)

_text = '从君士坦丁堡陷落到地中海海洋混战，一直到威尼斯《海洋霸权》的史诗，其实这三本书的角度和视角并不一致，但是总有一种主题和精神气质贯穿始终。' \
        '基督文明和伊斯兰文明在地中海的碰撞，献上了交织着血与泪的史诗。'
filter_tokens = hanLP_token_filter(_text)
print(_text)
print(filter_tokens)
