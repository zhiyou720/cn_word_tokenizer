class TokenFilter:
    def __init__(self, tokenizer: str = 'jieba', ignore_tag='-'):
        """

        :param tokenizer:
            "thu": 清华分词: http://thulac.thunlp.org/ （可以找到词性集）
            "jieba": jieba 分词: https://github.com/fxsjy/jieba （可以找到词性集）
            "hanlp": hanLP 分词：https://github.com/hankcs/HanLP （找到的词性集可能过时）
            默认jieba分词
        """
        if tokenizer == 'thu':
            from thu_tokenizer import thu_tokenize
            self.tokenize = thu_tokenize
        elif tokenizer == 'hanlp':
            from hanLP_tokenizer import hanLP_tokenize
            self.tokenize = hanLP_tokenize
        else:
            from jieba_tokenizer import jieba_tokenize
            self.tokenize = jieba_tokenize

        self.ignore_tag = ignore_tag

    def filtrate(self, text):
        tagged_tokens = self.tokenize(text)
        return [x[0] for x in tagged_tokens if x[1] != self.ignore_tag]


if __name__ == '__main__':
    _text = '今天又给大家送福利啦,现在加入腾讯汽车会员，享中石油、中石化实名充值九折卡，最高100万的驾乘意外险、违章代缴6折优惠等更多！'
    _ = TokenFilter()
    __ = _.filtrate(_text)
    print(__)
    _text = '从君士坦丁堡陷落到地中海海洋混战，一直到威尼斯《海洋霸权》的史诗，其实这三本书的角度和视角并不一致，' \
            '但是总有一种主题和精神气质贯穿始终。基督文明和伊斯兰文明在地中海的碰撞，献上了交织着血与泪的史诗。'
    _ = TokenFilter()
    __ = _.filtrate(_text)
    print(__)
