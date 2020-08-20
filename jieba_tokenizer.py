import jieba
import jieba.posseg as jpg

from nltk.corpus import stopwords


class JiebaTokenizer:

    def __init__(self, user_dict_path, puncs_path, considered_tags_path, stop_word_path, ignore_tag='-'):
        self.user_dict_path = user_dict_path
        self.puncs_path = puncs_path
        self.considered_tags_path = considered_tags_path
        self.stop_word_path = stop_word_path
        self.ignore_tag = ignore_tag

        self.load_user_dict()

    def tokenizer(self, text):
        """
        :param text: str
        :return:
        """

        tokens_tagged = [(list(x)[0], list(x)[1]) for x in jpg.lcut(text)]

        tokens_tagged = self.tag_puncs(tokens_tagged)

        tokens_tagged = self.tag_unconsidered(tokens_tagged)

        tokens_tagged = self.tag_stop_words(tokens_tagged)

        return tokens_tagged

    def load_user_dict(self):
        user_dict = [x.split(' ')[0] for x in self.load_txt_data(self.user_dict_path)]
        for x in user_dict:
            jieba.add_word(x)

    @staticmethod
    def load_txt_data(path, mode='utf-8-sig', origin=False):
        """
        This func is used to reading txt file
        :param origin:
        :param path: path where file stored
        :param mode:
        :type path: str
        :return: string lines in file in a list
        :rtype: list
        """
        if type(path) != str:
            raise TypeError
        res = []

        file = open(path, 'rb')
        lines = file.read().decode(mode, errors='ignore')
        for line in lines.split('\n'):
            line = line.strip()
            if origin:
                res.append(line)
            else:
                if line:
                    res.append(line)
        file.close()
        return res

    def tag_puncs(self, tokens_tagged):
        puncs = set(x.split('\t')[1] for x in self.load_txt_data(self.puncs_path))
        new_tokens_tagged = []
        for i in range(len(tokens_tagged)):
            if tokens_tagged[i][0] not in puncs:
                new_tokens_tagged.append(tokens_tagged[i])
            else:
                new_tokens_tagged.append((tokens_tagged[i][0], self.ignore_tag))

        return new_tokens_tagged

    def tag_stop_words(self, tokens_tagged):
        stop_word = set(self.load_txt_data(self.stop_word_path) + stopwords.words('english'))
        new_tokens_tagged = []

        for i in range(len(tokens_tagged)):
            if tokens_tagged[i][0].lower() not in stop_word:
                new_tokens_tagged.append(tokens_tagged[i])
            else:
                new_tokens_tagged.append((tokens_tagged[i][0], self.ignore_tag))

        return new_tokens_tagged

    def tag_unconsidered(self, tokens_tagged):
        considered_tags = set(self.load_txt_data(self.considered_tags_path))

        new_tokens_tagged = []
        for i in range(len(tokens_tagged)):
            if tokens_tagged[i][1] in considered_tags:
                new_tokens_tagged.append(tokens_tagged[i])
            else:
                new_tokens_tagged.append((tokens_tagged[i][0], self.ignore_tag))
        return new_tokens_tagged


_jieba_tokenizer = JiebaTokenizer(user_dict_path='conf/user_dict.txt',
                                  puncs_path='conf/punctuation.dat',
                                  considered_tags_path='conf/jieba_considered_tags.txt',
                                  stop_word_path='conf/stopwords.txt', )
jieba_tokenize = _jieba_tokenizer.tokenizer

if __name__ == '__main__':
    _text = '今天又给大家送福利啦,现在加入腾讯汽车会员，享中石油、中石化实名充值九折卡，最高100万的驾乘意外险、违章代缴6折优惠等更多！'

    _ = jieba_tokenize(_text)
    print(_)

    _text = '从君士坦丁堡陷落到地中海海洋混战，一直到威尼斯《海洋霸权》的史诗，其实这三本书的角度和视角并不一致，' \
            '但是总有一种主题和精神气质贯穿始终。基督文明和伊斯兰文明在地中海的碰撞，献上了交织着血与泪的史诗。'
    _ = jieba_tokenize(_text)
    print(_)
