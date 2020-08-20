import thulac
from nltk.corpus import stopwords


class ThuTokenizer:
    def __init__(self, model_path, user_dict_path, stop_word_path, considered_tag_path, ignore_tag='-'):

        self.zh_model = thulac.thulac(model_path=model_path, user_dict=user_dict_path)

        self.stop_word_path = stop_word_path
        self.considered_tag_path = considered_tag_path
        self.ignore_tag = ignore_tag

    def tokenize(self, text):
        """
        :param text: str
        :return:
        """

        tokens_tagged = [(x[0], x[1]) for x in self.zh_model.cut(text)]

        tokens_tagged = self.remove_tag_unconsidered(tokens_tagged)

        tokens_tagged = self.remove_stop_words(tokens_tagged)

        return tokens_tagged

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

    def remove_tag_unconsidered(self, tokens_tagged):
        considered_tags = set(self.load_txt_data(self.considered_tag_path))

        new_tokens_tagged = []

        for i in range(len(tokens_tagged)):
            if tokens_tagged[i][1] in considered_tags:
                new_tokens_tagged.append(tokens_tagged[i])
            else:
                new_tokens_tagged.append((tokens_tagged[i][0], self.ignore_tag))

        return new_tokens_tagged

    def remove_stop_words(self, tokens_tagged):
        stop_word_dict = set(self.load_txt_data(self.stop_word_path) + stopwords.words('english'))
        new_tokens_tagged = []

        for i in range(len(tokens_tagged)):
            if tokens_tagged[i][0].lower() not in stop_word_dict:
                new_tokens_tagged.append(tokens_tagged[i])
            else:
                new_tokens_tagged.append((tokens_tagged[i][0], self.ignore_tag))
        return new_tokens_tagged


thu_tokenizer = ThuTokenizer(model_path='conf/thulac.models',
                             user_dict_path='conf/user_dict.txt',
                             stop_word_path='conf/stopwords.txt',
                             considered_tag_path='conf/thu_considered_tags.txt')
thu_tokenize = thu_tokenizer.tokenize

if __name__ == '__main__':
    _text = '从君士坦丁堡陷落到地中海海洋混战，一直到威尼斯《海洋霸权》的史诗，其实这三本书的角度和视角并不一致，' \
            '但是总有一种主题和精神气质贯穿始终。基督文明和伊斯兰文明在地中海的碰撞，献上了交织着血与泪的史诗。'
    _ = thu_tokenize(_text)
    print(_)
    _text = '今天又给大家送福利啦,现在加入腾讯汽车会员，享中石油、中石化实名充值九折卡，最高100万的驾乘意外险、违章代缴6折优惠等更多！'
    _ = thu_tokenize(_text)
    print(_)