import hanlp
from hanlp.common.trie import Trie

from nltk.corpus import stopwords


class hanLPTokenizer:
    def __init__(self, hanlp_tokenizer, hanlp_tagger, user_dict_path, stop_words_path, consider_tags_path,
                 ignore_tag='-'):
        self.hanlp_tokenizer = hanlp_tokenizer
        self.tagger = hanlp_tagger
        self.ignore_tag = ignore_tag
        self.stop_words = self.load_stop_words(stop_words_path)
        self.considered_tags = self.load_consider_tags(consider_tags_path)
        self.user_dict = self.load_user_dict(user_dict_path)
        self.trie = Trie()
        self.trie.update(self.user_dict)
        self.tokenizer = hanlp.pipeline() \
            .append(self.split_sentences, output_key=('parts', 'offsets', 'words')) \
            .append(self.hanlp_tokenizer, input_key='parts', output_key='tokens') \
            .append(self.merge_parts, input_key=('tokens', 'offsets', 'words'), output_key='merged')

    def split_sentences(self, text: str):
        words = self.trie.parse_longest(text)
        sentences = []
        pre_start = 0
        offsets = []
        for word, value, start, end in words:
            if pre_start != start:
                sentences.append(text[pre_start: start])
                offsets.append(pre_start)
            pre_start = end
        if pre_start != len(text):
            sentences.append(text[pre_start:])
            offsets.append(pre_start)
        return sentences, offsets, words

    @staticmethod
    def merge_parts(parts, offsets, words):
        items = [(i, p) for (i, p) in zip(offsets, parts)]
        items += [(start, [word]) for (word, value, start, end) in words]
        # In case you need the tag, use the following line instead
        # items += [(start, [(word, value)]) for (word, value, start, end) in words]
        return [each for x in sorted(items) for each in x[1]]

    def tokenize(self, text):
        """

        :param text: str
        :return:
        """
        return self.tokenizer(text)['merged']

    def tag(self, tokens):
        """

        :param tokens: list
        :return:
        """
        return self.tagger(tokens)

    def tag_stop_words(self, tokens, tags):
        new_tags = []
        for i in range(len(tokens)):
            if tokens[i] in self.stop_words:
                new_tags.append(self.ignore_tag)
            else:
                new_tags.append(tags[i])
        return new_tags

    def tag_unconsidered_tags(self, tags):
        new_tags = []
        for tag in tags:
            if tag.lower() in self.considered_tags:
                new_tags.append(tag)
            else:
                new_tags.append(self.ignore_tag)
        return new_tags

    def tokenize_filter(self, text):
        tokens = self.tokenize(text)
        tags = self.tag(tokens)
        tags = self.tag_stop_words(tokens, tags)  # remove stop words
        tags = self.tag_unconsidered_tags(tags)  # tag filter
        tagged_tokens = []
        for i in range(len(tags)):
            tagged_tokens.append((tokens[i], tags[i]))
        return tagged_tokens

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

    def load_user_dict(self, path):
        raw = self.load_txt_data(path)
        user_word_dict = {}
        for i in range(len(raw)):
            word = raw[i].split('\t')[0]
            if word not in user_word_dict:
                user_word_dict[word] = ' '
        return user_word_dict

    def load_stop_words(self, path):
        return set(self.load_txt_data(path) + stopwords.words('english'))

    def load_consider_tags(self, path):
        return set([x.split('\t')[0].lower() for x in self.load_txt_data(path)])


_hanlp_tokenizer = hanlp.load('PKU_NAME_MERGED_SIX_MONTHS_CONVSEG')
_hanlp_tagger = hanlp.load(hanlp.pretrained.pos.CTB5_POS_RNN_FASTTEXT_ZH)
_user_dict_path = 'conf/user_dict.txt'
_stop_words_path = 'conf/stopwords.txt'
_consider_tags_path = 'conf/hanLP_considered_tags.txt'
_t = hanLPTokenizer(_hanlp_tokenizer, _hanlp_tagger, user_dict_path=_user_dict_path, stop_words_path=_stop_words_path,
                    consider_tags_path=_consider_tags_path)

hanLP_tokenize = _t.tokenize_filter

if __name__ == '__main__':
    _text = '吃饭，今天又给大家送福利啦,现在加入腾讯汽车会员，享中石油、中石化实名充值九折卡，最高100万的驾乘意外险、违章代缴6折优惠等更多！'

    _tokens = hanLP_tokenize(_text)
    print(_text)
    print(_tokens)
