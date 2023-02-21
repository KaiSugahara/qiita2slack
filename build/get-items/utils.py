import MeCab
mecab = MeCab.Tagger('-Ochasen -d /usr/lib/aarch64-linux-gnu/mecab/dic/mecab-ipadic-neologd')

def extract_words(text):
    
    """
        func:
            - 分かち書きをして，名詞のリストを返す
        args:
            - text: 文 str
        returns:
            - 名詞のリスト list
    """
    
    return [word.split()[0] for word in mecab.parse(text).splitlines() if "名詞" in word.split()[-1]]