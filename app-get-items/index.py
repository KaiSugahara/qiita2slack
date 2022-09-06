import time

def update():

    import requests
    import json
    import pandas as pd
    from tqdm import trange
    import datetime
    from sklearn.feature_extraction.text import TfidfTransformer

    time_now = datetime.datetime.now(tz=datetime.timezone(datetime.timedelta(hours=9)))
    print(time_now, "開始")

    ########################################
    # 関数: 10件（$pageページ目）の記事を取得
    ########################################

    def get_qiita_dataframe(page=1):

        headers = {
            # "Authorization": "Bearer "
        }
        response = requests.get(f"https://qiita.com/api/v2/items?page={page}&per_page=50", headers=headers)
        response = json.loads(response.text)

        return pd.DataFrame(
            [(item["id"], item["title"], item["url"], datetime.datetime.fromisoformat(item["created_at"])) for item in response],
            columns=["id", "title", "url", "datetime"]
        )

    ########################################
    # 関数: MeCabを用いて名詞を抽出
    ########################################

    import MeCab
    mecab = MeCab.Tagger('-Ochasen -d /usr/lib/aarch64-linux-gnu/mecab/dic/mecab-ipadic-neologd')

    def extract_words(text):
        # 分かち書きして、名詞に限定
        return [word.split()[0] for word in mecab.parse(text).splitlines() if "名詞" in word.split()[-1]]

    ########################################
    # 過去3日分（およそ）の記事を取得
    ########################################

    df_qiita = []

    for page in trange(1, 101):

        df = get_qiita_dataframe(page)
        time_last = df.iloc[-1]["datetime"]
        time_sub = time_now - time_last

        df_qiita.append(df)

        if time_sub > datetime.timedelta(days=3):
            break

    df_qiita = pd.concat(df_qiita)

    ########################################
    # タイトル情報を保存
    ########################################

    df_attribute = df_qiita.set_index("id")[["title", "url"]]
    df_attribute.to_csv("/data/data_attribute.csv")

    ########################################
    # 本文中の単語から埋め込みに変換
    ########################################

    # 本文を抽出
    text_list = df_qiita["title"]
    text_list = [extract_words(text) for text in text_list]

    # ID×単語 の頻度行列を生成
    df_count = [pd.Series(1, index=word_list).groupby(level=0).sum() for word_list in text_list]
    df_count = pd.concat(df_count, axis=1).T.fillna(0)
    df_count.index = df_qiita["id"]
    # 数字のみの単語は削除
    df_count = df_count.loc[:, ~df_count.columns.str.match("^\d+$")]
    # アルファベットは小文字に変換
    df_count.columns = df_count.columns.str.lower()
    # 重複した単語は集約
    df_count = df_count.groupby(level=0, axis=1).sum()
    # TF-IDFで変換
    # df_count = pd.DataFrame(TfidfTransformer(smooth_idf=False).fit_transform(df_count).toarray(), index=df_count.index, columns=df_count.columns)
    # 単語をソート
    df_count = df_count[df_count.columns.sort_values()]
    # 正規化
    df_count = (df_count.T / df_count.sum(axis=1)).T.fillna(0)

    # 単語の埋め込み行列を生成
    df_words_embed = requests.post('http://app-fasttext:80/', json=df_count.columns.to_list())
    df_words_embed = pd.DataFrame(df_words_embed.json()).T

    # 本文の埋め込み行列を生成
    df_count.dot(df_words_embed).to_csv("/data/data_embedding.csv")

    print(time_now, "完了")

while True:
    update()
    time.sleep(30 * 60)