import time

def update():

    import requests
    import pandas as pd
    import datetime
    import sqlite3
    import itertools
    import time
    import emoji

    from utils import extract_words

    time_now = datetime.datetime.now(tz=datetime.timezone(datetime.timedelta(hours=9)))
    print(time_now, "開始")

    """
        データベースの初期化
    """

    with sqlite3.connect('/db/qiita.db') as con:
        
        cur = con.cursor()
        
        if cur.execute("SELECT COUNT(*) FROM sqlite_master WHERE TYPE='table' AND name='PAGE_ATTRIBUTE'").fetchone()[0] == 0:
            cur = con.cursor()
            cur.execute("CREATE TABLE PAGE_ATTRIBUTE(id TEXT PRIMARY KEY, title TEXT, url TEXT, created_at INT)")

        """
        過去100ページ分（1ページあたり50件）の記事を取得
    """

    token = "1b1049910bdd487d1876fc80618390ae8ae32e5b"

    def get_qiita_dataframe(page=1):
        
        """
            func: 
                - Qiitaから50件（$pageページ目）の記事を取得
            args:
                - page: ページ番号
            returns:
                - None
        """

        headers = {
            "Authorization": f"Bearer {token}",
        }
        response = requests.get(f"https://qiita.com/api/v2/items?page={page}&per_page=50", headers=headers)
        
        # ステータス確認
        if response.status_code != 200:
            return False
        
        # JSONで受け取る
        response = response.json()
        
        # DataFrameに変換
        df = pd.DataFrame(response)[["id", "title", "url", "created_at"]]
        df["created_at"] = pd.to_datetime(df["created_at"]).map(pd.Timestamp.timestamp).astype(int) # 作成日時をタイムスタンプに変換
        
        # SQLite3に保存
        with sqlite3.connect('/db/qiita.db') as con:

            # dfのうち，既存のidを検索
            cur = con.cursor()
            query = f"SELECT id FROM PAGE_ATTRIBUTE WHERE id in (%s)" % (",".join(["?"] * df.shape[0]))
            exist_ids = list(itertools.chain.from_iterable(cur.execute(query, tuple(df["id"])).fetchall()))
            
            # 既存の記事は除外
            df = df[~df["id"].isin(exist_ids)].copy()

            # 挿入
            df.to_sql(
                name = 'PAGE_ATTRIBUTE',
                con = con,
                if_exists='append',
                index = False,
                method = 'multi',
            )
            
        return True
            
    for page in range(1, 101):
        get_qiita_dataframe(page=page)

    """
        タイトルを埋め込みに変換
    """

    with sqlite3.connect('/db/qiita.db') as con:
        
        # 3日前のタイムスタンプ
        timestamp_three_days_ago = int(time.time()) - (60 * 60 * 24 * 3)
        
        # 3日前までの記事一覧を抽出
        df_articles = pd.read_sql_query(
            f"SELECT id, title FROM PAGE_ATTRIBUTE WHERE created_at > {timestamp_three_days_ago}",
            con=con
        )
        
    # タイトルから名詞を抽出
    text_list = df_articles["title"]
    text_list = [extract_words(text) for text in text_list]

    # ID×単語 の頻度行列を生成
    df_count = [pd.Series(1, index=word_list).groupby(level=0).sum() for word_list in text_list]
    df_count = pd.concat(df_count, axis=1).T.fillna(0)
    df_count.index = df_articles["id"]

    # 数字のみの単語は削除
    df_count = df_count.loc[:, ~df_count.columns.str.match("^\d+$")]

    # アルファベットは小文字に変換
    df_count.columns = df_count.columns.str.lower()

    # 絵文字は削除
    df_count = df_count.loc[:, ~df_count.columns.map(emoji.is_emoji)]

    # 重複した単語は集約
    df_count = df_count.groupby(level=0, axis=1).sum()

    # 単語をソート
    df_count = df_count[df_count.columns.sort_values()]

    # 正規化
    df_count = (df_count.T / df_count.sum(axis=1)).T.fillna(0)

    # 単語の埋め込み行列を生成
    df_words_embed = requests.post('http://fasttext-vector-api:80/', json=df_count.columns.to_list())
    df_words_embed = pd.DataFrame(df_words_embed.json()).T

    # タイトルの埋め込み行列を生成
    df_title_embed = df_count.dot(df_words_embed)

    # SQLite3に保存
    with sqlite3.connect('/db/qiita.db') as con:

        # 挿入
        df_title_embed.to_sql(
            name = 'TITLE_EMBEDDING',
            con = con,
            if_exists='replace',
            index = True,
            method = 'multi',
            dtype={
                "id": "TEXT PRIMARY KEY",
            },
        )

    print(time_now, "完了")

while True:
    update()
    time.sleep(30 * 60)