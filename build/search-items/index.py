# FLASK
from flask import Flask, jsonify, request
app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

import pandas as pd
import numpy as np
import requests
import sqlite3
from sklearn.metrics.pairwise import cosine_similarity

@app.route("/", methods=["GET"])
def index():

    """
        READ
    """

    with sqlite3.connect('/db/qiita.db') as con:
        df_attribute = pd.read_sql_query("SELECT id, title, url FROM PAGE_ATTRIBUTE", con=con).set_index("id")
        df_embedding = pd.read_sql_query("SELECT * FROM TITLE_EMBEDDING", con=con).set_index("id")

    """
        対象単語の埋め込みを抽出
    """

    target_word = request.args.get("s", "")

    # Check
    if target_word == "":
        return jsonify({"message": "パラメータsが指定されていません。"}), 400

    target_embedding = requests.post('http://fasttext-vector-api:80/', json=[target_word])
    target_embedding = np.array(list(target_embedding.json().values()))
    target_embedding = target_embedding.mean(axis=0).reshape(1, -1)

    """
        タイトルと単語のコサイン類似度
    """

    cos_sim = cosine_similarity(target_embedding, df_embedding.to_numpy())[0]

    """
        類似度Top10の属性情報を抽出
    """

    TopK_indices = np.argsort(-cos_sim)[:10]
    TopK_ids = df_attribute.iloc[TopK_indices].index
    TopK_attributes = df_attribute.loc[TopK_ids]

    return jsonify(list(TopK_attributes.T.to_dict().values())), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80)