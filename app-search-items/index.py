# FLASK
from flask import Flask, jsonify, request
app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

import pandas as pd
import numpy as np
import requests
from sklearn.metrics.pairwise import cosine_similarity

@app.route("/", methods=["GET"])
def index():

    df_attribute = pd.read_csv("/data/data_attribute.csv", index_col=0)
    df_embedding = pd.read_csv("/data/data_embedding.csv", index_col=0)

    target_word = request.args.get("s", "")

    # Check
    if target_word == "":
        return jsonify({"message": "パラメータsが指定されていません。"}), 400

    target_embedding = requests.post('http://app-fasttext:80/', json=[target_word])
    target_embedding = np.array(list(target_embedding.json().values()))
    target_embedding = target_embedding.mean(axis=0).reshape(1, -1)

    cos_sim = cosine_similarity(target_embedding, df_embedding.to_numpy())[0]
    TopK = pd.Series(cos_sim).sort_values(ascending=False).head(10)

    return jsonify(list(df_attribute.iloc[TopK.index].T.to_dict().values())), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80)