from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# 乗客情報
value_list = {
    "pclass": ["上層クラス（お金持ち）", "中級クラス（一般階級）", "下層クラス（労働階級）"],
    "sex": ["男性", "女性"],
    "embarked": ["Southampton", "Cherbourg", "Queenstown"]
}


@app.route('/')  # ホーム画面
def home():
    return render_template('index.html', value_list=value_list,  title='Tatanic Predict Site')


@app.route('/', methods=["POST"])  # ホーム画面POST
def result():
    # データの受け取り
    post_array = {}
    post_array["pclass"] = int(request.form["pclass"])
    post_array["age"] = int(request.form["age"])
    post_array["sex"] = int(request.form["sex"])
    post_array["fare"] = request.form["fare"]
    post_array["family_size"] = request.form["familysize"]
    post_array["embarked"] = int(request.form["embarked"])
    # 予測するためにNumpy配列に変換
    predict_array = np.array([[post_array["pclass"], post_array["age"], post_array["sex"],
                               post_array["fare"], post_array["family_size"], post_array["embarked"]]])
    # 学習済みモデルを呼び出し
    forest = pickle.load(open('trained_model.pkl', 'rb'))
    # 学習済みモデルで予測
    predict = forest.predict(predict_array)
    # 学習結果を出力
    if predict == 1:
        result = "生存"
    else:
        result = "死亡"
    return render_template('index.html', value_list=value_list, title='flask test', result=result, post_array=post_array)


# main.pyを実行
if __name__ == "__main__":
    app.run(debug=True)
