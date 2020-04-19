from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

value_list={
    "pclass": ["上層クラス（お金持ち）", "中級クラス（一般階級）", "下層クラス（労働階級）"],
    "sex": ["男性", "女性"],
    "embarked": ["Southampton", "Cherbourg", "Queenstown"]
}

@app.route('/')
def hello():
    return render_template('index.html',value_list=value_list,  title='Tatanic Predict Site') #変更

@app.route('/', methods=["POST"])
def result():
    post_array = {}
    post_array["pclass"] = int(request.form["pclass"])
    post_array["age"] = int(request.form["age"])
    post_array["sex"] = int(request.form["sex"])
    post_array["fare"] = request.form["fare"]
    post_array["family_size"] = request.form["familysize"]
    post_array["embarked"] = int(request.form["embarked"])
    predict_array = np.array([[post_array["pclass"], post_array["age"], post_array["sex"], post_array["fare"], post_array["family_size"], post_array["embarked"]]])
    forest = pickle.load(open('trained_model.pkl', 'rb'))
    predict = forest.predict(predict_array)
    if predict == 1:
        result = "生存"
    else:
        result = "死亡"
    print(result)
    return render_template('index.html', value_list=value_list, title='flask test', result=result, post_array=post_array)

## おまじない
if __name__ == "__main__":
    app.run(debug=True)

    # test_features = test_two[["Pclass", "Age", "Sex", "Fare", "family_size", "Embarked"]].values

