from flask import Flask
from api import sentiment_blueprint

import config

app = Flask(__name__)
app.config.from_object('config.DevConfig')

model_path = None # 모델 경로
model = None # 모델 불러오기

app.register_blueprint(sentiment_blueprint)

#config test
print(f"{app.config['SAMPLE_VARIABLE']}, {app.config['DEV_SAMPLE']}")

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5000, debug=True)
