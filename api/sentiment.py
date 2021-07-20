from api import sentiment_blueprint
from app import model
from flask import request
from KOBERT import predict

@sentiment_blueprint.route('', methods=['POST'])
def sentiment_classification():
    """
    ToDo: 감정 긍정 부정 판별 기능 추가 하기
    
    json 
    [{'title': '누가봐도 SKT한테 돈 받고 쓴거 같은 기사', 'content': '[뉴스+]SKT 멤버십 개편에 대한 오해와 진실 [이데일리 김현아 기자] 국민 절반이 가입한 이동통신회사 SK텔레콤의 ‘T멤버십’이 8월 중 포인트 기반으로 바뀐다. 제휴사별 할인 대신 같은 비율로 포인트를 받은 뒤 원하는 사용처에서 몰아 쓸 수 있게 되니 혜택이 https://news.naver.com/main/read.naver?mode=LSD&mid=sec&oid=018&aid=0004978133&sid1=001 기자가 친절히(?) 오해와 진실을 밝혀주네요ㅋㅋ 고객보고 혜택 받으려면 부지런한 엄지족이 되라고 ㅋㅋ', 'community_id': '2', 'department': 'KT 멤버십'}]
    
    """
    #print(request.is_json)
    params = request.get_json()
    
    if request.method == 'POST':
        return predict(params['content'])
