import json
import logging
import os
import openai
import requests
import subject2
import subject3
from dto import ChatbotRequest

# 환경 변수 처리 필요!
openai.api_key = os.environ['OPENAI_API_KEY']
logger = logging.getLogger("Callback")

def callback_handler(request: ChatbotRequest) -> dict:
    print("utterance : ", request.userRequest.utterance)

    llm_text = subject2.llm_initial_setting(request.userRequest.utterance)
   # 참고링크 통해 payload 구조 확인 가능
    payload = {
        "version": "2.0",
        "template": {
            "outputs": [
                {
                    "simpleText": {
                        "text": llm_text
                    }
                }
            ]
        }
    }

    print(payload)
    # ===================== end =================================
    # 참고링크1 : https://kakaobusiness.gitbook.io/main/tool/chatbot/skill_guide/ai_chatbot_callback_guide
    # 참고링크1 : https://kakaobusiness.gitbook.io/main/tool/chatbot/skill_guide/answer_json_format

    url = request.userRequest.callbackUrl

    response = requests.post(url = url, data = json.dumps(payload), headers={"Content-Type": "application/json"})
    print(response.json())

def callback_handler2(request: ChatbotRequest) -> dict:
    subject2.initial_setting()

    utterance = request.userRequest.utterance
    functions = [
        {
            "name": "find_kakao_db",
            "func": lambda x: subject2.find_kakao_db(utterance),
            "description": "카카오 싱크에 대한 질문에 답변한다"
        }
    ]

    output_text = subject2.prompt_with_chromaDB(utterance, functions=functions, temperature=0.0)

    payload = {
        "version": "2.0",
        "template": {
            "outputs": [
                {
                    "simpleText": {
                        "text": output_text
                    }
                }
            ]
        }
    }
    url = request.userRequest.callbackUrl

    response = requests.post(url=url, data=json.dumps(payload), headers={"Content-Type": "application/json"})
    print(response.json())

def callback_handler3(request: ChatbotRequest) -> dict:
    result = subject3.callback_handler(request)

    payload = {
        "version": "2.0",
        "template": {
            "outputs": [
                {
                    "simpleText": {
                        "text": result["answer"]
                    }
                }
            ]
        }
    }
    url = request.userRequest.callbackUrl

    response = requests.post(url=url, data=json.dumps(payload), headers={"Content-Type": "application/json"})
    print(response.json())

