import json

import requests

from dto import ChatbotRequest
from samples import list_card
import aiohttp
import time
import logging
import openai
import os
import chromadb

from langchain.chains import RetrievalQA
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader

from langchain.agents import AgentType, Tool, initialize_agent
from langchain.llms import OpenAI

# 환경 변수 처리 필요!
openai.api_key = os.environ['OPENAI_API_KEY']
SYSTEM_MSG = "당신은 카카오 서비스 제공자입니다."
logger = logging.getLogger("Callback")

def llm_initial_setting(query):
    llm = OpenAI(temperature=0)
    doc_path = str('/Users/colin/llmProject/pythonProject1/uiTest/resources/project_data_카카오싱크.txt')
    loader = TextLoader(doc_path)
    documents = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)

    print("texts : ", texts)

    embeddings = OpenAIEmbeddings()
    docsearch = Chroma.from_documents(texts, embeddings, collection_name="kakao-sync")

    kakao_sync = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=docsearch.as_retriever(search_kwargs={"k": 1})
    )

    tools = [
        Tool(
            name="document of Kakao Sync service",
            func=kakao_sync.run,
            description="카카오 싱크에 대한 질문에 한글로 답변한다"
        )
    ]

    # Construct the agent. We will use the default agent type here.
    # See documentation for a full list of options.
    agent = initialize_agent(
        tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True, stream_prefix = True
    )

    return agent.run(
        f"{query}"
    )

def callback_handler(request: ChatbotRequest) -> dict:
    print("옴")
    # # ===================== start =================================
    # response = openai.ChatCompletion.create(
    #     model="gpt-3.5-turbo",
    #     messages=[
    #         {"role": "system", "content": SYSTEM_MSG},
    #         {"role": "user", "content": request.userRequest.utterance},
    #     ],
    #     temperature=0,
    # )
    # # focus
    # output_text = response.choices[0].message.content

    print("utterance : ", request.userRequest.utterance)

    llm_text = llm_initial_setting(request.userRequest.utterance)
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
