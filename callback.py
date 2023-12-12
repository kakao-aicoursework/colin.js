import json

import requests

from dto import ChatbotRequest
import logging
import openai
import os
import chromadb

from langchain import LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import SystemMessage
from langchain.chains import RetrievalQA
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader

from langchain.agents import AgentType, Tool, initialize_agent
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

# 환경 변수 처리 필요!
openai.api_key = os.environ['OPENAI_API_KEY']
SYSTEM_MSG = "당신은 카카오 서비스 제공자입니다."
logger = logging.getLogger("Callback")

def initial_setting():
    # 파일 읽기 및 정리
    file = open('/Users/colin/llmProject/pythonProject1/uiTest/resources/project_data_카카오싱크.txt', 'r',
                encoding='utf-8')
    contents = file.read()

    datas = []
    dataArray = contents.split('\n#')[1:]

    for data in dataArray:
        title, content = map(str.strip, data.split('\n', 1))
        document = f"{title}:{data}"
        datas.append(document)
        # contentArray = content.split('\n')
        # for text in contentArray:
        #     if text == "":
        #         continue
        #     else:
        #         document = f"{title}:{data}"
        #         datas.append(document)

    file.close()  # 파일을 닫아주는 라인

    idx = 1
    ids = []
    documents = []

    for data in datas:
        ids.append("id" + str(idx))
        # datas = data.split(':')
        documents.append(
            data
        )
        idx = idx + 1

    client = chromadb.PersistentClient()

    #적재된것이 있으면 삭제하고 새로 만듬
    #client.delete_collection(name="kakao_sync")

    kakao_sync_collection = client.get_or_create_collection(
        name="kakao_sync",
        metadata={"hnsw:space": "cosine"}
    )

    # DB 저장
    kakao_sync_collection.add(
        documents=documents,
        ids=ids
    )

def find_kakao_db(query: str):
    print("find_kakao_db 호출 : ")
    client = chromadb.PersistentClient()
    kakao_sync_collection = client.get_or_create_collection(
        name="kakao_sync",
        metadata={"hnsw:space": "cosine"}
    )

    query_result = kakao_sync_collection.query(
        query_texts=[query],
        n_results=2
    )

    print("query_result : ", query_result)

    search_results = []
    for document in query_result['documents'][0]:
        print("document : ")
        print(document)

        docu_array = document.split(':')

        search_results.append(
            {
                "content": docu_array[1]
            }
        )

    return search_results

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
            description="카카오 싱크에 대한 질문에 한글로 단계적으로 답변한다"
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

def prompt_with_chromaDB(query, temperature=0.0, model="gpt-3.5-turbo", functions=None):
    system_message = "assistant는 user의 내용을 bullet point 3줄로 요약하라. 영어인 경우 한국어로 번역해서 요약하라."
    system_message_prompt = SystemMessage(content=system_message)

    human_template = "{query}위 내용을 bullet point로 3줄로 한국어로 요약해"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt,
                                                    human_message_prompt])

    print(functions)

    llm = ChatOpenAI(model=model, temperature=temperature)
    tools = []
    if functions != None :
        print("functions 사용")
        tools = [
            Tool(
                **func
            ) for func in functions
        ]
        agent = initialize_agent(tools, llm, agent=AgentType.OPENAI_FUNCTIONS, verbose=True)
        final_result = ""
        output = agent.invoke(query)
        final_result = output["output"]
        return final_result
    else :
        print("functions 미사용")
        agent = LLMChain(llm=llm, prompt=chat_prompt)
        return agent.run(query)

def callback_handler2(request: ChatbotRequest) -> dict:
    initial_setting()

    utterance = request.userRequest.utterance
    functions = [
        {
            "name": "find_kakao_db",
            "func": lambda x: find_kakao_db(utterance),
            "description": "카카오 싱크에 대한 질문에 답변한다"
        }
    ]

    output_text = prompt_with_chromaDB(utterance, functions=functions, temperature=0.0)

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


def callback_handler(request: ChatbotRequest) -> dict:
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
