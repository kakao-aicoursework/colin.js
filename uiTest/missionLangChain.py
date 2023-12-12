import json
import openai
import tkinter as tk
import pandas as pd
from tkinter import scrolledtext
import tkinter.filedialog as filedialog
import os
import chromadb

from langchain.chains import RetrievalQA
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader

from langchain.agents import AgentType, Tool, initialize_agent
from langchain.llms import OpenAI

openai.api_key = os.environ['OPENAI_API_KEY']

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
    client = chromadb.PersistentClient()
    kakao_sync_collection = client.get_or_create_collection(
        name="kakao_sync",
        metadata={"hnsw:space": "cosine"}
    )

    query_result = kakao_sync_collection.query(
        query_texts=[query],
        n_results=1
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

def llm_initial_setting():
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
            description="카카오 싱크에 대한 질문에 답변한다"
        )
    ]

    # Construct the agent. We will use the default agent type here.
    # See documentation for a full list of options.
    agent = initialize_agent(
        tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
    )

    agent.run(
        "카카오싱크 기능이 무엇이 있는지 설명해주세요"
    )

#
# def read_prompt_template(file_path: str) -> str:
#     with open(file_path, "r") as f:
#         prompt_template = f.read()
#
#     return prompt_template
#
# def llm_generate_novel(genre, characters, news_text) -> dict[str, str]:
#     prompt_template = read_prompt_template('/content/drive/MyDrive/datas/prompt_template.txt')
#
#     chat = ChatOpenAI(temperature=0.8)
#     '''
#     chat = ChatOpenAI(
#         model_name='gpt-3.5-turbo-16k',
#         temperature = self.config.llm.temperature,
#         openai_api_key = self.config.llm.openai_api_key,
#         max_tokens=self.config.llm.max_tokens
#     )
#     '''
#
#     # system_message = "assistant는 마케팅 문구 작성 도우미로 동작한다. user의 내용을 참고하여 마케팅 문구를 작성해라"
#     # system_message_prompt = SystemMessage(content=system_message)
#
#     human_template = ("genre: {genre}\n"
#                       "characters: {characters}\n"
#                       "news_text: {news_text}\n"
#                       )
#
#     human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
#     human_message_prompt
#
#     chat_prompt = ChatPromptTemplate.from_messages([human_message_prompt])
#
#     chain = LLMChain(llm=chat, prompt=chat_prompt)
#
#     result = chain.run(genre=genre,
#                        characters=characters,
#                        news_text=news_text)
#
#     return {"results": result}

def main():
    # llm = OpenAI(temperature=0, max_tokens=1024)

    #파일 읽기 및 저장
    llm_initial_setting()

    # tools = [
    #     Tool(
    #         name="document of Kakao Sync service",
    #         func=find_kakao_db.run,
    #         description='''
    #             You are a Kakao Sync chatbot.
    #             If you have any questions about Kakao Sync, please answer them as kindly as possible
    #             Please answer in Korean
    #             Please answer step by step so that others can understand. Please check again step by step if the answer is correct."
    #             ''',
    #     )
    # ]
    #
    # # Construct the agent. We will use the default agent type here.
    # # See documentation for a full list of options.
    # agent = initialize_agent(
    #     tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
    # )
    #
    # agent.run(
    #     "카카오로 시작하기 버튼에 대해서 알려줘"
    # )

if __name__ == "__main__":
    main()
