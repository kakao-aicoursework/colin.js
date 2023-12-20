import logging
import os

import chromadb
import openai
from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import ConversationChain, LLMChain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter

from pathlib import Path
from langchain.document_loaders import (
    NotebookLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)

from langchain.tools import Tool
from langchain.utilities import GoogleSearchAPIWrapper

from dto import ChatbotRequest

# 환경 변수 처리 필요!
openai.api_key = os.environ['OPENAI_API_KEY']
logger = logging.getLogger("subject3")

CUR_DIR = os.path.dirname(os.path.abspath('/Users/colin/llmProject/pythonProject1/resources/'))
GUIDE_STEP1_PROMPT_TEMPLATE = os.path.join(CUR_DIR, "resources/prompt/input_analyze.txt")
GUIDE_STEP2_PROMPT_TEMPLATE = os.path.join(CUR_DIR, "resources/prompt/input_solution.txt")
INTENT_PROMPT_TEMPLATE = os.path.join(CUR_DIR, "resources/prompt/parse_intent.txt")
INTENT_LIST_TXT = os.path.join(CUR_DIR, "resources/prompt/intent_list.txt")
SEARCH_VALUE_CHECK_PROMPT_TEMPLATE = os.path.join(CUR_DIR, "resources/prompt/search_value_check.txt")
SEARCH_COMPRESSION_PROMPT_TEMPLATE = os.path.join(CUR_DIR, "resources/prompt/search_compress.txt")

def read_prompt_template(file_path: str) -> str:
    print(file_path)
    with open(file_path, "r") as f:
        prompt_template = f.read()

    return prompt_template

def create_chain(llm, template_path, output_key):
    return LLMChain(
        llm=llm,
        prompt=ChatPromptTemplate.from_template(
            template=read_prompt_template(template_path)
        ),
        output_key=output_key,
        verbose=True,
    )

llm = ChatOpenAI(temperature=0.1, max_tokens=200, model="gpt-3.5-turbo")

guide_step1_chain = create_chain(
    llm=llm,
    template_path=GUIDE_STEP1_PROMPT_TEMPLATE,
    output_key="input_analysis",
)
guide_step2_chain = create_chain(
    llm=llm,
    template_path=GUIDE_STEP2_PROMPT_TEMPLATE,
    output_key="output",
)
parse_intent_chain = create_chain(
    llm=llm,
    template_path=INTENT_PROMPT_TEMPLATE,
    output_key="intent",
)
search_value_check_chain = create_chain(
    llm=llm,
    template_path=SEARCH_VALUE_CHECK_PROMPT_TEMPLATE,
    output_key="output",
)
search_compression_chain = create_chain(
    llm=llm,
    template_path=SEARCH_COMPRESSION_PROMPT_TEMPLATE,
    output_key="output",
)
default_chain = ConversationChain(llm=llm, output_key="output")

# def setting_vectorDb():
#     files = []
#     files.append({"collection_name" : "kakao_sync", "file_name" : "project_data_카카오싱크"})
#     files.append({"collection_name": "kakao_social", "file_name": "project_data_카카오소셜"})
#     files.append({"collection_name": "kakao_channel", "file_name": "project_data_카카오톡채널"})
#
#     for fileObj in files:
#         print(fileObj)
#
#         # 파일 읽기 및 정리
#         file = open(f"/Users/colin/llmProject/pythonProject1/resources/datas/{fileObj['file_name']}.txt", 'r', encoding='utf-8')
#         contents = file.read()
#
#         print(contents)
#
#         datas = []
#         dataArray = contents.split('\n#')[1:]
#
#         # 데이터 정제 - title 과 내용을 하나의 document로 구성하는 것
#         # 쓸데 없는 데이터는 정제해서 제거하고 토큰을 좀 더 효율적으로 사용하면 좋다.
#
#         # 대/중/소 분류를 인덱스로 다르게 구성하면 좋다.
#         # 같은 대분류라고 가정했을때 중에, 다른 것들 연관관계를 따질때는 metadata에 넣으면 연관성 검색시에 연결이 되서 좀 더 나은 결과를 가져오게 할 수 있다.
#
#         #메타데이터에는 유사성있는 데이터를 넣는게 좋은데, llm에 질문해서 나온 데이터들을 metadata에 넣어보는것도 좋다.
#
#         #서치한 데이터를 벡터디비에 저장(업데이트)해서 다시 쓰면, 부족한 데이터를 보완할 수 있다.
#
#         for data in dataArray:
#             title, content = map(str.strip, data.split('\n', 1))
#             document = f"{title}:{content}"
#             datas.append(document)
#
#             idx = 1
#             ids = []
#             documents = []
#
#             for data in datas:
#                 ids.append(title.replace(" ", "-") + str(idx))
#                 documents.append(
#                     data
#                 )
#                 idx = idx + 1
#
#             client = chromadb.PersistentClient()
#
#             # 적재된것이 있으면 삭제하고 새로 만듬
#             client.delete_collection(name=fileObj['collection_name'])
#             collection = client.get_or_create_collection(
#                 name=fileObj['collection_name'],
#                 metadata={"hnsw:space": "cosine"}
#             )
#
#             # DB 저장
#             collection.add(
#                 documents=documents,
#                 ids=ids
#             )

CHROMA_PERSIST_DIR = "datas/chroma_persist"
CHROMA_COLLECTION_NAME = "kakao_api"

def create_vectordb () :
    db = Chroma(
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=OpenAIEmbeddings(),
        collection_name=CHROMA_COLLECTION_NAME,
    )
    print("vector db completed")
    return db

def query_kakao_suppl_info (query: str, db, use_retriever: bool = False) :
    if use_retriever:
        docs = db.get_relevant_documents(query, k=2)
    else:
        docs = db.similarity_search(query, k=2)

    str_docs = [doc.page_content for doc in docs]
    return str_docs

def upload_embeddings_from_dir(dir_path):
    failed_upload_files = []
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            ext = file.split(".")[-1]
            conds = [
                ext == "txt",
                ext == "py",
                ext == "md",
                ext == "ipynb"]
            if any(conds) :
                file_path = os.path.join(root, file)

                try:
                    upload_embedding_from_file(file_path, ext)
                    print("SUCCESS: ", file_path)
                except Exception as e:
                    print("FAILED: ", file_path + f"by({e})")
                    failed_upload_files.append(file_path)

LOADER_DICT = {
    "py": TextLoader,
    "md": UnstructuredMarkdownLoader,
    "ipynb": NotebookLoader,
}

def upload_embedding_from_file(file_path, ext):
    print("upload_embedding_from_file : ")
    print(file_path)
    if ext == "txt":
        documents = Path(file_path).open("r").readlines()
        documents = parse_kakao_doc_txt(documents)
    else:
        loader = LOADER_DICT.get(file_path.split(".")[-1])
        if loader is None:
            raise ValueError("Not supported file type")
        documents = loader(file_path).load()

    Chroma.from_documents(
        documents,
        OpenAIEmbeddings(),
        collection_name=CHROMA_COLLECTION_NAME,
        persist_directory=CHROMA_PERSIST_DIR,
    )
    print('db success')


def read_embedding_from_file():
    _db = Chroma.from_documents(
        collection_name=CHROMA_COLLECTION_NAME,
        embedding_function=OpenAIEmbeddings(),
        persist_directory=CHROMA_PERSIST_DIR,
    )
    _retriever = _db.as_retriever()
    print('db success')
    return _retriever

"""

카카오 데이터 포맷에 맞춰

category, title, type 으로 나누며 각각
category : 데이터의 기본 카테고리. 카카오싱크, 카카오소셜 등
title : 소제목. 개요, 기능 소개, 과정예시 등
type : 본문의 타입. text, table 등

으로 구성 됨

"""
def parse_markdown_manually(data_in_line: list):
    meta_datas = []
    datas = []
    doc_title = ""
    tmp_data_name = ""
    tmp_data_context = ""

    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)

    for i, v in enumerate(data_in_line):
        meta_data = {"source": "local"}
        v = v.strip()
        # skip empty line
        if len(v) == 0: continue
        # get title
        if i == 0:
            doc_title = v
            continue

        if v[0] == "#":
            # save previous data
            meta_data["category"] = doc_title
            meta_data["title"] = tmp_data_name
            meta_data["type"] = "text"

            if len(tmp_data_context) > 0:
                # split text if it over chunk size
                splitted = splitter.split_text(tmp_data_context)
                for vv in splitted:
                    datas.append(vv)
                    meta_datas.append(meta_data)
            # set new data n keep going
            tmp_data_name = v[1:].strip()
            tmp_data_context = ""
            continue

        # accumulate context
        tmp_data_context += ("\n" + v)

    # save last data
    meta_data = {}
    meta_data["category"] = doc_title
    meta_data["title"] = tmp_data_name
    meta_data["type"] = "text"
    datas.append(tmp_data_context)
    meta_datas.append(meta_data)
    # remove trash
    datas.pop(0)
    meta_datas.pop(0)

    return datas, meta_datas


def parse_kakao_doc_txt(data_in_line: list):
    datas, meta_datas = parse_markdown_manually(data_in_line=data_in_line)
    # for i in range(len(data[0])) :
    #     print(data[0][i], data[1][i])
    #     print("=--=-=-=-=-=-=")
    docs = [
        Document(page_content=data, metadata=meta_data) for data, meta_data in zip(datas, meta_datas)
    ]
    return docs

# def find_kakao_db(query: str, collection_name: str):
#     print("find_kakao_db 호출 : ")
#     client = chromadb.PersistentClient()
#     kakao_sync_collection = client.get_or_create_collection(
#         name=collection_name,
#         metadata={"hnsw:space": "cosine"}
#     )
#
#     query_result = kakao_sync_collection.query(
#         query_texts=[query],
#         n_results=2
#     )
#
#     print("query_result : ", query_result)
#
#     search_results = []
#     for document in query_result['documents'][0]:
#         print("document : ")
#         print(document)
#
#         docu_array = document.split(':')
#
#         search_results.append(
#             {
#                 "content": docu_array[1]
#             }
#         )
#
#     return search_results

def gernerate_answer(user_message, vdb) -> dict[str, str]:
    print(user_message)

    context = dict(user_message=user_message)
    context["input"] = context["user_message"]
    context["intent_list"] = read_prompt_template(INTENT_LIST_TXT)

    intent = parse_intent_chain.run(context)

    print(intent)

    # 질문 그대로 하면안되고, 키워드만 뽑아서 검색 시켜야할듯
    # ex ) 카카오싱크 기능이 무엇이 있는지 알려주세요 -> 카카오싱크 기능
    # context["search_related_documents"] = query_web_search(
    #     context["user_message"]
    # )

    if intent == "kakao_sync" or intent == "kakao_social" or intent == "kakao_channel":
        #context["related_documents"] = find_kakao_db(context["user_message"], intent)
        context["related_documents"] = query_kakao_suppl_info(context["user_message"], vdb)
        answer = ""
        for step in [guide_step1_chain, guide_step2_chain]:
            context = step(context)
            answer += context[step.output_key]
            answer += "\n\n"
    else:
        answer = default_chain.run(context["user_message"])
    return {"answer": answer}

def query_web_search(user_message: str) -> str:
    print("query_web_search :" ,user_message)
    search = GoogleSearchAPIWrapper(
        google_api_key=os.getenv("GOOGLE_API_KEY", "AIzaSyC01H10G_FJbVjApF6QmBVRjDwQM0BPeyQ"),
        google_cse_id=os.getenv("GOOGLE_CSE_ID", "a72b6475e8dc448cf")
    )

    search_tool = Tool(
        name="Google Search",
        description="Search Google for recent results.",
        func=search.run,
    )

    context = {"user_message": user_message}
    context["related_web_search_results"] = search_tool.run(user_message)

    print("결과: " + search_tool.run(user_message))

    has_value = search_value_check_chain.run(context)

    print("query_web_search : ", has_value)
    if has_value == "Y":
        return search_compression_chain.run(context)
    else:
        return ""

def callback_handler(request: ChatbotRequest) -> dict:
    #setting_vectorDb()
    vdb = create_vectordb()
    upload_embeddings_from_dir("/Users/colin/llmProject/pythonProject1/resources/datas/")
    return gernerate_answer(request.userRequest.utterance, vdb)

