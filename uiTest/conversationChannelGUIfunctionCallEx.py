import json
import openai
import tkinter as tk
import pandas as pd
from tkinter import scrolledtext
import tkinter.filedialog as filedialog
import os
import chromadb

openai.api_key = os.environ['OPENAI_API_KEY']

def send_message(message_log, functions, gpt_model="gpt-3.5-turbo", temperature=0.1):
    response = openai.ChatCompletion.create(
        model=gpt_model,
        messages=message_log,
        temperature=temperature,
        functions=functions,
        function_call='auto',
    )

    response_message = response["choices"][0]["message"]

    print("response_message : ")
    print(response_message)

    if response_message.get("function_call"):
        available_functions = {
            "find_kakao_db": find_kakao_db,
        }
        function_name = response_message["function_call"]["name"]
        fuction_to_call = available_functions[function_name]
        function_args = json.loads(response_message["function_call"]["arguments"])
        # 사용하는 함수에 따라 사용하는 인자의 개수와 내용이 달라질 수 있으므로
        # **function_args로 처리하기
        function_response = fuction_to_call(**function_args)

        print("function_name : " + function_name)
        print("fuction_to_call :")
        print(fuction_to_call)
        print("function_args :")
        print(function_args)
        print("function_response : ")
        print(function_response)

        json_data = json.dumps(function_response, ensure_ascii=False)

        print(function_response)

        # 함수를 실행한 결과를 GPT에게 보내 답을 받아오기 위한 부분
        message_log.append(response_message)  # GPT의 지난 답변을 message_logs에 추가하기
        message_log.append(
            {
                "role": "function",
                "name": function_name,
                "content": json_data,
            }
        )  # 함수 실행 결과도 GPT messages에 추가하기

        print("message_log : ", message_log)

        response = openai.ChatCompletion.create(
            model=gpt_model,
            messages=message_log,
            temperature=temperature,
        )  # 함수 실행 결과를 GPT에 보내 새로운 답변 받아오기

        print("response : ")
        print(response)
        print(response.choices[0].message.content)

    return response.choices[0].message.content

def find_kakao_db(query: str):
    client = chromadb.PersistentClient()
    kakao_channel_collection = client.get_or_create_collection(
        name="kakao-channel",
        metadata={"hnsw:space": "cosine"}
    )

    query_result = kakao_channel_collection.query(
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
                "title": docu_array[0],
                "content": docu_array[1]
            }
        )

    return search_results

def initial_setting():
    # 파일 읽기 및 정리
    file = open('/Users/colin/llmProject/pythonProject1/uiTest/resources/project_data_카카오톡채널.txt', 'r',
                encoding='utf-8')
    contents = file.read()

    datas = []
    dataArray = contents.split('\n#')[1:]

    for data in dataArray:
        title, content = map(str.strip, data.split('\n', 1))
        contentArray = content.split('\n')
        for text in contentArray:
            if text == "":
                continue
            else:
                document = f"{title}:{text}"
                datas.append(document)

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
    client.delete_collection(name="kakao-channel")

    kakao_channel_collection = client.get_or_create_collection(
        name="kakao-channel",
        metadata={"hnsw:space": "cosine"}
    )

    # DB 저장
    kakao_channel_collection.add(
        documents=documents,
        ids=ids
    )

def main():
    #파일 읽기 및 저장
    initial_setting()

    message_log = [
        {
            "role": "system",
            "content": '''
            You are a Kakao chatbot.
            If you have any questions about Kakao, please answer them as kindly as possible
            Please answer in Korean
            Please answer step by step so that others can understand. Please check again step by step if the answer is correct."
            '''
        }
    ]

    functions = [
        {
            "name": "find_kakao_db",
            "description": "Answer the questions related to Kakao",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What are you curious about among the information related to Kakao channel",
                    },
                },
                "required": ["query"],
            },
        }
    ]

    def show_popup_message(window, message):
        popup = tk.Toplevel(window)
        popup.title("")

        # 팝업 창의 내용
        label = tk.Label(popup, text=message, font=("맑은 고딕", 12))
        label.pack(expand=True, fill=tk.BOTH)

        # 팝업 창의 크기 조절하기
        window.update_idletasks()
        popup_width = label.winfo_reqwidth() + 20
        popup_height = label.winfo_reqheight() + 20
        popup.geometry(f"{popup_width}x{popup_height}")

        # 팝업 창의 중앙에 위치하기
        window_x = window.winfo_x()
        window_y = window.winfo_y()
        window_width = window.winfo_width()
        window_height = window.winfo_height()

        popup_x = window_x + window_width // 2 - popup_width // 2
        popup_y = window_y + window_height // 2 - popup_height // 2
        popup.geometry(f"+{popup_x}+{popup_y}")

        popup.transient(window)
        popup.attributes('-topmost', True)

        popup.update()
        return popup

    def on_send():
        user_input = user_entry.get()
        user_entry.delete(0, tk.END)

        if user_input.lower() == "quit":
            window.destroy()
            return

        message_log.append({"role": "user", "content": user_input})
        conversation.config(state=tk.NORMAL)  # 이동
        conversation.insert(tk.END, f"You: {user_input}\n", "user")  # 이동
        thinking_popup = show_popup_message(window, "처리중...")
        window.update_idletasks()
        # '생각 중...' 팝업 창이 반드시 화면에 나타나도록 강제로 설정하기
        response = send_message(message_log, functions)

        thinking_popup.destroy()

        message_log.append({"role": "assistant", "content": response})

        # 태그를 추가한 부분(1)
        conversation.insert(tk.END, f"gpt assistant: {response}\n", "assistant")
        conversation.config(state=tk.DISABLED)
        # conversation을 수정하지 못하게 설정하기
        conversation.see(tk.END)

    window = tk.Tk()
    window.title("GPT AI")

    font = ("맑은 고딕", 10)

    conversation = scrolledtext.ScrolledText(window, wrap=tk.WORD, bg='#f0f0f0', font=font)
    # width, height를 없애고 배경색 지정하기(2)
    conversation.tag_configure("user", background="#c9daf8")
    # 태그별로 다르게 배경색 지정하기(3)
    conversation.tag_configure("assistant", background="#e4e4e4")
    # 태그별로 다르게 배경색 지정하기(3)
    conversation.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    # 창의 폭에 맞추어 크기 조정하기(4)

    input_frame = tk.Frame(window)  # user_entry와 send_button을 담는 frame(5)
    input_frame.pack(fill=tk.X, padx=10, pady=10)  # 창의 크기에 맞추어 조절하기(5)

    user_entry = tk.Entry(input_frame)
    user_entry.pack(fill=tk.X, side=tk.LEFT, expand=True)

    send_button = tk.Button(input_frame, text="Send", command=on_send)
    send_button.pack(side=tk.RIGHT)

    window.bind('<Return>', lambda event: on_send())
    window.mainloop()


if __name__ == "__main__":
    main()
