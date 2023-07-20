import os
import re

# api keys go here
import keys

OPENAI_API_KEY = keys.OPENAI_API_KEY
HUGGINGFACEHUB_API_TOKEN = keys.HUGGINGFACEHUB_API_TOKEN
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN
import random
import openai
import elasticsearch
import threading

# modified langchain.chat_models ChatOpenAI
from modifiedLangchainClasses.openai import ChatOpenAI


from langchain import LLMChain


from langchain.tools import DuckDuckGoSearchRun
from langchain.tools import BaseTool

# modified langchain.retrievers ElasticSearchBM25Retriever
from modifiedLangchainClasses.elastic_search_bm25 import ElasticSearchBM25Retriever

from langchain.agents import initialize_agent
from langchain.agents import ZeroShotAgent, Tool, AgentExecutor

from langchain.memory import ConversationBufferMemory
from langchain.memory import ConversationBufferWindowMemory

import queue
import logging
import json


def interact(webinput_queue, weboutput_queue, modelChoice_queue, user_id):
    log_file_path = f"log_from_user_{user_id}.log"

    # logger for each thread
    logger = logging.getLogger(f"UserID-{user_id}")
    logger.setLevel(logging.INFO)

    # file handler for each thread
    file_handler = logging.FileHandler(log_file_path)
    formatter = logging.Formatter(
        "%(asctime)s [%(threadName)s - %(thread)s] %(levelname)s: %(message)s"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    print("start interact!")

    # region setting&init
    with open("config.json") as f:
        config = json.load(f)

    web_output: str
    input_query: str
    elasticsearch_url = "https://25cc-115-145-212-85.ngrok-free.app:443"
    retriever = ElasticSearchBM25Retriever(
        elasticsearch.Elasticsearch(elasticsearch_url), "600k"
    )

    # TODO n번째로 addressing하지 않는경우...
    # class isbnsearch_Tool(BaseTool) :
    #     name = "search by isbn"
    #     description = (
    #         "You can use this tool if you need simple information about the book. "
    #         "This tool searches book's title, author, and publisher by isbn "
    #         "This tool must not be used before or after the elastic tool is used. "
    #         "The input to this tool must be isbn of the book. "
    #     )

    #     def _run(self, query: str):
    #         print("\nisbn_search")
    #         result = retriever.isbn_to_book(query)
    #         return result
    #     def _arun(self, radius: int):
    #         raise NotImplementedError("This tool does not support async")

    class booksearch_Tool(BaseTool):
        name = "booksearch"
        description = (
            "Use this tool when searching based on brief information about a book you have already found. "
            "Use this tool to get simple information about books. "
            "This tool searches book's title, author, publisher and isbn. "
            "Input to this tool can be single title, author, or publisher without any format. "
            "The format for the Final Answer should be (number) title : book's title, author :  book's author, pubisher :  book's publisher. "
        )

        def _run(self, query: str):
            print("\nbook_search")
            result = retriever.get_book_info(query)
            return f"{result} I should give final answer based on these information. "

        def _arun(self, query: str):
            raise NotImplementedError("This tool does not support async")

    # tool that says cannot perform task
    class cannot_Tool(BaseTool):
        name = "cannot"
        description = (
            "Use this tool when there are no available tool to fulfill user's request. "
        )

        def _run(self, query: str):
            result = "Cannot perform task. "
            print(result)
            global web_output
            web_output = result
            result += "Chain finished. I now know final answer. "
            return result

        def _arun(self, query: str):
            raise NotImplementedError("This tool does not support async")

    # class elastic_Tool(BaseTool):
    #     name = "elastic"
    #     description = (
    #         "You must only use this tool when you recommend books for users. "
    #         "You must never use this tool with queries not related to this tool. "
    #         "You must not use this tool unless the user questions about the book. "
    #         "You should always pass query as korean language. "
    #         "You should be conservative when passing action input. Try not to miss out any keywords. "
    #         "If you found some books, you should give Final Answer based on the books found. The Final answer must include all the books found.  "
    #         "The format for the Final Answer should be (number) title : book's title, author :  book's author, pubisher :  book's publisher"
    #     )

    #     def _run(self, query: str):
    #         nonlocal input_query
    #         nonlocal web_output


    #         default_num = 3
    #         num = default_num
    #         pattern = r'\b(?<!\S)(\d{1,2})(?=(?:권|개)\b)'
    #         matches = re.finditer(pattern, input_query)

    #         for match in matches:
    #             num = int(match.group(1))
    #             print(num)
            
    #         print("\n----debug number----")
    #         print(num)
    #         print("----debug number----\n")

    #         recommendList = list()
    #         recommendList.clear()
    #         bookList = list()
    #         bookList.clear()
    #         count = 0

    #         def isbookPass(userquery: str, bookinfo) -> bool:
    #             logger.info("---------------knn, bm25----------------")
    #             logger.info(bookinfo)
    #             logger.info("----------------------------------------\n")
    #             completion = openai.ChatCompletion.create(
    #                 model="gpt-3.5-turbo",
    #                 messages=[
    #                     {
    #                         "role": "system",
    #                         "content": (
    #                             "Based on the user's question {user's question about the desired type of book} "
    #                             "and the provided information about the recommended book {recommended book information}, evaluate the recommendation. "
    #                             "Does the recommended book fulfill the user's requirements? "
    #                             "Please provide an explaination first and evaluation in the format P or F at the end. "
    #                             "If the evaluation is unclear, please provide a brief justification and default to F."
    #                         ),
    #                     },
    #                     {
    #                         "role": "user",
    #                         "content": f"user question:{userquery} recommendations:{bookinfo}",
    #                     },
    #                 ],
    #             )

    #             logger.info("-----------evaluation------------")
    #             logger.info(completion["choices"][0]["message"]["content"])
    #             logger.info("--------------------------------\n")

    #             pf = str(completion["choices"][0]["message"]["content"])
    #             ck = False
    #             for c in reversed(pf):
    #                 if c == "P":
    #                     return True
    #                 elif c == "F":
    #                     return False
    #             if ck == False:
    #                 print("\nsmth went wrong\n")
    #                 return False

    #         result = retriever.get_relevant_documents(query)
    #         if config["enable_simultaneous_evaluation"]:
    #             bookresultQueue = queue.Queue()

    #             def append_list_thread(userquery: str, bookinfo):
    #                 nonlocal bookresultQueue
    #                 if isbookPass(userquery, bookinfo):
    #                     bookresultQueue.put(bookinfo)
    #                 return

    #             threadlist = []
    #             for book in result:
    #                 t = threading.Thread(
    #                     target=append_list_thread, args=(input_query, book)
    #                 )
    #                 threadlist.append(t)
    #                 t.start()

    #             for t in threadlist:
    #                 t.join()

    #             while not bookresultQueue.empty():
    #                 book = bookresultQueue.get()
    #                 recommendList.append(book)
    #                 # 가져온 도서데이터에서 isbn, author, publisher만 list에 appned
    #                 bookList.append(
    #                     {
    #                         "author": book.author,
    #                         "publisher": book.publisher,
    #                         "title": book.title,
    #                         "isbn": book.isbn,
    #                     }
    #                 )
    #                 print(book)
    #         else:
    #             while len(recommendList) < num and count < len(
    #                 result
    #             ):  # 총 num개 찾을때까지 PF...
    #                 if isbookPass(input_query, result[count]):
    #                     recommendList.append(result[count])
    #                     # 가져온 도서데이터에서 isbn, author, publisher만 list에 appned
    #                     bookList.append(
    #                         {
    #                             "author": result[count].author,
    #                             "publisher": result[count].publisher,
    #                             "title": result[count].title,
    #                             "isbn": result[count].isbn,
    #                         }
    #                     )
    #                     print(result[count])
    #                 count += 1
    #         print(f"\neval done in thread{threading.get_ident()}")
    #         # 최종 출력을 위한 설명 만들기
    #         if len(recommendList) >= num:
    #             completion = openai.ChatCompletion.create(
    #                 model="gpt-3.5-turbo",
    #                 messages=[
    #                     {
    #                         "role": "system",
    #                         "content": (
    #                             "You are a recommendation explainer. "
    #                             "You take a user request and three recommendations and explain why they were recommeded in terms of relevance and adequacy. "
    #                             "You should not make up stuff and explain grounded on provided recommendation data. "
    #                             "You should explain in korean language(한국어)"
    #                         ),
    #                     },
    #                     {
    #                         "role": "user",
    #                         "content": f"user question:{input_query} recommendations:{recommendList[0:num]}",
    #                     },
    #                 ],
    #             )

    #             logger.info("--------------explainer-------------------")
    #             logger.info(completion["choices"][0]["message"]["content"])
    #             logger.info("------------------------------------------\n")
    #             web_output = completion["choices"][0]["message"]["content"]
    #             logger.info(f"web output set to {web_output}")
    #             return f"{bookList[0:num]}  "
    #         else:
    #             print(
    #                 f"smth went wrong: less then {num} pass found in thread{threading.get_ident()}"
    #             )
    #             return "less then three books found"

    #     def _arun(self, radius: int):
    #         raise NotImplementedError("This tool does not support async")

    class elastic_Tool(BaseTool):
        name = "elastic_test"
        default_num = 3
        description = (
            "You must only use this tool when you recommend books for users. "
            "You must never use this tool with queries not related to this tool. "
            "You must not use this tool unless the user questions about the book. "
            "You should always pass query as korean language. "
            "You should be conservative when passing action input. Try not to miss out any keywords. "
            f"The format for the Action input should be (query, number of books to recommend(If the user specifies about the number. The default number should be {default_num}))"
            "If you found some books, you should give Final Answer based on the books found. The Final answer must include all the books found.  "
            "The format for the Final Answer should be (number) title : book's title, author :  book's author, publisher :  book's publisher"
        )
        def extract_variables(self, input_string:str):
            variables_list = input_string.strip('()').split(', ')
            # 리스트의 요소를 변수로 저장
            name = variables_list[0]
            num = int(variables_list[1])  # 숫자로 저장하려면 형변환 필요
            return name, num
        
    #I must give Final Answer base
        def _run(self, query: str):

            elastic_input, num = self.extract_variables(query)
            
            nonlocal input_query
            nonlocal web_output




            recommendList = list()
            recommendList.clear()
            bookList = list()
            bookList.clear()
            count = 0

            def isbookPass(userquery: str, bookinfo) -> bool:
                logger.info("---------------knn, bm25----------------")
                logger.info(bookinfo)
                logger.info("----------------------------------------\n")
                completion = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "Based on the user's question {user's question about the desired type of book} "
                                "and the provided information about the recommended book {recommended book information}, evaluate the recommendation. "
                                "Does the recommended book fulfill the user's requirements? "
                                "Please provide an explaination first and evaluation in the format P or F at the end. "
                                "If the evaluation is unclear, please provide a brief justification and default to F."
                            ),
                        },
                        {
                            "role": "user",
                            "content": f"user question:{userquery} recommendations:{bookinfo}",
                        },
                    ],
                )

                logger.info("-----------evaluation------------")
                logger.info(completion["choices"][0]["message"]["content"])
                logger.info("--------------------------------\n")

                pf = str(completion["choices"][0]["message"]["content"])
                ck = False
                for c in reversed(pf):
                    if c == "P":
                        return True
                    elif c == "F":
                        return False
                if ck == False:
                    print("\nsmth went wrong\n")
                    return False

            result = retriever.get_relevant_documents(elastic_input)
            if config["enable_simultaneous_evaluation"]:
                bookresultQueue = queue.Queue()

                def append_list_thread(userquery: str, bookinfo):
                    nonlocal bookresultQueue
                    if isbookPass(userquery, bookinfo):
                        bookresultQueue.put(bookinfo)
                    return

                threadlist = []
                for book in result:
                    t = threading.Thread(
                        target=append_list_thread, args=(input_query, book)
                    )
                    threadlist.append(t)
                    t.start()

                for t in threadlist:
                    t.join()

                while not bookresultQueue.empty():
                    book = bookresultQueue.get()
                    recommendList.append(book)
                    # 가져온 도서데이터에서 isbn, author, publisher만 list에 appned
                    bookList.append(
                        {
                            "author": book.author,
                            "publisher": book.publisher,
                            "title": book.title,
                            "isbn": book.isbn,
                        }
                    )
                    print(book)
            else:
                while len(recommendList) < num and count < len(
                    result
                ):  # 총 num개 찾을때까지 PF...
                    if isbookPass(input_query, result[count]):
                        recommendList.append(result[count])
                        # 가져온 도서데이터에서 isbn, author, publisher만 list에 appned
                        bookList.append(
                            {
                                "author": result[count].author,
                                "publisher": result[count].publisher,
                                "title": result[count].title,
                                "isbn": result[count].isbn,
                            }
                        )
                        print(result[count])
                    count += 1
            print(f"\neval done in thread{threading.get_ident()}")
            # 최종 출력을 위한 설명 만들기
            if len(recommendList) >= num:
                completion = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a recommendation explainer. "
                                "You take a user request and three recommendations and explain why they were recommeded in terms of relevance and adequacy. "
                                "You should not make up stuff and explain grounded on provided recommendation data. "
                                "You should explain in korean language(한국어)"
                            ),
                        },
                        {
                            "role": "user",
                            "content": f"user question:{input_query} recommendations:{recommendList[0:num]}",
                        },
                    ],
                )

                logger.info("--------------explainer-------------------")
                logger.info(completion["choices"][0]["message"]["content"])
                logger.info("------------------------------------------\n")
                web_output = completion["choices"][0]["message"]["content"]
                logger.info(f"web output set to {web_output}")
                return f"{bookList[0:num]}  "
            else:
                print(
                    f"smth went wrong: less then {num} pass found in thread{threading.get_ident()}"
                )
                return "less then three books found"

        def _arun(self, radius: int):
            raise NotImplementedError("This tool does not support async")


    tools = [elastic_Tool(), cannot_Tool(), DuckDuckGoSearchRun(), booksearch_Tool()]

    prefix = """Have a conversation with a human, answering the following questions as best you can. You have access to the following tools:"""
    suffix = """For daily conversation, try not to use any tools. The name of the tool that can be entered into Action can only be elastic, cannot, booksearch, and duckduckko_search. If the user asks for recommendation of books, you should answer with just title, author, and publisher. You must finish the chain right after elastic tool is used. Begin!
    {chat_history}
    Question: {input}
    {agent_scratchpad}"""

    # memory
    prompt = ZeroShotAgent.create_prompt(
        tools,
        prefix=prefix,
        suffix=suffix,
        input_variables=["input", "chat_history", "agent_scratchpad"],
    )

    memory = ConversationBufferMemory(memory_key="chat_history")

    llm_chain = LLMChain(llm=ChatOpenAI(temperature=0.4), prompt=prompt)

    agent = ZeroShotAgent(
        llm_chain=llm_chain,
        tools=tools,
        verbose=True,
    )

    agent_chain = AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        verbose=True,
        memory=memory,
        handle_parsing_errors=True,
    )
    # endregion

    while 1:
        webinput = webinput_queue.get()
        modelchoice = modelChoice_queue.get()
        input_query = webinput
        web_output = None
        print("GETTING WEB INPUT")
        logger.warning(f"USERINPUT : {webinput}")
        if modelchoice == "openai":
            if webinput == "stop":
                break
            else:
                chain_out = agent_chain.run(input=webinput)
            print(f"PUTTING WEB OUTPUT in thread{threading.get_ident()}")
            if web_output is None:
                weboutput_queue.put(chain_out)
                logger.warning(f"OUTPUT : {chain_out}")
            else:
                weboutput_queue.put(web_output)
                logger.warning(f"OUTPUT : {web_output}")
        elif modelchoice == "option2":
            # TODO:run some other agent_chain
            print(f"PUTTING WEB OUTPUT in thread{threading.get_ident()}")
            # put chain out
            weboutput_queue.put(f"option2 WIP <=> {input_query}")
        elif modelchoice == "option3":
            # TODO:run some other agent_chain
            print(f"PUTTING WEB OUTPUT in thread{threading.get_ident()}")
            # put chain out
            weboutput_queue.put(f"option3 WIP <=> {input_query}")
