import os
import re
from typing import Any

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

# es=Elasticsearch([{'host':'localhost','port':9200}])
# es.sql.query(body={'query': 'select * from global_res_todos_acco...'})
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


toolList = ["booksearch", "cannot", "elastic_test"]

def interact(webinput_queue, weboutput_queue, modelChoice_queue, user_id):
    chatturn = 0
    recommended_isbn = list()
    # region logging setting
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
    # endregion
    print("start interact!")

    # region setting&init
    with open("config.json") as f:
        config = json.load(f)

    web_output: str
    input_query: str
    elasticsearch_url = config["elasticsearch_url"]
    retriever = ElasticSearchBM25Retriever(
        elasticsearch.Elasticsearch(
            elasticsearch_url,
            verify_certs=False,
        ),
        "600k",
    )


    # tool that process elasticsearch based on the book's simple information(title, author, publisher)
    class booksearch_Tool(BaseTool):
        name = "booksearch"
        description = (
            "Use this tool to search for simple information about a book you have already found." 
            "It searches for book titles, authors, publishers, and ISBN." 
            "Input can be a single title, author, or publisher, but you must explicitly state what you are searching for. " 
            "Use 'author:' followed by the name of the book's author, 'publisher:' followed by the name of the book's publisher, or 'title:' followed by the name of the book's title." 
            "The format for the Final Answer should be (number) title: [Book's Title], author: [Book's Author], publisher: [Book's Publisher]."
        )

        # Goes into different search function based on the information the user gave to the tool.
        def _run(self, query: str):
            print("\nbook_search")
            if "author: " in query:
                print("\n=====author=====")
                result = retriever.get_author_info(query)
            elif "publisher: " in query:
                print("\n=====publisher=====")
                result = retriever.get_publisher_info(query)
            elif "title: " in query:
                print("\n=====title=====")
                result = retriever.get_title_info(query)
            return f"{result} I should give final answer based on these information. "

        def _arun(self, query: str):
            raise NotImplementedError("This tool does not support async")
        
    class translate(BaseTool):
        default_num = 3
        name = "translate"
        description = (
            "Use this tool when the query from the user is not one of (English, Korean). "
            f"Format for the Observation must be (query, number of books to recommend) if the number is specified, otherwise (query , {default_num})."
        )

        def _run(self, query: str):
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a translator."
                            f"Translate the {query} to Korean language. Also, summarize them into keywords(just in Korean) without any formats. "
                        ),
                    }
                ],
            )
            return completion["choices"][0]["message"]["content"]
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
            # 강제 출력하려면 주석해제
            # nonlocal web_output
            # web_output = result
            result += "Thought:Couldn't perform task. I must inform user.\n"
            result += "Final Answer: "
            return result

        def _arun(self, query: str):
            raise NotImplementedError("This tool does not support async")

    # tool that recommends books for the user based on the user's query
    # The tool first searches the books for the user, and then checks if the found books are valid for the results
    # Finally, the tool creates the reason for the recommendation for the user and prints the result to the user(web).
    class elastic_Tool(BaseTool):
        name = "elastic_test"

        default_num = config["default_number_of_books_to_return"]
        default_language = config["default_language_for_explaination"]

        description = (
            "Use this tool only for recommending books to users in Korean. "
            "Don't use it for unrelated queries. "
            f"Format for the Action input must be (query, number of books to recommend) if the number is specified, otherwise (query , {default_num})."
            "REMEMBER YOU SHOULD FOLLOW THE FORMAT!"
            "Final Answer format: (number) title: [Book's Title], author: [Book's Author], publisher: [Book's Publisher]." 
            "Input may include the year."
        )

        # The action input for the tool must be in a format of (summary of user's query, number of books the user wants to get)
        # Therefore to get variables inside (), the tool uses extract_variables.
        def extract_variables(self, input_string: str):
            variables_list = input_string.strip("()\n").split(", ")
            name = variables_list[0]
            num = int(variables_list[1])
            return name, num

        # The function that filters out the books that are already recommended to the user.
        # Everytime the tool recommends the book to the user, recommended_isbn memorizes the book's isbn.
        # In the function, it compares those isbn to candidates for recommendations. If the isbn already exists in the list, the book would be excluded from the result
        def filter_recommended_books(self, result):
            filtered_result = []
            for book in result:
                if book.isbn not in [item["isbn"] for item in recommended_isbn]:
                    filtered_result.append(book)
                else:
                    print("\nalready recommended this book!")
                    print(book.title)
                    print("\n")
            return filtered_result

        # I must give Final Answer base
        def _run(self, query: str):
            elastic_input, num= self.extract_variables(query)

            nonlocal input_query
            nonlocal web_output

            # List that holds isbn of books that are recommended to the user.
            recommendList = list()
            recommendList.clear()

            # List that holds the information of books that would be shown to the langchain agent.
            bookList = list()
            bookList.clear()

            count = 0

            # Function that checks the books whether they are valid for the final result for the user based on the (user's query, title of the book, introduction of the book)
            def isbookPass(userquery: str, bookinfo) -> bool:
                logger.info("---------------knn, bm25----------------")
                logger.info(bookinfo)
                logger.info("----------------------------------------\n")
                try:
                    completion = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {
                                "role": "system",
                                "content": (
                                    "Based on the user's question about {desired type of book} and the provided information about the recommended book {recommended book information}, evaluate the recommendation. " 
                                    "Explain the alignment between the user's request and the recommended book, providing supporting reasons." 
                                    "Conclude with your evaluation as 'Evaluation: P' (Positive) or 'Evaluation: F' (Negative). "
                                    "If the evaluation is unclear or the recommended book doesn't directly address the user's request, default to 'Evaluation: F'. "
                                    "Ensure no sentences follow the evaluation result."
                                ),
                            },
                            {
                                "role": "user",
                                "content": f"user question:{userquery} recommendations:{bookinfo}",
                            },
                        ],
                    )
                except openai.error.APIError as e:
                    pf = "F"
                    logger.error(f"OpenAI API returned an API Error: {e}")
                    print(f"OpenAI API returned an API Error: {e}")
                    pass
                except openai.error.APIConnectionError as e:
                    pf = "F"
                    logger.error(f"Failed to connect to OpenAI API: {e}")
                    print(f"Failed to connect to OpenAI API: {e}")
                    pass
                except openai.error.RateLimitError as e:
                    pf = "F"
                    logger.error(f"OpenAI API request exceeded rate limit: {e}")
                    print(f"OpenAI API request exceeded rate limit: {e}")
                    pass
                except:
                    pf = "F"
                    logger.error("Unknown error while evaluating")
                    print("Unknown error while evaluating")
                    pass
                logger.info(completion["choices"][0]["message"]["content"])

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
            result = self.filter_recommended_books(result)

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
                    bookList.append(
                        {
                            "author": book.author,
                            "publisher": book.publisher,
                            "title": book.title,
                            "isbn": book.isbn,
                        }
                    )
                for i in range(num):
                    recommended_isbn.append(
                        {
                            "turnNumber": chatturn,
                            "author": recommendList[i].author,
                            "publisher": recommendList[i].publisher,
                            "title": recommendList[i].title,
                            "isbn": recommendList[i].isbn,
                        }
                    )
            else:
                while len(recommendList) < num and count < len(
                    result
                ):  # 총 num개 찾을때까지 PF...
                    if isbookPass(input_query, result[count]):
                        recommendList.append(result[count])
                        # 가져온 도서데이터에서 isbn, author, publisher만 list에 appned
                        recommended_isbn.append(
                            {
                                "turnNumber": chatturn,
                                "author": result[count].author,
                                "publisher": result[count].publisher,
                                "title": result[count].title,
                                "isbn": result[count].isbn,
                            }
                        )

                        bookList.append(
                            {
                                "author": result[count].author,
                                "publisher": result[count].publisher,
                                "title": result[count].title,
                                "isbn": result[count].isbn,
                            }
                        )
                    count += 1
            print(f"\n{recommended_isbn}")
            print(f"\neval done in thread{threading.get_ident()}")
            

            # The part where the language model creates the reason for the recommendation that would be shown to the user in the web. 
            if len(recommendList) >= num:
                completion = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                f"As a recommendation explainer, I provide {num} book recommendations, explaining their relevance and adequacy based on provided data without making up information." 
                                f"Each book is explained in one sentence using language: {self.default_language}."
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
                return f"less then {num} pass found"

        def _arun(self, radius: int):
            raise NotImplementedError("This tool does not support async")

    tools = [elastic_Tool(), cannot_Tool(), booksearch_Tool(), translate()]

    prefix = """Have a conversation with a human, answering questions using the provided tools."""
    suffix = """
    For daily conversation, avoid using any tools. Keep in mind that the current year is 2023.
    The eligible tools for Action are elastic, cannot, booksearch only.
    Finish the chain right after using the elastic tool.
    If the query is not English or Korean, you should use translate tool first. 
    Begin!
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
        chatturn += 1
