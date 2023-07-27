from flask import Flask, render_template, request, session
from fullOpenAI import interactgpt3
from opensourceLLM_generate import interactOpensourceLLM
import threading
import queue
import uuid
import json

app = Flask(__name__)
app.secret_key = "12341234"  # temporary secret key
idthreadDict = {}
input_queue_dict = {}
modelChoice_queue_dict = {}
output_queue_dict = {}
with open("config.json") as f:
    config = json.load(f)


def generate_user_id():
    return str(uuid.uuid4())


@app.route("/demo")
def home():
    global input_queue_dict
    global modelChoice_queue_dict
    global output_queue_dict
    global idthreadDict
    if "user_id" in session:
        user_id = session["user_id"]
    else:
        # generate new user_id and store
        user_id = generate_user_id()
        session["user_id"] = user_id
        print(f"process got userid of : {user_id}")
        print(f"new session for user : {user_id}")

    input_queue_dict[user_id] = queue.Queue()
    output_queue_dict[user_id] = queue.Queue()
    modelChoice_queue_dict[user_id] = queue.Queue()

    # start server-side loop in separate thread
<<<<<<< Updated upstream
    server_thread = threading.Thread(
        target=interact,
        args=(
            input_queue_dict[user_id],
            output_queue_dict[user_id],
            modelChoice_queue_dict[user_id],
            user_id,
        ),
    )
=======
    if config["modelchoice"] == "openai":
        server_thread = threading.Thread(
            target=interactgpt3,
            args=(
                input_queue_dict[user_id],
                output_queue_dict[user_id],
                langchoice_queue_dict[user_id],
                user_id,
            ),
        )
    elif config["modelchoice"] == "opensource_LLM":
        server_thread = threading.Thread(
            target=interactOpensourceLLM,
            args=(
                input_queue_dict[user_id],
                output_queue_dict[user_id],
                langchoice_queue_dict[user_id],
                user_id,
            ),
        )
>>>>>>> Stashed changes
    server_thread.daemon = True
    server_thread.start()
    print(f"thread id {server_thread} started for user {user_id}")
    idthreadDict[user_id] = server_thread
    return render_template("demo.html", user_id=user_id)


@app.route("/process", methods=["POST"])
def process():
    global input_queue_dict
    global modelChoice_queue_dict
    global output_queue_dict
    global idthreadDict
    print("in process")
    # check user_id already in session
    if "user_id" in session:
        user_id = session["user_id"]
    else:
        # generate new user_id and store
        # should not happen basically
        user_id = generate_user_id()
        session["user_id"] = user_id

    input_data = request.form["inputField"]
    print(f"user input : {input_data}")
    # put user input into input queue
    input_queue_dict[user_id].put(input_data)
    model_choice = request.form["dropdown"]
    print(f"model choice : {model_choice}")
    modelChoice_queue_dict[user_id].put(model_choice)

    # wait output from the server-side loop

    output = output_queue_dict[user_id].get()

    return output


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80)
