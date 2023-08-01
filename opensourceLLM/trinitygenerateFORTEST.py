import re
from transformers import pipeline
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from peft import PeftModel
from peft import LoraConfig
from peft import TaskType
from peft import get_peft_model
import torch

PROMPT_DICT = {
    "prompt_input": (
        "### Prompt(명령):\n{prompt}\n\n### Input(입력):\n{input}\n\n### Response(응답):\n"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task.\n"
        "아래는 작업을 설명하는 명령어입니다.\n\n"
        "Write a response that appropriately completes the request.\n명령어에 따른 요청을 적절히 완료하는 응답을 작성하세요.\n\n"
        "### Instruction(명령어):\n{prompt}\n\n### Response(응답):"
    ),
}
USER_QUERY_PROMPT = "input에 주어진 사용자 질의에 응답하는 문구를 생성해줘"
BOOK_RECOMMENDATION_PROMPT = "input에 주어진 책 정보와 사용자 질의를 바탕으로 그 책을 추천한 사유를 생성해줘"

MODEL_ID = "skt/ko-gpt-trinity-1.2B-v0.5"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"

model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID, padding_side="right", model_max_length=512
)
tokenizer.add_special_tokens(
    {
        "eos_token": DEFAULT_EOS_TOKEN,
        "bos_token": DEFAULT_BOS_TOKEN,
        "unk_token": DEFAULT_UNK_TOKEN,
    }
)
tokenizer.pad_token = tokenizer.eos_token
config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=16, lora_dropout=0.05, bias="none"
)

# model = get_peft_model(model, config)
model = PeftModel.from_pretrained(
    model=model,
    model_id="/home/work/ColossalAI_Test/lora_results/trinity_models/n_80_r_8_alpha_16_book_input_completion_199_user_query_215/checkpoint-480/",
)
# model = model.base_model.model


def generate_recommendation_start(user_input: str, device: str):
    torch.cuda.set_device(device)
    print("user input in trinitygenerate : " + user_input)
    generator = pipeline(
        "text-generation", model=model, tokenizer=tokenizer, device=device
    )

    generation_args = dict(
        num_beams=4,
        repetition_penalty=2.0,
        no_repeat_ngram_size=4,
        max_new_tokens=512,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_k=50,
        early_stopping=True,
    )
    mapped_prompt = PROMPT_DICT["prompt_input"].format_map(
        {"prompt": USER_QUERY_PROMPT, "input": user_input}
    )
    response = generator(mapped_prompt, **generation_args)
    result = (response[0]["generated_text"]).replace(mapped_prompt, "")
    # initial msg
    print("-----------generation------------")
    print(result)
    print("---------------------------------")
    return result


def generate_recommendation_book(user_input: str, book: str, device: str):
    torch.cuda.set_device(device)
    print("user input in trinitygenerate book: " + user_input)
    print("book in trinitygenerate book: " + book)
    generator = pipeline(
        "text-generation", model=model, tokenizer=tokenizer, device=device
    )

    generation_args = dict(
        num_beams=4,
        repetition_penalty=2.0,
        no_repeat_ngram_size=4,
        max_new_tokens=512,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_k=50,
        early_stopping=True,
    )
    # book dataset of book data
    book_with_prompt = PROMPT_DICT["prompt_input"].format_map(
        {
            "prompt": BOOK_RECOMMENDATION_PROMPT,
            "input": "user_query: {" + user_input + "}, book: {" + book + "}",
        }
    )

    book_result = generator(book_with_prompt, **generation_args)

    pattern = r"title:\s*\[([^]]+)\],\s*author:\s*\[([^]]+)\]"

    title_and_author_result = re.findall(pattern, book_with_prompt)
    final_result = (
        "["
        + title_and_author_result[0][0]
        + "] ("
        + title_and_author_result[0][1]
        + " 저)"
        + "<br>"
    )
    # title and author
    final_result += book_result[0]["generated_text"].replace(book_with_prompt, "")
    # book reccommendation
    # print(final_result)
    return final_result
