from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline
from peft import PeftModel


# function to generate intention from user input string
def determine_intention(model, tokenizer, user_input: str):
    generator = pipeline('text-generation', model=model, tokenizer=tokenizer, device=0)

    generation_args = dict(
        num_beams=2,  # beam search. 2 Beams
        repetition_penalty=2.0,  # give penalty on the token that was generated previously
        no_repeat_ngram_size=4,  # ngram of 4 will never repeat
        max_new_tokens=1024,  # max new tokens to generate
        eos_token_id=tokenizer.eos_token_id,  # set eos token
        do_sample=True,  # sampling tokens
        top_p=0.1,  # Tokens with top 10% probability will be chosen only => less diversity
        early_stopping=True,  # when 2 beams are found, end it there
        temperature=1.0  # if set less than 1.0, something goes wrong with probability calculation (quantization issue)
    )

    mapped_prompt = PROMPT_DICT['prompt_input'].format_map({'input': user_input})
    response = generator(mapped_prompt, **generation_args)
    result = (response[0]['generated_text']).replace(mapped_prompt, '')  # remove input and prompts
    return result


# configuration
IGNORE_INDEX = -100
PAD_TOKEN = "[PAD]"
EOS_TOKEN = "</s>"
BOS_TOKEN = "</s>"
UNK_TOKEN = "</s>"
# special tokens
CLUE_TOKEN = "<CLUE>"
REASONING_TOKEN = "<REASONING>"
LABEL_TOKEN = "<LABEL>"

# prompt for intention generation
PROMPT_DICT = {
    "prompt_input": (
        "이 모델은 '입력 문장'을 분석해서 의도를 분류하는 모델입니다. " +
        "분류할 의도로는 '메타 정보 검색', '키워드 검색', '유사도서 검색', '판매량 검색', 그리고 '그 외'가 있습니다. " +
        "입력 문장이 책 추천과 관련이 없는 경우, '그 외'로 분류합니다. " +
        "입력 문장이 책 추천과 관련이 있는 경우, 다음과 같이 분류합니다. " +
        "저자, 출판사, 기간 등 메타 정보가 포함된 경우 '메타 정보 검색'으로 분류합니다. " +
        "키워드 정보만 포함된 경우 '키워드 검색'으로 분류합니다. " +
        "제목 정보가 포함된 경우 예외 없이 '유사도서 검색'으로 분류합니다. " +
        "판매량 정보가 들어간 경우 예외 없이 '판매량 검색'으로 분류합니다. " +
        "우선, 입력 데이터를 받으면 주어진 입력 데이터에서 의도를 분류할 때 도움이 될 수 있는 단서들을 추출합니다. " +
        "그리고 각 단서마다 어떤 종류의 단서인지 추가합니다. " +
        "그 다음, 주어진 입력 데이터와 단서들을 바탕으로 어떤 의도인지 추론하는 글을 생성합니다. " +
        "그 후, 입력 데이터와 단서들과 추론한 글을 바탕으로 의도를 분류합니다. " +
        "반드시 '단서들', '추론', '의도' 순서대로 생성해야 합니다. " +
        "의도를 추론할 때 반드시 주어진 의도를 문자 그대로 생성해야 합니다.\n"
        "입력: {input}\n\n"
    )
}


MODEL_ID = "rycont/kakaobrain__kogpt-6b-8bit"  # Kakao KoGPT model 8-bit quantized

# get model and tokenizer
model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID, padding_side="right", model_max_length=1024
)
tokenizer.add_special_tokens(
    {
        "eos_token": EOS_TOKEN,
        "bos_token": BOS_TOKEN,
        "unk_token": UNK_TOKEN,
    }
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_tokens([CLUE_TOKEN, REASONING_TOKEN, LABEL_TOKEN], special_tokens=True)
model.resize_token_embeddings(len(tokenizer))

# get PeftModel
model = PeftModel.from_pretrained(model=model,
                                 model_id="./intention_models/lora_results/kakao_models/just_intention_title_brackets_n_100_lr_1e_5/checkpoint-1600/")
