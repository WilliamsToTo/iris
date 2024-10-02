import re
import numpy as np
import random
import torch
import hf_olmo
import requests
import json
from crossref.restful import Works
from tqdm.autonotebook import tqdm
import pandas as pd
from itertools import combinations
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

seed = 42
torch.manual_seed(seed)


def send_query_to_llm_chat(query, model, tokenizer, num_return_sequences):
    # feed query to model, and then generate response
    # if "gemma" in model.config.model_type or "llama" in model.config.model_type:
    #     query = tokenizer.apply_chat_template(messages, tokenize=False)
    print(model.device)
    inputs = tokenizer(query, return_tensors="pt").to(model.device)
    input_length = inputs.input_ids.shape[1]
    if hasattr(model.config, "max_position_embeddings"):
        max_length = model.config.max_position_embeddings
    elif hasattr(model.config, "max_sequence_length"):
        max_length = model.config.max_sequence_length
    else:
        max_length = 2048
    print("input length: ", input_length, max_length - input_length - 10)

    if model.config.model_type == "llama":
        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
    else:
        terminators = [
            tokenizer.eos_token_id,
        ]

    with torch.no_grad():
        generate_ids = model.generate(inputs.input_ids, attention_mask=inputs.attention_mask, do_sample=True, top_p=0.9,
                                      repetition_penalty=1.25, temperature=0.8, max_new_tokens=max_length - input_length - 10,
                                      eos_token_id=terminators, num_return_sequences=num_return_sequences)
        generate_response = tokenizer.batch_decode(generate_ids[:, inputs.input_ids.shape[-1]:], skip_special_tokens=True,
                                             clean_up_tokenization_spaces=False)

    return generate_response

demos_1 = [
    # {"role": "system", "content": "You are an expert in causal discovery, consistently and accurately identifying the correct causal relationships between two events.",
    # },
    {"role": "user", "content":
        "This task is to determine the cause-and-effect relationship between two events based on commonsense knowledge. "
        "We are interested in the causal relationship between 'it is raining' and 'carrying an umbrella'. "
        "Does 'it is raining' cause 'carrying an umbrella'? "
        "Let's provide a step-by-step explanation, then give your final answer using yes or no."},
    {"role": "assistant", "content":
        "Step-by-Step Explanation: "
        "1. Observation: The presence of rain often results in individuals using umbrellas. "
        "2. Rationale: People use umbrellas to protect themselves from getting wet during rain. "
        "3. Conclusion: Rain acts as a cause for the use of an umbrella. The final answer is Yes. "
        "Final Answer: Yes"},
    {"role": "user", "content":
        "This task is to determine the cause-and-effect relationship between two events based on commonsense knowledge. "
        "We are interested in the causal relationship between 'it is raining' and 'carrying an umbrella'. "
        "Does 'carrying an umbrella' causes 'it is raining'? "
        "Let's provide a step-by-step explanation, then give your final answer using yes or no. "},
    {"role": "assistant", "content":
        "Step-by-Step Explanation: "
        "1. Observation: The act of opening an umbrella does not influence weather patterns or conditions. "
        "2. Rationale: Rain is a meteorological event determined by atmospheric conditions. The use of an umbrella does not have the capability to influence weather conditions. "
        "3. Conclusion: There is no causal relationship where the umbrella causes rain. The answer is No. "
        "Final Answer: No "},
    {"role": "user", "content":
        "This task is to determine the cause-and-effect relationship between two events based on commonsense knowledge. "
        "We are interested in the causal relationship between 'Eating food' and 'feeling Hunger'. "
        "Does 'Eating food' causes 'feeling Hunger'? "
        "Let's provide a step-by-step explanation, then give your final answer using yes or no. "},
    {"role": "assistant", "content":
        "Step-by-Step Explanation: "
        "1. Observation: Eating involves consuming food, which typically reduces the feeling of hunger. "
        "2. Rationale: The act of eating is intended to satiate hunger, not induce it. "
        "3. Conclusion: Eating does not cause hunger. Instead, it alleviates it. The answer is No. "
        "Final Answer: No"},
    {"role": "user", "content":
        "This task is to determine the cause-and-effect relationship between two events based on commonsense knowledge. "
        "We are interested in the causal relationship between 'Eating food' and 'feeling Hunger'. "
        "Does 'Feeling hunger' causes 'Eating food'? "
        "Let's provide a step-by-step explanation, then give your final answer using yes or no."},
    {"role": "assistant", "content":
        "Step-by-Step Explanation: "
        "1. Observation: The sensation of hunger typically motivates individuals to eat. "
        "2. Rationale: Hunger is a biological signal that the body needs nutrients. This sensation prompts the behavior of eating to replenish energy and nutrients. "
        "3. Conclusion: It is reasonable to conclude that hunger causes eating. The answer is Yes. "
        "Final Answer: Yes"},
    {"role": "user", "content":
        "This task is to determine the cause-and-effect relationship between two events based on commonsense knowledge. "
        "We are interested in the causal relationship between 'the number of ice cream sales' and 'the number of drowning incident'. "
        "Does 'the number of ice cream sales' causes 'the number of drowning incident'? "
        "Let's provide a step-by-step explanation, then give your final answer using yes or no."},
    {"role": "assistant", "content":
        "Step-by-Step Explanation: "
        "1. Observation: Increases in ice cream sales and drowning incidents may occur simultaneously, especially during warmer months. "
        "2. Rationale: While both ice cream sales and drowning incidents might rise during the summer, this does not imply that one causes the other. Instead, both are likely correlated with a third factor, such as higher temperatures or increased outdoor activities during warm weather. "
        "3. Conclusion: The simultaneous increase in ice cream sales and drowning incidents is better explained by a common cause (warm weather) rather than a direct causal relationship between the two. The answer is No. "
        "Final Answer: No"},
    {"role": "user", "content":
        "This task is to determine the cause-and-effect relationship between two events based on commonsense knowledge. "
        "We are interested in the causal relationship between 'the number of ice cream sales' and 'the number of drowning incident'. "
        "Does 'the number of drowning incident' causes 'the number of ice cream sales'? "
        "Let's provide a step-by-step explanation, then give your final answer using yes or no."},
    {"role": "assistant", "content":
        "Step-by-Step Explanation: "
        "1. Observation: Drowning incidents do not have a direct impact on the demand or sales of ice cream. "
        "2. Rationale: The occurrence of drowning incidents is an unfortunate event that does not influence people's consumption behavior regarding ice cream. Similar to the previous explanation, any observed correlation is more likely due to external factors like seasonality rather than a direct causal link. "
        "3. Conclusion: There is no logical or direct pathway through which drowning incidents could cause an increase in ice cream sales. Any correlation observed is likely due to external, confounding variables. The answer is No. "
        "Final Answer: No"},
]

def create_prompt_general(query, demos, model, tokenizer):
    single_prompt = {"role": "user", "content": query}
    prompt = demos + [single_prompt]

    chat_template_models = ["mistral", "gemma", "llama", "olmo", "qwen2"]
    if model.config.model_type in chat_template_models:
        input_str = tokenizer.apply_chat_template(prompt, add_generation_prompt=True, tokenize=False)
    elif model.config.model_type == "falcon":
        chat_template = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '>>QUESTION<<\n' + message['content'] }}\n{% elif message['role'] == 'assistant' %}\n{{ '>>ANSWER<<\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '>>ANSWER<<' }}\n{% endif %}\n{% endfor %}"
        input_str = tokenizer.apply_chat_template(prompt, chat_template=chat_template, add_generation_prompt=True, tokenize=False)
    elif model.config.model_type == "bloom":
        chat_template = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ 'Question: \n' + message['content'] }}\n{% elif message['role'] == 'assistant' %}\n{{ 'Answer: \n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ 'Answer: ' }}\n{% endif %}\n{% endfor %}"
        input_str = tokenizer.apply_chat_template(prompt, chat_template=chat_template, add_generation_prompt=True, tokenize=False)
    elif model.config.model_type == "gpt2":
        tokenizer.chat_template = "{{ eos_token }}{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"
        input_str = tokenizer.apply_chat_template(prompt, add_generation_prompt=True, tokenize=False)
    else:
        raise ValueError("Model type not supported")
    #input_str = input_str.replace("\t", "")
    return input_str

def create_prompt_causal_question(cause, effect, model, tokenizer, shot_num):
    demos = demos_1
    prompt = {"role": "user", "content": f"This task is to determine the cause-and-effect relationship between two events based on commonsense knowledge. "
                                         f"We are interested in the causal relationship between '{cause}' and '{effect}'. "
                                         f"Does '{cause}' cause '{effect}'?"
                                         f"Let's provide a step-by-step explanation, then give your final answer using yes or no."
              }
    #demos.append(prompt)
    chat_template_models = ["mistral", "gemma", "llama", "olmo", "qwen2"]
    if model.config.model_type in chat_template_models:
        input_str = tokenizer.apply_chat_template(demos[:shot_num*2]+[prompt], add_generation_prompt=True, tokenize=False)
    elif model.config.model_type == "falcon":
        #chat_template = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '>>QUESTION<<\n' + message['content'] }}\n{% elif message['role'] == 'assistant' %}\n{{ '>>ANSWER<<\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '>>ANSWER<<' }}\n{% endif %}\n{% endfor %}",
        chat_template = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '>>QUESTION<<\n' + message['content'] }}\n{% elif message['role'] == 'assistant' %}\n{{ '>>ANSWER<<\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '>>ANSWER<<' }}\n{% endif %}\n{% endfor %}"
        input_str = tokenizer.apply_chat_template(demos[:shot_num * 2] + [prompt], chat_template=chat_template, add_generation_prompt=True, tokenize=False)
    elif model.config.model_type == "bloom":
        chat_template = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ 'Question: \n' + message['content'] }}\n{% elif message['role'] == 'assistant' %}\n{{ 'Answer: \n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ 'Answer: ' }}\n{% endif %}\n{% endfor %}"
        input_str = tokenizer.apply_chat_template(demos[:shot_num * 2] + [prompt], chat_template=chat_template,
                                                  add_generation_prompt=True, tokenize=False)
    elif model.config.model_type == "gpt2":
        tokenizer.chat_template = "{{ eos_token }}{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"
        input_str = tokenizer.apply_chat_template(demos[:shot_num * 2] + [prompt], add_generation_prompt=True, tokenize=False)
    else:
        raise ValueError("Model type not supported")
    #input_str = input_str.replace("\t", "")
    return input_str

def extract_answer_chat_yes_no(answer):
    answer = answer.replace('*', '')
    words = re.findall(r'\w+', answer.lower())

    yes_exist = False
    no_exist = False
    for word in words:
        if word == "yes":
            yes_exist = True
        elif word == "no":
            no_exist = True
    #falcon extraction
    # for word in words:
    #     if word == "yes":
    #         yes_exist = True
    #         break
    #     elif word == "no":
    #         no_exist = True
    #         break
    if yes_exist:
        extracted = 'yes'
    elif no_exist:
        extracted = 'no'
    else:
        extracted = 'fail'

    return extracted

def selfCheck_causalRelationVerify(cause, effect, model, tokenizer, num_return_sequences, shot_num):
    # create prompt
    query = create_prompt_causal_question(cause, effect, model, tokenizer, shot_num)
    responses = send_query_to_llm_chat(query, model, tokenizer, num_return_sequences=num_return_sequences)
    selfCheckScoreMap = {'yes': 0.0, 'no': 1.0, 'fail': 0.5}
    responses_agree = []
    for r1 in tqdm(responses):
        responses_agree.append([])
        for r2 in responses:
            if r1 != r2:
                selfCheckPrompt = f"Given the question: Does '{cause}' causes '{effect}'? We have two responses.\n" \
                                    f"Response 1: {r1}\n" \
                                    f"Response 2: {r2}\n" \
                                    f"Does Response 2 agree with Response 1? Let's provide a step-by-step explanation, then give your final answer using 'yes' or 'no'."

                llm_query = create_prompt_general(selfCheckPrompt, [], model, tokenizer)
                selfCheckResponse = send_query_to_llm_chat(llm_query, model, tokenizer, num_return_sequences=1)
                extract_answer = extract_answer_chat_yes_no(selfCheckResponse[0])
                responses_agree[-1].append(selfCheckScoreMap[extract_answer])
            else:
                # response always agree itself
                responses_agree[-1].append(selfCheckScoreMap['yes'])
    responses_agreeScores = []
    for agreements in responses_agree:
        responses_agreeScores.append(np.mean(agreements))
    max_value = max(responses_agreeScores)
    max_index = responses_agreeScores.index(max_value)
    checked_response = responses[max_index]
    final_answer = extract_answer_chat_yes_no(checked_response)
    print(final_answer, max_value)
    return final_answer

def selfCheck_majorVote(cause, effect, model, tokenizer, num_return_sequences, shot_num, threshold=0.5):
    # create prompt
    query = create_prompt_causal_question(cause, effect, model, tokenizer, shot_num)
    responses = send_query_to_llm_chat(query, model, tokenizer, num_return_sequences=num_return_sequences)
    print(responses)
    extracted_answers = []
    for response in responses:
        extracted_answers.append(extract_answer_chat_yes_no(response))

    yes_confidence = extracted_answers.count('yes') / len(extracted_answers)
    no_confidence = extracted_answers.count('no') / len(extracted_answers)
    fail_confidence = extracted_answers.count('fail') / len(extracted_answers)

    if yes_confidence > threshold:
        return 'yes'
    elif no_confidence > threshold:
        return 'no'
    else:
        return 'fail'

def build_payload(query, start=1, num=10):
    API_KEY = "AIzaSyD-MWKiYNEeGT_l66OdJjL7SfpP8B08Lqo"
    SEARCH_ENGINE_ID_ENTIREWEB = "43d71a651b6434fac"
    SEARCH_ENGINE_ID_CONSTRAINT = "52a5c04c8da3b4620"
    payload = {
        'key': API_KEY,
        'q': query,
        'cx': SEARCH_ENGINE_ID_ENTIREWEB,
        'start': start,
        'num': num,
        'lr': 'lang_en',
        'hl': 'en'
    }
    return payload

def make_request(payload):
    response = requests.get('https://www.googleapis.com/customsearch/v1', params=payload)
    if response.status_code != 200:
        print(f"Request failed with status code: {response.status_code}")
        print(f"Response text: {response.text}")
        print(f"Response headers: {response.headers}")
        raise Exception(f"Request failed with status code: {response.status_code}")
    return response.json()

def googleRetrieve(causes, effects):
    all_retrieved_results = []
    for cause in causes:
        for effect in effects:
            cause = cause.lower()
            effect = effect.lower()
            # query = f"\"{cause} causes {effect}\""
            query = f"\"{cause}\" AND \"{effect}\""
            print(query)
            for start_idx in range(1, 10, 10): # (1, 20, 10)
                payload = build_payload(query, start=start_idx, num=10)
                response = make_request(payload)
                if "items" in response:
                    all_retrieved_results += response["items"]
            if len(all_retrieved_results) > 20:
                return all_retrieved_results
    return all_retrieved_results

def remove_jats_tags(s):
    # Remove all JATS tags
    cleaned_string = re.sub(r'<\/?jats:[^>]+>', '', s)
    return cleaned_string

def crossrefapiRetreve(causes, effects):
    works = Works()
    abstracts = []
    for cause in causes:
        for effect in effects:
            cause = cause.lower()
            effect = effect.lower()

            w1 = works.query(f'{cause} {effect}').order('desc')
            print(w1.url)
            count = 0
            max_search = 2000
            for item in tqdm(w1, total=max_search):
                if 'abstract' in item:
                    abstract = remove_jats_tags(item['abstract'])
                    if cause in abstract and effect in abstract:
                        abstracts.append(abstract)

                if count > max_search:
                    break
                if len(abstracts) > 20:
                    print(f'{causes} {effects}, count: ', len(abstracts))
                    return abstracts

                count += 1

    print(f'{causes} {effects}, count: ', len(abstracts))
    return abstracts


def retrieveLLMVerfy(cause_name, effect_name, cause_desc, effect_desc, claim2docs, model, tokenizer, num_return_sequences, threshold=0.5):
    verify_demos = [
        {"role": "user", "content":
            "Please evaluate the following claim based on the provided document:\n"
            "Claim: 'it is raining' causes 'carrying an umbrella'.\n "
            "Document: Rain, as a natural weather condition, often compels individuals to use umbrellas to protect themselves from getting wet.\n"
            "Does the document verify the claim? Please provide a step-by-step justification then respond with 'Yes' or 'No'."},
        {"role": "assistant", "content":
            "Step-by-Step Explanation: "
            "1. Understand the Claim: The claim implies a causal relationship where the occurrence of rain (cause) leads to people carrying umbrellas (effect). "
            "2. Analyze the Document Content: The document directly addresses the behavior of individuals using umbrellas as a response to rain. "
            "3. Conclusion on Verification: The document explicitly states that rain leads to people using umbrellas to avoid getting wet. This directly supports the causal claim. "
            "The final Answer is Yes, the document verifies the claim."},
        {"role": "user", "content":
            "Please evaluate the following claim based on the provided document:\n"
            "Claim: 'carrying an umbrella' causes 'it is raining'.\n"
            "Document: People carry umbrellas because they expect rain. An equilibrium with umbrellas must include a significant possibility of rain, but we don’t say that carrying umbrellas must lead to rain. If we take away people’s umbrellas, it will not prevent rain, and if we require people to carry umbrellas, it will not cause rain.\n"
            "Does the document verify the claim? Please provide a step-by-step justification then respond with 'Yes' or 'No'."},
        {"role": "assistant", "content":
            "Step-by-Step Explanation: "
            "1. Understand the Claim: This claim implies that the presence of umbrellas has a causal impact on the occurrence of rain. "
            "2. Analyze the Document Content: ``People carry umbrellas because they expect rain'' indicating that the reason for carrying umbrellas is the anticipation of rain, not that carrying them causes rain. ``We don’t say that carrying umbrellas must 'lead to' rain '' directly stating that there is no causal relationship from umbrellas to rain. ``If we take away people’s umbrellas, it will not prevent rain, and if we require people to carry umbrellas, it will not cause rain'' further emphasizing the lack of a causal effect from carrying umbrellas on the occurrence of rain."
            "3. Conclusion on Verification: The document clearly states that carrying an umbrella is a response to the expectation of rain, not a cause of it. It specifically negates any causal relationship from carrying umbrellas to causing rain, as evidenced by the explanations given about expectations, causality, and hypothetical scenarios."
            "The final Answer is No, the document does not verify the claim. On the contrary, it explicitly refutes it."}
    ]

    snippets = claim2docs[f"{cause_name} causes {effect_name}"]
    # add in-context learning
    for snippet in snippets:
        verificationPrompt = f"Please evaluate the following claim based on the provided document:\n" \
                    f"Claim: '{cause_name}' causes '{effect_name}'.\n" \
                    f"Document: {snippet}.\n" \
                    f"Does the document verify the claim? Please provide a step-by-step justification then respond with 'Yes' or 'No'."

        llm_query = create_prompt_general(verificationPrompt, demos=verify_demos, model=model, tokenizer=tokenizer)
        responses = send_query_to_llm_chat(llm_query, model, tokenizer, num_return_sequences=num_return_sequences)
        print(llm_query, responses)
        extracted_answers = []
        for response in responses:
            extracted_answers.append(extract_answer_chat_yes_no(response))

        print(extracted_answers)
        yes_confidence = extracted_answers.count('yes') / len(extracted_answers)

        if yes_confidence > threshold:
            return 'yes'

    return 'no'


# device = 'cpu'
# if torch.cuda.is_available():
#     device = 'cuda'
# quantization_config = BitsAndBytesConfig(load_in_8bit=True)
# openllm_path = "/data/ccu/taof/olmo/olmo_7B_instruct_hf/"
# tokenizer = AutoTokenizer.from_pretrained(openllm_path)
# model = AutoModelForCausalLM.from_pretrained(openllm_path, quantization_config=quantization_config,
#                                              torch_dtype=torch.bfloat16, device_map="auto", )
# model.eval()
#
# selfCheck_causalRelationVerify(cause="smoker", effect="cancer", model=model, tokenizer=tokenizer,
#                                num_return_sequences=10, shot_num=3)






