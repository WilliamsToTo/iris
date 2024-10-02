import re
import numpy as np
import random
import torch
import hf_olmo
from tqdm.autonotebook import tqdm
import pandas as pd
from .utils import PreviousEdges, adjacency_matrix_to_edge_list
from itertools import combinations
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

seed = 42
torch.manual_seed(seed)


def send_query_to_llm_chat(query, model, tokenizer, num_return_sequences):
    # feed query to model, and then generate response
    # if "gemma" in model.config.model_type or "llama" in model.config.model_type:
    #     query = tokenizer.apply_chat_template(messages, tokenize=False)
    inputs = tokenizer(query, return_tensors="pt").to(model.device)
    input_length = inputs.input_ids.shape[1]
    if hasattr(model.config, "max_position_embeddings"):
        max_length = model.config.max_position_embeddings
    elif hasattr(model.config, "max_sequence_length"):
        max_length = model.config.max_sequence_length
    else:
        max_length = 2048
    print("input length: ", input_length, max_length - input_length - 10)
    with torch.no_grad():
        generate_ids = model.generate(inputs.input_ids, attention_mask=inputs.attention_mask, do_sample=True, top_p=0.9,
                                      repetition_penalty=1.25, temperature=0.8,
                                      max_new_tokens=max_length - input_length - 10,
                                      eos_token_id=tokenizer.eos_token_id,
                                      num_return_sequences=num_return_sequences)
        generate_response = tokenizer.batch_decode(generate_ids[:, inputs.input_ids.shape[-1]:], skip_special_tokens=True,
                                             clean_up_tokenization_spaces=False)

    return generate_response

def send_query_to_llm_complete(query, model, tokenizer, num_return_sequences=1):
    # feed query to model, and then generate response
    inputs = tokenizer(query, return_tensors="pt").to(model.device)
    input_length = inputs.input_ids.shape[1]
    if hasattr(model.config, "max_position_embeddings"):
        max_length = model.config.max_position_embeddings
    elif hasattr(model.config, "max_sequence_length"):
        max_length = model.config.max_sequence_length
    else:
        max_length = 2048
    print("input length: ", input_length, max_length - input_length - 10)
    with torch.no_grad():
        generate_ids = model.generate(inputs.input_ids, attention_mask=inputs.attention_mask, do_sample=True, top_p=0.9,
                                      repetition_penalty=1.25, temperature=0.8, max_new_tokens=256,
                                      eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.eos_token_id,
                                      num_return_sequences=num_return_sequences)
        generate_response = tokenizer.batch_decode(generate_ids[:, inputs.input_ids.shape[-1]:], skip_special_tokens=True,
                                             clean_up_tokenization_spaces=False)

    return generate_response

def send_query_to_llm_prob(query, model, tokenizer):
    # feed query to model, and then generate response
    inputs = tokenizer(query, return_tensors="pt").to(model.device)
    input_length = inputs.input_ids.shape[1]
    print("input length: ", input_length)
    with torch.no_grad():
        inputs.pop('token_type_ids', None)
        model_outpus = model(**inputs)
        logits = model_outpus.logits
    a_id = tokenizer.get_vocab()["A"]
    b_id = tokenizer.get_vocab()["B"]
    c_id = tokenizer.get_vocab()["C"]
    if logits[0][-1][a_id] > logits[0][-1][b_id] and logits[0][-1][a_id] > logits[0][-1][c_id]:
        return "A"
    elif logits[0][-1][b_id] > logits[0][-1][a_id] and logits[0][-1][b_id] > logits[0][-1][c_id]:
        return  "B"
    else:
        return "C"

def extract_answer_chat(answer):
    answer = answer.replace('*', '')
    mappping = {
        'a': 'A',
        'b': 'B',
        'c': 'C'
    }
    match1 = re.search(r'<answer>(.*?)</answer>', answer.lower())
    match2 = re.search(r'answer:\s*([a-z])', answer.lower())
    match3 = re.search(r'answer is\s*([a-z])', answer.lower())

    # Extract the content
    result = ''
    if match1:
        result = match1.group(1)
    elif match2:
        result = match2.group(1)
    elif match3:
        result = match3.group(1)
    if result not in ['a', 'b', 'c']:
        result = 'c'

    return mappping[result]

def extract_answer_chat_yes_no(answer):
    answer = answer.replace('*', '')

    # match1 = re.search(r"answer is (yes|no)", answer.lower())
    # match2 = re.search(r'answer:\s*(yes|no)', answer.lower())
    # match3 = re.search(r'answer is:\s*(yes|no)', answer.lower())
    # match4 = re.search(r"answer.*\sis\s(yes|no)\.", answer.lower())

    # Extract the content
    # result = 'no'
    # if match1:
    #     result = match1.group(1)
    # elif match2:
    #     result = match2.group(1)
    # elif match3:
    #     result = match3.group(1)
    # elif match4:
    #     result = match4.group(1)
    words = re.findall(r'\w+', answer.lower())
    yes_exist = False
    no_exist = False
    for word in words:
        if word == "yes":
            yes_exist = True
        elif word == "no":
            no_exist = True
    if yes_exist and no_exist:
        return 'no'
    elif yes_exist:
        return 'yes'
    elif no_exist:
        return 'no'
    else:
        return 'no'


def create_samples_description(var_names_and_desc, df):
    query = "Below is a description of some samples:\n"
    descriptions = []
    for idx, (index, row) in enumerate(df.iterrows()):
        description = f"In sample {idx}, " + ", ".join([f"{var_names_and_desc[col].name} is {row[col]}" for col in df.columns]) + ".\n"
        descriptions.append(description)
    return query + "".join(descriptions)


def select_min_unique_subset(df):
    subset_indices = set()

    for column in df.columns:
        # Drop duplicates prioritizing the first occurrence, keeping unique values in each iteration
        unique_vals = df.drop_duplicates(subset=[column], keep='first')

        # Update the subset indices with the indices of rows containing unique values for the current column
        subset_indices.update(unique_vals.index)

        # Optional: Break early if all rows are selected
        if len(subset_indices) == len(df):
            break

    # Create a subset DataFrame using the selected indices
    subset_df = df.loc[sorted(subset_indices)]

    return subset_df

def openllm_pairwise_chat(var_names_and_desc, prompts, df, include_samples=False, openllm_path=None):
    # load open llm model
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    tokenizer = AutoTokenizer.from_pretrained(openllm_path)
    model = AutoModelForCausalLM.from_pretrained(openllm_path, quantization_config=quantization_config, torch_dtype=torch.float16, device_map="auto",)

    # list all edges
    edges = list(combinations(df.columns, 2))
    previous_edges = PreviousEdges()
    cause2effect = []
    predict_options = []
    for e in tqdm(edges, desc="Edges"):
        if random.random() < 0.5:
            head, tail = e
        else:
            tail, head = e
        head = var_names_and_desc[head]
        tail = var_names_and_desc[tail]

        messages = [
        {"role": "user", "content": '''
            This task is to determine the cause-and-effect relationship between two variables in a causal graph. 
            Below is an example of this task.
            Here is a description of the causal variables in this causal graph:
            rain: whether it is raining
            umbrella: whether a person is carrying an umbrella
            We are interested in the causal relationship between "rain" and "umbrella".
            Which cause-and-effect relationship is more likely?
            A. "rain" causes "umbrella".
            B. "umbrella" causes "rain".
            C. There is no causal relationship between "rain" and "umbrella".
            Let's provide a step-by-step explanation, then provide your final answer within the tags <answer>A/B/C</answer>.'''},
        {"role": "assistant", "content": '''
            Step-by-Step Explanation:
            
            1. Observation of Natural Phenomena: It is a common observation that people carry umbrellas when it is raining or when they anticipate rain. This behavior is a preparatory action to protect oneself from getting wet.
            
            2. Causal Directionality: The occurrence of rain is a natural phenomenon that is not influenced by human actions such as carrying an umbrella. Therefore, it is logical to infer that the presence of rain can influence a person's decision to carry an umbrella.
            
            3. Counterfactual Reasoning: If "umbrella" caused "rain", we would expect rain to occur as a result of someone deciding to carry an umbrella. However, this is against our understanding of meteorological processes, where rain is determined by atmospheric conditions, not human actions.
            
            4. Elimination of Alternative: Option C suggests there is no causal relationship between "rain" and "umbrella". However, the predictive behavior of carrying an umbrella based on weather conditions suggests a direct causal link, where the anticipation or occurrence of rain leads to the action of carrying an umbrella for protection.
            
            Therefore, based on the logic that natural phenomena like rain can influence human behavior to carry items for protection (umbrellas), but not vice versa, the most likely cause-and-effect relationship is that "rain" causes "umbrella".
            
            The final answer is: <answer>A</answer>'''},
        {"role": "user", "content": '''Below is another example of this task.
            Here is a description of the causal variables in this causal graph:
            Eating: whether or not a person consumes food.
            Hunger: whether or not a person feels hunger.
            We are interested in the causal relationship between "Eating" and "Hunger".
            Which cause-and-effect relationship is more likely?
            A. "Eating" causes "Hunger".
            B. "Hunger" causes "Eating".
            C. There is no causal relationship between "Eating" and "Hunger".
            Let's provide a step-by-step explanation, then provide your final answer within the tags <answer>A/B/C</answer>.'''},
        {"role": "assistant", "content": '''
            Step-by-Step Explanation:
            
            1. Biological Signals: Hunger is a physiological signal indicating the body's need for nutrients. 
            
            2. Response to Hunger: The natural response to feeling hunger is to consume food. 
            
            3. Causal Direction: The sequence of these events indicates that hunger (the physiological need) precedes eating (the action to satisfy the need). Therefore, the causal direction flows from the sensation of hunger to the act of eating.
            
            4. Counterfactual Consideration: The statement "Eating causes Hunger" (Option A) contradicts typical biological responses, as eating is generally understood to satiate hunger rather than induce it. 
            
            5. Existence of Causal Relationship: While Option C suggests the absence of a causal relationship between eating and hunger, everyday experiences and biological knowledge affirm that hunger directly influences the decision to eat.
            The final answer is: <answer>B</answer>
         '''},
        {"role": "user", "content": '''Below is another example of this task.
            Here is a description of the causal variables in this causal graph:
            Ice cream sale: the number of ice cream sales.
            Drowning incident: the number of drowning incidents.
            We are interested in the causal relationship between "Ice cream sale" and "Drowning incident".
            Which cause-and-effect relationship is more likely?
            A. "Ice cream sale" causes "Drowning incident".
            B. "Drowning incident" causes "Ice cream sale".
            C. There is no causal relationship between "Ice cream sale" and "Drowning incident".
            Let's provide a step-by-step explanation, then provide your final answer within the tags <answer>A/B/C</answer>.'''},
        {"role": "assistant", "content": '''
        Step-by-Step Explanation:
    
            1. Common Cause: Both ice cream sales and drowning incidents tend to increase during the warmer months. This correlation is due to the increase in temperature, which encourages outdoor activities. People are more likely to buy ice cream and engage in activities near water bodies during this time.
            
            2. Lack of Direct Influence: The act of purchasing ice cream does not directly influence the occurrence of drowning incidents, nor do drowning incidents impact the sale of ice cream. 
                    
            3. Analysis of Causality: To establish a cause-and-effect relationship, one would expect that changes in one variable directly result in changes in the other. However, altering the number of ice cream sales independently of the season (e.g., through promotions or price changes) would not logically affect drowning incident rates, and vice versa.
            
            4. Conclusion: Given above analysis, there is no direct causal relationship between them.
            The final answer is: <answer>C</answer>'''},
        ]

        demonstrate = ""
        for msg in messages:
            demonstrate += msg["content"] + "\n"

        query = f'''{prompts["user"]} 
                Here is a description of the causal variables in this causal graph:
                '''
        for var in var_names_and_desc:
            causal_var = var_names_and_desc[var]
            query += f'''{causal_var.name}: {causal_var.description}\n'''

        if include_samples:
            min_df = select_min_unique_subset(df)
            query += create_samples_description(var_names_and_desc, min_df)

        query += f'''
        Here are the causal relationships you know so far:
        {previous_edges.get_previous_relevant_edges_string(head.name, tail.name)}
        We are interested in the causal relationship between "{head.name}" and "{tail.name}".
        '''

        query += f'''
        Which cause-and-effect relationship is more likely? 
        A. "{head.name}" causes "{tail.name}". 
        B. "{tail.name}" causes "{head.name}". 
        C. There is no causal relationship between "{head.name}" and "{tail.name}".
        Let's provide a step-by-step explanation, then provide your final answer within the tags <answer>A/B/C</answer>.'''

        messages.append({"role": "user", "content": query})
        query = demonstrate + query
        chat_template_models = ["mistral", "gemma", "llama", "falcon"] #"falcon"
        if model.config.model_type in chat_template_models:
            query = tokenizer.apply_chat_template(messages, tokenize=False)

        print("query: ", query)
        need_longer_response = True
        try_count = 0
        while need_longer_response and try_count < 20:
            response = send_query_to_llm_chat(query, model, tokenizer, num_return_sequences=1)[0].strip()
            try_count += 1
            if len(response) > 10:
                need_longer_response = False
        print("response: ", response)
        choice = extract_answer_chat(response)
        predict_options.append(choice)
        print(choice)

        previous_edges.add_edge_pair_wise(head.name, tail.name, choice)
        if choice == 'A':
            cause2effect.append((head.name, tail.name))
        elif choice == 'B':
            cause2effect.append((tail.name, head.name))
    adj_matrix = previous_edges.get_adjacency_matrix([var_names_and_desc[var].name for var in df.columns])

    result_dict = {}
    for head, tail in cause2effect:
        if head in result_dict:
            # Append the tail to the existing list for this head
            result_dict[head].append(tail)
        else:
            # If the head is not a key, add it with the tail in a new list
            result_dict[head] = [tail]
    result_dict["predict_options"] = predict_options
    result_dict["adj_matrix"] = adj_matrix.tolist()
    print(result_dict)

    return adj_matrix, result_dict


def openllm_pairwise_chat_yes_no(var_names_and_desc, prompts, df, include_samples=False, openllm_path=None):
    # load open llm model
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    tokenizer = AutoTokenizer.from_pretrained(openllm_path)
    model = AutoModelForCausalLM.from_pretrained(openllm_path, quantization_config=quantization_config,
                                                 torch_dtype=torch.float16, device_map="auto", )

    # list all edges
    edges = list(combinations(df.columns, 2))
    previous_edges = PreviousEdges()
    cause2effect = []
    predict_options = []
    for e in tqdm(edges, desc="Edges"):
        if random.random() < 0.5:
            head, tail = e
        else:
            tail, head = e
        head = var_names_and_desc[head]
        tail = var_names_and_desc[tail]

        messages = [
            {"role": "user", "content": '''
            This task is to determine the cause-and-effect relationship between two variables in a causal graph. 
            Below is an example of this task.
            Here is a description of the causal variables in this causal graph:
            rain: whether it is raining
            umbrella: whether a person is carrying an umbrella
            We are interested in the causal relationship between "rain" and "umbrella". Please answer the following question using 'yes' or 'no'.
            Does "rain" cause "umbrella"?
            Let's provide a step-by-step explanation, then give your final answer within format: Final Answer: yes or no.'''},
            {"role": "assistant", "content": '''
            Step-by-Step Explanation:
            1. Causality Principle: In a causal relationship, the occurrence of one event leads to the occurrence of another.
            2. Observation: The presence of rain often results in individuals using umbrellas.
            3. Rationale: People use umbrellas to protect themselves from getting wet during rain. 
            4. Conclusion: Rain acts as a cause for the use of an umbrella. The final answer is Yes.
            Final Answer: Yes'''},
            {"role": "user", "content": '''
            This task is to determine the cause-and-effect relationship between two variables in a causal graph. 
            Below is an example of this task.
            Here is a description of the causal variables in this causal graph:
            rain: whether it is raining
            umbrella: whether a person is carrying an umbrella
            We are interested in the causal relationship between "rain" and "umbrella". Please answer the following question using 'yes' or 'no'.
            Does "umbrella" causes "rain"?
            Let's provide a step-by-step explanation, then give your final answer within format: Final Answer: yes or no.'''},
            {"role": "assistant", "content": '''
            Step-by-Step Explanation:
            1. Causality Principle: A cause leads to an effect, not the reverse.
            2. Observation: The act of opening an umbrella does not influence weather patterns or conditions.
            3. Rationale: Rain is a meteorological event determined by atmospheric conditions. The use of an umbrella does not have the capability to influence weather conditions.
            4. Conclusion: There is no causal relationship where the umbrella causes rain. The answer is No.
            Final Answer: No'''},
            {"role": "user", "content": '''Below is another example of this task.
            Here is a description of the causal variables in this causal graph:
            Eating: whether or not a person consumes food.
            Hunger: whether or not a person feels hunger.
            We are interested in the causal relationship between "Eating" and "Hunger". Please answer the following question using 'yes' or 'no'.
            Does "Eating" causes "Hunger"?
            Let's provide a step-by-step explanation, then give your final answer within format: Final Answer: yes or no.'''},
            {"role": "assistant", "content": '''
            Step-by-Step Explanation:
            1. Causality Principle: A cause leads to an effect, typically altering the state of the effect in a predictable way.
            2. Observation: Eating involves consuming food, which typically reduces the feeling of hunger.
            3. Rationale: The act of eating is intended to satiate hunger, not induce it. 
            4. Conclusion: Eating does not cause hunger. Instead, it alleviates it. The answer is No.
            Final Answer: No'''},
            {"role": "user", "content": '''Below is another example of this task.
            Here is a description of the causal variables in this causal graph:
            Eating: whether or not a person consumes food.
            Hunger: whether or not a person feels hunger.
            We are interested in the causal relationship between "Eating" and "Hunger". Please answer the following question using 'yes' or 'no'.
            Does "Hunger" causes "Eating"?
            Let's provide a step-by-step explanation, then give your final answer within format: Final Answer: yes or no.'''},
            {"role": "assistant", "content": '''
            Step-by-Step Explanation:
            1. Causality Principle: An event or condition (cause) leads to an outcome (effect).
            2. Observation: The sensation of hunger typically motivates individuals to eat.
            3. Rationale: Hunger is a biological signal that the body needs nutrients. This sensation prompts the behavior of eating to replenish energy and nutrients.
            4. Conclusion: It is reasonable to conclude that hunger causes eating. The answer is Yes.
            Final Answer: Yes'''},
            {"role": "user", "content": '''Below is another example of this task.
            Here is a description of the causal variables in this causal graph:
            Ice cream sale: the number of ice cream sales.
            Drowning incident: the number of drowning incidents.
            We are interested in the causal relationship between "Ice cream sale" and "Drowning incident". Please answer the following question using 'yes' or 'no'.
            Does "Ice cream sale" causes "Drowning incident"?
            Let's provide a step-by-step explanation, then give your final answer within format: Final Answer: yes or no.'''},
            {"role": "assistant", "content": '''
            Step-by-Step Explanation:
            1. Causality Principle: For one event to cause another, there must be a direct relationship where changes in the first lead to changes in the second.
            2. Observation: Increases in ice cream sales and drowning incidents may occur simultaneously, especially during warmer months.
            3. Rationale: While both ice cream sales and drowning incidents might rise during the summer, this does not imply that one causes the other. Instead, both are likely correlated with a third factor, such as higher temperatures or increased outdoor activities during warm weather.
            4. Conclusion: The simultaneous increase in ice cream sales and drowning incidents is better explained by a common cause (warm weather) rather than a direct causal relationship between the two. The answer is No.
            Final Answer: No'''},
            {"role": "user", "content": '''Below is another example of this task.
            Here is a description of the causal variables in this causal graph:
            Ice cream sale: the number of ice cream sales.
            Drowning incident: the number of drowning incidents.
            We are interested in the causal relationship between "Ice cream sale" and "Drowning incident". Please answer the following question using 'yes' or 'no'.
            Does "Drowning incident" causes "Ice cream sale"?
            Let's provide a step-by-step explanation, then give your final answer within format: Final Answer: yes or no.'''},
            {"role": "assistant", "content": '''
            Step-by-Step Explanation:
            1. Causality Principle: A causal relationship implies that the occurrence or change in one variable directly affects another.
            2. Observation: Drowning incidents do not have a direct impact on the demand or sales of ice cream.
            3. Rationale: The occurrence of drowning incidents is an unfortunate event that does not influence people's consumption behavior regarding ice cream. Similar to the previous explanation, any observed correlation is more likely due to external factors like seasonality rather than a direct causal link.
            4. Conclusion: There is no logical or direct pathway through which drowning incidents could cause an increase in ice cream sales. Any correlation observed is likely due to external, confounding variables. The answer is No.
            Final Answer: No'''},
        ]

        # demonstrate = ""
        # for msg in messages:
        #     demonstrate += msg["content"] + "\n"

        instruct = f'''
                Here is a description of the causal variables in this causal graph:
                {head.name}: {head.description}
                {tail.name}: {tail.description}
                '''

        if include_samples:
            min_df = select_min_unique_subset(df)
            instruct += create_samples_description(var_names_and_desc, min_df)

        instruct += f'''
        We are interested in the causal relationship between "{head.name}" and "{tail.name}". Please answer the following question using 'yes' or 'no'.
        '''

        question_a = f'''Does "{head.name}" causes "{tail.name}"?'''
        question_b = f'''Does "{tail.name}" causes "{head.name}"?'''
        questions = {"A": question_a, "B": question_b}
        for key, question in questions.items():
            query = instruct + \
                    f'''{question}
                    Let's provide a step-by-step explanation, then give your final answer within format: Final Answer: yes or no.'''

            messages.append({"role": "user", "content": query})

            chat_template_models = ["mistral", "gemma", "llama"]  # "falcon"
            if model.config.model_type in chat_template_models:
                input_str = tokenizer.apply_chat_template(messages, tokenize=False)
            elif model.config.model_type == "falcon" or model.config.model_type == "olmo":
                input_str = tokenizer.apply_chat_template(messages[6:], tokenize=False)

            print("whole input: ", input_str)
            need_longer_response = True
            try_count = 0
            while need_longer_response and try_count < 20:
                response = send_query_to_llm_chat(input_str, model, tokenizer, num_return_sequences=1)[0].strip()
                try_count += 1
                if len(response) > 10:
                    need_longer_response = False
            print("response: ", response)
            messages.append({"role": "assistant", "content": response})
            answer = extract_answer_chat_yes_no(response)
            predict_options.append((question, response, answer))
            print(answer)
            if answer == "yes":
                previous_edges.add_edge_pair_wise(head.name, tail.name, key)
                if key == 'A':
                    cause2effect.append((head.name, tail.name))
                elif key == 'B':
                    cause2effect.append((tail.name, head.name))
    adj_matrix = previous_edges.get_adjacency_matrix([var_names_and_desc[var].name for var in df.columns])

    result_dict = {}
    for head, tail in cause2effect:
        if head in result_dict:
            # Append the tail to the existing list for this head
            result_dict[head].append(tail)
        else:
            # If the head is not a key, add it with the tail in a new list
            result_dict[head] = [tail]
    result_dict["predict_options"] = predict_options
    result_dict["adj_matrix"] = adj_matrix.tolist()
    print(result_dict)

    return adj_matrix, result_dict


def decide_final_choice(multi_choices):
    a_count = multi_choices.count('A')
    b_count = multi_choices.count('B')
    c_count = multi_choices.count('C')
    if a_count + b_count >= c_count:
        if a_count >= b_count:
            return 'A'
        elif b_count > a_count:
            return 'B'
    else:
        return 'C'
    

def openllm_pairwise_chat_multi_inference(var_names_and_desc, prompts, df, include_statistics=False, openllm_path=None, num_inference=5):
    # load open llm model
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    tokenizer = AutoTokenizer.from_pretrained(openllm_path)
    model = AutoModelForCausalLM.from_pretrained(openllm_path, quantization_config=quantization_config,
                                                 torch_dtype=torch.float16, device_map="auto", )

    # list all edges
    edges = list(combinations(df.columns, 2))
    previous_edges = PreviousEdges()
    cause2effect = []
    predict_options = []
    for e in tqdm(edges, desc="Edges"):
        if random.random() < 0.5:
            head, tail = e
        else:
            tail, head = e
        head = var_names_and_desc[head]
        tail = var_names_and_desc[tail]

        messages = [
            {"role": "user", "content": '''
            This task is to determine the cause-and-effect relationship between two variables in a causal graph. 
            Below is an example of this task.
            Here is a description of the causal variables in this causal graph:
            rain: whether it is raining
            umbrella: whether a person is carrying an umbrella
            We are interested in the causal relationship between "rain" and "umbrella".
            Which cause-and-effect relationship is more likely?
            A. "rain" causes "umbrella".
            B. "umbrella" causes "rain".
            C. There is no causal relationship between "rain" and "umbrella".
            Let's provide a step-by-step explanation, then provide your final answer within the tags <answer>A/B/C</answer>.'''},
            {"role": "assistant", "content": '''
            Step-by-Step Explanation:

            1. Observation of Natural Phenomena: It is a common observation that people carry umbrellas when it is raining or when they anticipate rain. This behavior is a preparatory action to protect oneself from getting wet.

            2. Causal Directionality: The occurrence of rain is a natural phenomenon that is not influenced by human actions such as carrying an umbrella. Therefore, it is logical to infer that the presence of rain can influence a person's decision to carry an umbrella.

            3. Counterfactual Reasoning: If "umbrella" caused "rain", we would expect rain to occur as a result of someone deciding to carry an umbrella. However, this is against our understanding of meteorological processes, where rain is determined by atmospheric conditions, not human actions.

            4. Elimination of Alternative: Option C suggests there is no causal relationship between "rain" and "umbrella". However, the predictive behavior of carrying an umbrella based on weather conditions suggests a direct causal link, where the anticipation or occurrence of rain leads to the action of carrying an umbrella for protection.

            Therefore, based on the logic that natural phenomena like rain can influence human behavior to carry items for protection (umbrellas), but not vice versa, the most likely cause-and-effect relationship is that "rain" causes "umbrella".

            The final answer is: <answer>A</answer>'''},
            {"role": "user", "content": '''Below is another example of this task.
            Here is a description of the causal variables in this causal graph:
            Eating: whether or not a person consumes food.
            Hunger: whether or not a person feels hunger.
            We are interested in the causal relationship between "Eating" and "Hunger".
            Which cause-and-effect relationship is more likely?
            A. "Eating" causes "Hunger".
            B. "Hunger" causes "Eating".
            C. There is no causal relationship between "Eating" and "Hunger".
            Let's provide a step-by-step explanation, then provide your final answer within the tags <answer>A/B/C</answer>.'''},
            {"role": "assistant", "content": '''
            Step-by-Step Explanation:

            1. Biological Signals: Hunger is a physiological signal indicating the body's need for nutrients. 

            2. Response to Hunger: The natural response to feeling hunger is to consume food. 

            3. Causal Direction: The sequence of these events indicates that hunger (the physiological need) precedes eating (the action to satisfy the need). Therefore, the causal direction flows from the sensation of hunger to the act of eating.

            4. Counterfactual Consideration: The statement "Eating causes Hunger" (Option A) contradicts typical biological responses, as eating is generally understood to satiate hunger rather than induce it. 

            5. Existence of Causal Relationship: While Option C suggests the absence of a causal relationship between eating and hunger, everyday experiences and biological knowledge affirm that hunger directly influences the decision to eat.
            The final answer is: <answer>B</answer>
         '''},
            {"role": "user", "content": '''Below is another example of this task.
            Here is a description of the causal variables in this causal graph:
            Ice cream sale: the number of ice cream sales.
            Drowning incident: the number of drowning incidents.
            We are interested in the causal relationship between "Ice cream sale" and "Drowning incident".
            Which cause-and-effect relationship is more likely?
            A. "Ice cream sale" causes "Drowning incident".
            B. "Drowning incident" causes "Ice cream sale".
            C. There is no causal relationship between "Ice cream sale" and "Drowning incident".
            Let's provide a step-by-step explanation, then provide your final answer within the tags <answer>A/B/C</answer>.'''},
            {"role": "assistant", "content": '''
        Step-by-Step Explanation:

            1. Common Cause: Both ice cream sales and drowning incidents tend to increase during the warmer months. This correlation is due to the increase in temperature, which encourages outdoor activities. People are more likely to buy ice cream and engage in activities near water bodies during this time.

            2. Lack of Direct Influence: The act of purchasing ice cream does not directly influence the occurrence of drowning incidents, nor do drowning incidents impact the sale of ice cream. 

            3. Analysis of Causality: To establish a cause-and-effect relationship, one would expect that changes in one variable directly result in changes in the other. However, altering the number of ice cream sales independently of the season (e.g., through promotions or price changes) would not logically affect drowning incident rates, and vice versa.

            4. Conclusion: Given above analysis, there is no direct causal relationship between them.
            The final answer is: <answer>C</answer>'''},
        ]

        demonstrate = ""
        for msg in messages:
            demonstrate += msg["content"] + "\n"

        query = f'''{prompts["user"]} 
                Here is a description of the causal variables in this causal graph:
                '''
        for var in var_names_and_desc:
            causal_var = var_names_and_desc[var]
            query += f'''{causal_var.name}: {causal_var.description}\n'''

        query += f'''
        Here are the causal relationships you know so far:
        {previous_edges.get_previous_relevant_edges_string(head.name, tail.name)}
        We are interested in the causal relationship between "{head.name}" and "{tail.name}".
        '''

        # if include_statistics:
        #     arr = df[[head.symbol, tail.symbol]].to_numpy().T
        #     corr_coef = np.corrcoef(arr)[0, 1]
        #     corr_coef = round(corr_coef, 2)
        #     query += f'''
        #     To help you, the Pearson correlation coefficient between "{head.name}" and "{tail.name}" is {corr_coef}
        #     '''
        query += f'''
        Which cause-and-effect relationship is more likely? 
        A. "{head.name}" causes "{tail.name}". 
        B. "{tail.name}" causes "{head.name}". 
        C. There is no causal relationship between "{head.name}" and "{tail.name}".
        Let's provide a step-by-step explanation, then provide your final answer within the tags <answer>A/B/C</answer>.'''

        messages.append({"role": "user", "content": query})
        query = demonstrate + query
        chat_template_models = ["mistral", "gemma", "llama", "falcon"]  # "falcon"
        if model.config.model_type in chat_template_models:
            query = tokenizer.apply_chat_template(messages, tokenize=False)

        multi_responses = []
        while len(multi_responses) < num_inference:
            response = send_query_to_llm_chat(query, model, tokenizer, num_return_sequences=1)[0].strip()
            if len(response.split()) > 5:
                multi_responses.append(response)

        multi_choices = []
        for response in multi_responses:
            print(response)
            one_choice = extract_answer_chat(response)
            print(one_choice)
            multi_choices.append(one_choice)

        choice = decide_final_choice(multi_choices)
        predict_options.append(multi_choices)

        previous_edges.add_edge_pair_wise(head.name, tail.name, choice)
        if choice == 'A':
            cause2effect.append((head.name, tail.name))
        elif choice == 'B':
            cause2effect.append((tail.name, head.name))
    adj_matrix = previous_edges.get_adjacency_matrix([var_names_and_desc[var].name for var in df.columns])

    result_dict = {}
    for head, tail in cause2effect:
        if head in result_dict:
            # Append the tail to the existing list for this head
            result_dict[head].append(tail)
        else:
            # If the head is not a key, add it with the tail in a new list
            result_dict[head] = [tail]
    result_dict["predict_options"] = predict_options
    result_dict["adj_matrix"] = adj_matrix.tolist()
    print(result_dict)

    return adj_matrix, result_dict
