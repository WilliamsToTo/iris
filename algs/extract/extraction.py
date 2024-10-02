import re
import torch
import pandas as pd
from openai import OpenAI
from collections import Counter
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

def send_query_to_openai(message, model):
    key = "your api key"
    client = OpenAI(api_key=key)
    completion = client.chat.completions.create(
                            model=model,
                            messages=message
                        )
    print(completion)
    print(completion.choices[0].message.content)
    chat_gpt_response = completion.choices[0].message.content

    return chat_gpt_response

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
                                      repetition_penalty=1.25, temperature=0.8, max_new_tokens=1000,
                                      eos_token_id=terminators, num_return_sequences=num_return_sequences)
        generate_response = tokenizer.batch_decode(generate_ids[:, inputs.input_ids.shape[-1]:], skip_special_tokens=True,
                                             clean_up_tokenization_spaces=False)

    return generate_response

def extract_first_float(text):
    # Use a regular expression to find all floats in the string
    matches = re.findall(r'[-+]?\d*\.\d+|\d+', text)
    # Check if at least one float is found
    if matches:
        # Return the first match converted to float
        return float(matches[0])
    else:
        # Return None if no float is found
        return None

def extract_value(response, var_type):
    response = response.lower()
    if var_type == "bool":
        # extract True or False after 'The value of '
        target_substring = "the value of"
        start_index = response.find(target_substring)
        # If the substring is found
        if start_index != -1:
            # Start searching after the target substring
            start_index += len(target_substring)
            # Further narrow down to the relevant part of the string
            relevant_part = response[start_index:]
            print(relevant_part)
            # Check which comes first and is found
            if "true" in relevant_part:
                return "True"
            elif "false" in relevant_part:
                return "False"
            else:
                return None
        else:
            return None


def extract_variable(response):
    response = response.lower()
    pattern = r"<var>(.*?)</var>"
    match = re.search(pattern, response)

    # Check if a match was found
    if match:
        # Return the matched group
        return match.group(1)  # This returns the content within the parentheses
    else:
        # Return None if no match was found
        return None


def extract_relations(response):
    response = response.lower()
    # Define a regular expression pattern to find all matches of "(cause -> effect)"
    pattern = r"\((.*?) -> (.*?)\)"

    # Use re.findall to extract all matches as a list of tuples
    matches = re.findall(pattern, response)

    # Convert each match to a tuple
    relations = [(cause.strip(), effect.strip()) for cause, effect in matches]

    return relations


def extract_value_of_variables(doc, variables, model, tokenizer):
    # create query and generate answer
    doc = " ".join(doc)
    doc = " ".join(doc.split())
    doc_info = f"Given a document: {doc}\n\n"
    exacted_values = {}
    for var, value_type in variables.items():
        if value_type == 'bool':
            query = doc_info + f"Please complete the below task.\n" \
                     f"We have a variable named '{var}'. The value of variable '{var}' is True or False. \n" \
                     f"True indicates that the existence of '{var}' can be inferred from the document, whereas False suggests that the existence of '{var}' cannot be inferred from this document." \
                     f"Based on the document provided, what is the most appropriate value for '{var}' that can be inferred?\n" \
                     f"Please form the answer according to the following format. " \
                     f"First, provide an introductory sentence that explains what information will be discussed.  " \
                     f"Next, list generated answer in detail, ensuring clarity and precision. " \
                     f"Finally, conclude the final answer of the inferred value for '{var}' using the following template:\n" \
                     f"The value of '{var}' is ____. "
            message = [
                {"role": "system", "content": "You are a helpful assistant for analyzing, abstracting and processing text data. "
                                              "You should always answer as helpfully as possible. "
                                              "Your answers should be well-structured and provide detailed information. "
                                              "They should also have an engaging tone."},
                {"role": "user", "content": query}
            ]
            if not isinstance(model, str):
                input_str = tokenizer.apply_chat_template(message, add_generation_prompt=True, tokenize=False)
                print(input_str)
                response = send_query_to_llm_chat(input_str, model, tokenizer, num_return_sequences=1)
                print(response)
                value = extract_value(response[0], var_type=value_type)
                print(value)
            else:
                response = send_query_to_openai(message=message, model=model)
                print(response)
                value = extract_value(response, var_type=value_type)
                print(value)
        else:
            value="value type is not supported"

        exacted_values[var] = value

    return exacted_values

def extract_verification_result(response):
    response = response.lower()
    # Check which comes first and is found
    if "true" in response:
        return "True"
    elif "false" in response:
        return "False"
    else:
        return "Unknown"

def causal_claim_verification(claim, docs, model, tokenizer):
    all_veracity = []
    for url, doc in docs.items():
        doc = " ".join(doc)
        doc = " ".join(doc.split())
        doc_info = f"Given a document: {doc}\n\n"
        query = doc_info + f"Please complete the below task.\n" \
                           f"We have claim: '{claim}'." \
                           f"We need to check the veracity of this claim. The value of veracity is True or False or Unknown." \
                           f"True indicates that the given document supports this claim. False indicates that the given document refutes the claim. Unknown indicates that the given document has no relation to the claim."\
                           f"Please form the answer with a logical reasoning chain according to the following format. " \
                           f"First, provide an introductory sentence that explains what information will be discussed. " \
                           f"Next, list the logical reasoning chain in detail, ensuring clarity and precision. " \
                           f"Finally, conclude the veracity of claim '{claim}' using the following template:\n" \
                           f"The veracity of claim '{claim}' is ___. "
        message = [
            {"role": "system",
             "content": "You are a helpful assistant for verifying claim using given document. "
                        "You should always answer as helpfully as possible. "
                        "Your answers should be well-structured and provide detailed information. "
                        "They should also have an engaging tone."},
            {"role": "user", "content": query}
        ]
        if not isinstance(model, str):
            input_str = tokenizer.apply_chat_template(message, add_generation_prompt=True, tokenize=False)
            print(input_str)
            response = send_query_to_llm_chat(input_str, model, tokenizer, num_return_sequences=1)
            print(response)
            veracity = extract_verification_result(response[0])
            print(veracity)
        else:
            response = send_query_to_openai(message=message, model=model)
            veracity = extract_verification_result(response)
            print(veracity)

        all_veracity.append(veracity)

    return all_veracity


def abstract_new_variables(doc, variables, model, tokenizer):
    # create query and generate answer
    doc = " ".join(doc)
    doc = " ".join(doc.split())
    doc_info = f"Given a document: {doc}\n\n"

    variable_names = ", ".join(variables)

    query = doc_info + f"Please complete the below task.\n" \
             f"We have some given variables: '{variable_names}'." \
             f"What are the high-level variables in the provided document that have causal relations to variables in the given variable set?\n" \
             f"Please form the answer according to the following format. " \
             f"First, propose as many variables as possible that have causal relationships with the given variables, based on your understanding of the document. Please ensure these proposed variables are different from the ones already provided. " \
             f"Next, refine your list of candidate variables by reducing semantic overlap among them and shortening their names for clarity. " \
             f"Finally, determine the most reliable variable candidate as the final answer using the template provided below:\n" \
             f"The new abstracted variable is <var>____</var>. "
    message = [
        {"role": "system", "content": "You are a helpful assistant for analyzing, abstracting and processing text data. "
                                      "You should always answer as helpfully as possible. "
                                      "Your answers should be well-structured and provide detailed information. "
                                      "They should also have an engaging tone."},
        {"role": "user", "content": query}
    ]
    if not isinstance(model, str):
        input_str = tokenizer.apply_chat_template(message, add_generation_prompt=True, tokenize=False)
        print(input_str)
        response = send_query_to_llm_chat(input_str, model, tokenizer, num_return_sequences=1)
        print(response)
        variable = extract_variable(response[0])
        print(variable)
    else:
        response = send_query_to_openai(message=message, model=model)
        print(response)
        variable = extract_variable(response)
        print(variable)


    return variable
