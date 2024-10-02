import numpy as np
from openai import OpenAI
import random
from tqdm.autonotebook import tqdm
from .utils import PreviousEdges, adjacency_matrix_to_edge_list
from itertools import combinations


def llm_pairwise(var_names_and_desc, prompts, df, include_statistics=False):
    client = OpenAI(api_key="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")

    # list all edges
    edges = list(combinations(df.columns, 2))
    previous_edges = PreviousEdges()
    cause2effect = []
    for e in tqdm(edges):
        if random.random() < 0.5:
            head,tail = e
        else:
            tail,head = e            
        head = var_names_and_desc[head]
        tail = var_names_and_desc[tail]

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

        if include_statistics:
            arr = df[[head.symbol, tail.symbol]].to_numpy().T
            corr_coef = np.corrcoef(arr)[0,1]
            corr_coef = round(corr_coef, 2)
            query += f'''
            To help you, the Pearson correlation coefficient between "{head.name}" and "{tail.name}" is {corr_coef}
            '''
        query += f'''
        Which cause-and-effect relationship is more likely? 
        A. "{head.name}" causes "{tail.name}". 
        B. "{tail.name}" causes "{head.name}". 
        C. There is no causal relationship between "{head.name}" and "{tail.name}".
        Let’s work this out in a step by step way to be sure that we have the right answer. 
        Then provide your final answer within the tags <Answer>A/B/C</Answer>.'''
        response = client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=[
            {
                "role": "system",
                "content": prompts["system"]
            },
            {
                "role": "user",
                "content": query
            }
            ],
            temperature=0.7,
            max_tokens=2048,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        
        answer = response.choices[0].message.content
        print(response)
        idx = answer.find('<Answer>')
        if idx == -1:
            print("NO ANSWER FOUND")
            print("This was the answer:", answer)
            continue
        choice = answer[idx+8]

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

    return adj_matrix, result_dict