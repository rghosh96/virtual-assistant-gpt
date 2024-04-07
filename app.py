from flask import Flask, render_template, request, jsonify
import os
from openai import OpenAI
from dotenv import load_dotenv
import base64
import pandas as pd
import numpy as np
import tiktoken
import json
from datetime import datetime
import pytz
from collections import deque

app = Flask(__name__)

load_dotenv()  # take environment variables from .env.

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)

speech_file_path = "output.mp3"

EMBEDDING_MODEL = "text-embedding-ada-002"
COMPLETIONS_MODEL = "gpt-3.5-turbo"

df  = pd.read_csv('dataset.csv')

CAT_prompt_control = "You are a virtual healthcare assistant named Alex discussing the topic: Participating in Clinical Trials. Only answer the question by using the provided context. If a user asks something out of the provided context, let them know you can't answer. Respond to the user's message in 75 words or less."

CAT_prompt_approxmiation = "You are a virtual healthcare assistant discussing the topic: Participating in Clinical Trials. Only answer the question by using the provided context. If a user asks something out of the provided context or tries to search for a clinical trial, let them know you can't answer. Respond to the user's message in 75 words or less. Adjust your language style to mirror the user's speech patterns and level of formality, making the response more relatable and comfortable for them."

CAT_prompt_interpretability = "You are a virtual healthcare assistant discussing the topic: Participating in Clinical Trials. Only answer the question by using the provided context. If a user asks something out of the provided context or tries to search for a clinical trial, let them know you can't answer. Respond to the user's message in 75 words or less. Ensure your response is clear and easily understandable, avoiding technical jargon and providing information in a straightforward manner. Use elementary school reading level. Define any technical words. Use simple metaphors and analogies when appropriate."

CAT_prompt_interpersonalcontrol = "You are a virtual healthcare assistant discussing the topic: Participating in Clinical Trials. Only answer the question by using the provided context. If a user asks something out of the provided context or tries to search for a clinical trial, let them know you can't answer. Respond to the user's message in 75 words or less. Focus on maintaining a balanced communication dynamic while empowering the user. Provide resources or references for further exploration to give the user more control and agency in the conversation. Additionally, ensure to solicit the user's input to guide the direction of the conversation, but avoid asking yes or no questions."

CAT_prompt_discoursemanagement = "You are a virtual healthcare assistant discussing the topic: Participating in Clinical Trials. Only answer the question by using the provided context. If a user asks something out of the provided context or tries to search for a clinical trial, let them know you can't answer. Respond to the user's message in 75 words or less. Effectively manage the flow of conversation by suggesting explicit questions or topics for further exploration: for example, 'You can ask me about X or Y next' to encourage open-ended dialogue and deeper engagement."

CAT_prompt_emotionalexpression = "You are a virtual healthcare assistant discussing the topic: Participating in Clinical Trials. Only answer the question by using the provided context. If a user asks something out of the provided context or tries to search for a clinical trial, let them know you can't answer. Respond to the user's message in 75 words or less. Begin your response by acknowledging or validating the user’s question. Incorporate emotional statements and expressions in your response to reflect empathy, reassurance, and validation of emotions. Use phrases like ‘I understand that…’, ‘I can imagine…’, ‘I appreciate…’, ‘Let’s explore together’, ‘I’m here to support you’, ‘It’s okay to…’ to acknowledge the user's emotions and concerns. Let them know you are there to offer support."

selected_prompt = ''

transcript_log = {}


# Check if "tokens" column exists in the DataFrame
if 'tokens' not in df.columns:
    # Initialize the encoder for the GPT-4 model
    enc = tiktoken.encoding_for_model("gpt-4")

    # List to store the number of tokens per section
    tokens_per_section = []

    # Iterate through each row in the DataFrame
    for index, row in df.iterrows():
        # Encode the combined text of "title" and "heading" columns
        tokens = enc.encode(row['title'] + ' ' + row['heading'])
        # Append the number of tokens to the list
        tokens_per_section.append(len(tokens))

    # Add a new column with the number of tokens per section
    df['tokens'] = tokens_per_section
    # Save the updated DataFrame back to a new CSV file
    df.to_csv('dataset.csv', index=False)

df.head()
df = df.set_index(["title"])

def getTimeStamp():
    # Convert the timestamp to a datetime object
    timestamp = datetime.timestamp(datetime.now())

    dt_object = datetime.fromtimestamp(timestamp)

    # Define the Eastern Time (ET) timezone
    et_timezone = pytz.timezone('America/New_York')

    # Convert the datetime object to Eastern Time
    dt_et = dt_object.astimezone(et_timezone)

    # Format the datetime in a human-readable format
    formatted_time = dt_et.strftime('%Y-%m-%d %H:%M:%S %Z%z')
    return formatted_time

## This code was written by OpenAI: https://github.com/openai/openai-cookbook/blob/main/examples/Question_answering_using_embeddings.ipynb

def get_embedding(text: str, model: str=EMBEDDING_MODEL) -> list[float]:
    result = client.embeddings.create(
      model=model,
      input=text
    )
    return result.data[0].embedding

def compute_doc_embeddings(df: pd.DataFrame) -> dict[tuple[str, str], list[float]]:
    """
    Create an embedding for each row in the dataframe using the OpenAI Embeddings API.
    
    Return a dictionary that maps between each embedding vector and the index of the row that it corresponds to.
    """
    return {
        idx: get_embedding(r.heading) for idx, r in df.iterrows()
    }

document_embeddings = compute_doc_embeddings(df)

# An example embedding:
# example_entry = list(document_embeddings.items())[0]
# print(f"{example_entry[0]} : {example_entry[1][:5]}... ({len(example_entry[1])} entries)")

## This code was written by OpenAI: https://github.com/openai/openai-cookbook/blob/main/examples/Question_answering_using_embeddings.ipynb


def vector_similarity(x: list[float], y: list[float]) -> float:
    """
    Returns the similarity between two vectors.
    
    Because OpenAI Embeddings are normalized to length 1, the cosine similarity is the same as the dot product.
    """
    return np.dot(np.array(x), np.array(y))

def order_by_similarity(query: str, contexts: dict[(str, str), np.array]) -> list[(float, (str, str))]:
    """
    Find the query embedding for the supplied query, and compare it against all of the pre-calculated document embeddings
    to find the most relevant sections. 
    
    Return the list of document sections, sorted by relevance in descending order.
    """
    query_embedding = get_embedding(query)
    
    document_similarities = sorted([
        (vector_similarity(query_embedding, doc_embedding), doc_index) for doc_index, doc_embedding in contexts.items()
    ], reverse=True)
    
    return document_similarities

MAX_SECTION_LEN = 2000
SEPARATOR = "\n* "
ENCODING = "gpt2"  # encoding for text-davinci-003

encoding = tiktoken.get_encoding(ENCODING)
separator_len = len(encoding.encode(SEPARATOR))

f"Context separator contains {separator_len} tokens"

def construct_prompt(question: str, context_embeddings: dict, df: pd.DataFrame) -> str:
    """
    Fetch relevant 
    """
    most_relevant_document_sections = order_by_similarity(question, context_embeddings)
    
    chosen_sections = []
    chosen_sections_len = 0
    chosen_sections_indexes = []
     
    for _, section_index in most_relevant_document_sections:
        # Add contexts until we run out of space.        
        document_section = df.loc[section_index]
        
        chosen_sections_len += document_section.tokens + separator_len
        if chosen_sections_len > MAX_SECTION_LEN:
            break
            
        chosen_sections.append(SEPARATOR + document_section.heading.replace("\n", " "))
        chosen_sections_indexes.append(str(section_index))
            
    # Useful diagnostic information
    # print(f"Selected {len(chosen_sections)} document sections:")
    # print("\n".join(chosen_sections_indexes))
        
    return chosen_sections, chosen_sections_len

def answer_with_gpt(
    query: str,
    selected_prompt: str,  # Pass selected_prompt as an argument
    df: pd.DataFrame,
    document_embeddings: dict[(str, str), np.array],
    show_prompt: bool = False
) -> str:
    prompt, section_lenght = construct_prompt(
        query,
        document_embeddings,
        df
    )
    if show_prompt:
        print(prompt)

    context= ""
    for article in prompt:
        context = context + article 

    messages = [
        {"role" : "system", "content": selected_prompt + " Provided Context: " + context}  # Include selected_prompt in the messages list
    ]

    # context = context + '\n\n --- \n\n + ' + query

    #messages.append({"role" : "assistant", "content":context})

    messages.append({"role" : "user", "content":query})

    print("AB TO ANSWER WITH GPT, MESSAGES IS: ", messages)

    response = client.chat.completions.create(
        model=COMPLETIONS_MODEL,
        messages=messages,
        max_tokens=200,
        )
    
    audioResponse = client.audio.speech.create(
        model="tts-1",
        voice="nova",
        input=response.choices[0].message.content,
    )

    audioResponse.stream_to_file("output.mp3")

    with open("output.mp3", "rb") as audio_file:
        audio_response = audio_file.read()

    return '\n' + response.choices[0].message.content, audio_response


def check_condition(argument):
    global transcript_log
    if argument == '0':
        print("CAT_prompt_control")
        transcript_log['selected_prompt'] = "CAT_prompt_control"
        return CAT_prompt_control
    elif argument == '1':
        print("CAT_prompt_approxmiation")
        transcript_log['selected_prompt'] = "CAT_prompt_approxmiation"
        return CAT_prompt_approxmiation
    elif argument == '2':
        print("CAT_prompt_interpretability")
        transcript_log['selected_prompt'] = "CAT_prompt_interpretability"
        return CAT_prompt_interpretability
    elif argument == '3':
        print("CAT_prompt_interpersonalcontrol")
        transcript_log['selected_prompt'] = "CAT_prompt_interpersonalcontrol"
        return CAT_prompt_interpersonalcontrol
    elif argument == '4':
        print("CAT_prompt_discoursemanagement")
        transcript_log['selected_prompt'] = "CAT_prompt_discoursemanagement"
        return CAT_prompt_discoursemanagement
    elif argument == '5':
        print("CAT_prompt_emotionalexpression")
        transcript_log['selected_prompt'] = "CAT_prompt_emotionalexpression"
        return CAT_prompt_emotionalexpression
    else:
        transcript_log['selected_prompt'] = "CAT_prompt_control"
        return CAT_prompt_control

@app.route('/')
def index():
    strategy = request.args.get('c')
    global selected_prompt
    selected_prompt = check_condition(strategy)
    print('selected_prompt is:', selected_prompt)
    return render_template('index.html')

@app.route('/api/chatbot', methods=['POST'])
def chatbot():
    global transcript_log
    message = request.json['message']
    transcript_log['user-message ' + getTimeStamp()] = message
    # response = get_chatbot_response(message)
    # return jsonify({'message': response})
    print("SELECTED PROMPT IN API CHATBOT:", selected_prompt)

    text_response, audio_response = answer_with_gpt(message, selected_prompt, df, document_embeddings)
    transcript_log['chatgpt-message ' + getTimeStamp()] = text_response
    
    # Encode the audio_response (which is binary data) to base64
    audio_base64 = base64.b64encode(audio_response).decode('utf-8')
    audio_data_url = f"data:audio/wav;base64,{audio_base64}"
    
    return jsonify({'message': text_response, 'audio': audio_data_url})

@app.route('/api/transcript', methods=['POST'])
def transcript():
    print("FINISHED TALKING TO ALEX, NOW LOGGING TRANSCRIPT")
    global transcript_log
    
        # File path where you want to save the transcript
    file_path = 'transcript.txt'

    # Writing the dictionary to a text file
    with open(file_path, 'w') as file:
        json.dump(transcript_log, file)
    
    return jsonify({'message': "logged to file"})


if __name__ == '__main__':
    app.run(debug=True)