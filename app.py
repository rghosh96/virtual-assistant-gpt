from flask import Flask, render_template, request, jsonify
import os
from openai import OpenAI
from dotenv import load_dotenv
import base64
import pandas as pd
import numpy as np
import tiktoken

app = Flask(__name__)

load_dotenv()  # take environment variables from .env.

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
print(OPENAI_API_KEY)

client = OpenAI(api_key=OPENAI_API_KEY)

speech_file_path = "output.mp3"

EMBEDDING_MODEL = "text-embedding-ada-002"
COMPLETIONS_MODEL = "gpt-3.5-turbo"

df  = pd.read_csv('dataset.csv')

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
    print(f"Selected {len(chosen_sections)} document sections:")
    print("\n".join(chosen_sections_indexes))
        
    return chosen_sections, chosen_sections_len

def answer_with_gpt(
    query: str,
    df: pd.DataFrame,
    document_embeddings: dict[(str, str), np.array],
    show_prompt: bool = False
) -> str:
    info = "You are a virtual healthcare assistant, only answer the question by using the provided context. Keep your responses to 1-2 sentences and be sure to directly address the user’s message. If a user asks something out of the provided context, let them know you can't answer."
    messages = [
        {"role" : "system", "content": info}
    ]
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

    context = context + '\n\n --- \n\n + ' + query

    messages.append({"role" : "user", "content":context})
    response = client.chat.completions.create(
        model=COMPLETIONS_MODEL,
        messages=messages
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

# prompt = "What is amoxycillan?"
# response, sections_tokens = answer_with_gpt(prompt, df, document_embeddings)
# print(response)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/chatbot', methods=['POST'])
def chatbot():
    message = request.json['message']
    # response = get_chatbot_response(message)
    # return jsonify({'message': response})

    text_response, audio_response = answer_with_gpt(message, df, document_embeddings)
    
    # Encode the audio_response (which is binary data) to base64
    audio_base64 = base64.b64encode(audio_response).decode('utf-8')
    audio_data_url = f"data:audio/wav;base64,{audio_base64}"
    
    return jsonify({'message': text_response, 'audio': audio_data_url})

# def get_chatbot_response(message):
#     # prompt = "You are a healthcare assistant who is educating people on clinical trials. Your knowledge is limited to NIH and clinicaltrials.gov information (including research studies). Keep your responses to 1-2 sentences. If a user asks something out of context, let them know you can't answer and prompt them to ask a different question about clinical trials. Here's the user's message: " + message
#     response, sections_tokens = answer_with_gpt(message, df, document_embeddings)
#     print(response)
#     try:
#         chat_completion = client.chat.completions.create(
#             messages=[
#                 {
#                     "role": "user",
#                     "content": prompt,
#                 }
#             ],
#             model="gpt-3.5-turbo",
#         )
#         print("GOT IT")
#         print(chat_completion.choices[0].message.content)

#         response = client.audio.speech.create(
#             model="tts-1",
#             voice="nova",
#             input=chat_completion.choices[0].message.content,
#         )

#         response.stream_to_file("output.mp3")

#         with open("output.mp3", "rb") as audio_file:
#             audio_response = audio_file.read()

#         return chat_completion.choices[0].message.content, audio_response
#     except Exception as e:
#         print("Error:", e)
#         return "Sorry, I couldn't process your request at the moment."

if __name__ == '__main__':
    app.run(debug=True)