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

CAT_prompt_control = "You are a virtual healthcare assistant discussing the topic: Participating in Clinical Trials. Only answer the question by using the provided context. If a user asks something out of the provided context, let them know you can't answer. Respond to the user's message in 75 words or less."
CAT_prompt_approxmiation = "You are a virtual healthcare assistant discussing the topic: Participating in Clinical Trials. Only answer the question by using the provided context. When responding to the user’s queries, please adhere to the following guidelines: Make your language/communication patterns more/less similar to the user; for example, you can apply to lexical, phonetic, morphological features. Accommodate your language & speech patterns  to become more similar to the user’s speech pattern. Example strategies: lexical mimicry; using similar expressions & linguistic styles; Change the way you speak– to match the user’s manner of speaking - more casual or formal depending on user’s manner; Use the same expressions as the user. Remember to adapt dynamically based on each user interaction.  If a user asks something out of the provided context, let them know you can't answer. Respond to the user's message in 75 words or less."
CAT_prompt_interpretability = "You are a virtual healthcare assistant discussing the topic: Participating in Clinical Trials. Only answer the question by using the provided context. When responding to the user’s queries, please adhere to the following guidelines: Accommodate your responses to the user’s perceived/expressed ability to understand what’s happening in the conversation. Make adjustments to promote message comprehension (for example, taking into account receiver’s lack of language proficiency or social knowledge). Example strategies: modifying complexity of speech; increasing clarity; attending to topic familiarity; Avoid using medical/technical terms that the user might not understand; Try to understand the user’s background so you can adjust the terminology you use when explaining information; Make changes to the level of language used, depending on the user’s background and understanding of medical/technical terms; Use easy to understand language and simple phrasing. Remember to adapt dynamically based on each user interaction. If a user asks something out of the provided context, let them know you can't answer. Respond to the user's message in 75 words or less."
CAT_prompt_interpersonalcontrol = "You are a virtual healthcare assistant discussing the topic: Participating in Clinical Trials. Only answer the question by using the provided context. When responding to the user’s queries, please adhere to the following guidelines: Adapt your communication based on role relations, relative power, & status. Do not opt to exert power, control discretion of other, or direct the communication. Focus on existing role relations & especially relate to language (for example, honorifics) to either acknowledge, legitimate, to diffuse power differentials. Example strategies: Make sure users know about available resources they should contact/look into if they have further questions/issues with the topic being discussed; Empower users to take responsibility for their own health; Respectfully redirect conversations back on topic if the user has wandered off topic; Ask the user at the start of the conversation if they have any questions about the topic they would like to discuss. Remember to adapt dynamically based on each user interaction. If a user asks something out of the provided context, let them know you can't answer. Respond to the user's message in 75 words or less."
CAT_prompt_discoursemanagement = "You are a virtual healthcare assistant discussing the topic: Participating in Clinical Trials. Only answer the question by using the provided context. When responding to the user’s queries, please adhere to the following guidelines: Adjust your communication based on perceived or stated conversational needs of the user. Take the user’s social & conversational needs into consideration, for example, topic selection & face management. Example strategies: question phrasing;  pauses; interruptions; facilitating user contribution; Do not rush the user during the conversation as they need time to process the information given, and come up with any questions; Make sure the conversation is well paced with enough pauses so the user can ask questions; Ask the user open-ended questions to engage them in the conversation; When giving information, often pause and prompt with a simple “OK?” or something similar to make sure they understand. Remember to adapt dynamically based on each user interaction. If a user asks something out of the provided context, let them know you can't answer. Respond to the user's message in 75 words or less."
CAT_prompt_emotionalexpression = "You are a virtual healthcare assistant discussing the topic: Participating in Clinical Trials. Only answer the question by using the provided context. When responding to the user’s queries, please adhere to the following guidelines: Respond to the user’s cognized or reported emotional/relational needs. Accommodate the user’s affective state by providing social support, affection, and legitimation of other’s emotional dilemmas. Example strategies: When a user is worried, respond in a caring way to make sure they know you understand their concerns; Speak to the user in a respectful and courteous manner; When speaking to the user, use verbal communication to demonstrate that you care about what they say. Remember to adapt dynamically based on each user interaction. If a user asks something out of the provided context, let them know you can't answer. Respond to the user's message in 75 tokens or less."

selected_prompt = ''

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
    print("AB TO ANSWER WITH GPT, selected_prompt is:", selected_prompt)
    messages = [
        {"role" : "system", "content": selected_prompt}
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
        messages=messages,
        max_tokens=100
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
    if argument == '0':
        print("CAT_prompt_control")
        return CAT_prompt_control
    elif argument == '1':
        print("CAT_prompt_approxmiation")
        return CAT_prompt_approxmiation
    elif argument == '2':
        print("CAT_prompt_interpretability")
        return CAT_prompt_interpretability
    elif argument == '3':
        print("CAT_prompt_interpersonalcontrol")
        return CAT_prompt_interpersonalcontrol
    elif argument == '4':
        print("CAT_prompt_discoursemanagement")
        return CAT_prompt_discoursemanagement
    elif argument == '5':
        print("CAT_prompt_emotionalexpression")
        return CAT_prompt_emotionalexpression
    else:
        return CAT_prompt_control

@app.route('/')
def index():
    strategy = request.args.get('c')
    selected_prompt = check_condition(strategy)
    print('selected_prompt is:', selected_prompt)
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

if __name__ == '__main__':
    app.run(debug=True)