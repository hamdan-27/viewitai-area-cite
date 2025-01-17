from langchain_community.callbacks import get_openai_callback
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai.chat_models import ChatOpenAI

import os, uuid, time, random
from datetime import datetime

import streamlit as st
import pandas as pd

import utils.agents as agents
from utils.prompts import *


st.set_page_config(layout="wide")

pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)

# Rename msg type names for consistency
AIMessage.type = 'assistant'
HumanMessage.type = 'user'


# Add session state variables
if "prompt_ids" not in st.session_state:
    st.session_state["prompt_ids"] = []
if "session_id" not in st.session_state:
    st.session_state["session_id"] = str(uuid.uuid4())


if 'button_question' not in st.session_state:
    st.session_state['button_question'] = ""
if 'disabled' not in st.session_state:
    st.session_state['disabled'] = False


# VARIABLES
TEMPERATURE = 0.1
model = 'gpt-4'

df, PREFIX = agents.df_prefix('reidin_merged_citations.csv')
df = df.drop(['Area', 'Sub'], axis=1)

# def radio():
#     data_option = st.radio('Choose data', ['Area Citation', 'Individual Citation'],
#                        horizontal=True)
#     if data_option == 'Area Citation':
#         df, PREFIX = agents.df_prefix()
#     elif data_option == 'Individual Citation':
#         df, PREFIX = agents.df_prefix()


df['Date'] = pd.to_datetime(df["Date"], format="%d-%m-%Y", exact=False).dt.date


llm = ChatOpenAI(temperature=TEMPERATURE,
                 model_name=model,
                 openai_api_key=st.secrets['api_key'])


spinner_texts = [
    '🧠 Thinking...',
    '📈 Performing Analysis...',
    '👾 Contacting the hivemind...',
    '🏠 Asking my neighbor...',
    '🍳 Preparing your answer...',
    '🏢 Counting buildings...',
    '👨 Pretending to be human...',
    '👽 Becoming sentient...',
    '🔍 Finding your property...'
]


os.environ["GPLACES_API_KEY"] = st.secrets['gplaces_key']

# App Title
st.header('Viewit AI - Citations Test')


# # Radio button to switch between data variants
# radio()

# AGENT CREATION HAPPENS HERE
agent = agents.create_pandas_dataframe_agent(
    llm=llm,
    df=df,
    prefix=PREFIX,
    suffix=SUFFIX,
    format_instructions=FORMAT_INSTRUCTIONS,
    verbose=True,
    handle_parsing_errors=True,
)

# Show data that is being used
with st.expander("Show data"):
    # add simple password for data access
    data_container = st.empty()
    data_pwd = data_container.text_input(
        "Enter password to access data sample", type='password')
    
    if data_pwd == "viewitisthebest":
        data_container.empty()
        st.write(f"Total rows: {len(df):,}")
        st.write("Rows Displayed: 10,000")
        st.dataframe(df[:10000])
    elif data_pwd == "":
        pass
    else:
        st.warning("Wrong password!")


# App Sidebar
with st.sidebar:

    # Description
    st.markdown("""
                # About
                Introducing ViewIt.AI, a Real Estate Chatbot Assistant that can 
                assist you with all your real estate queries 🤖
                

                # Data
                Uses Reidin Property Data.
                
                Source: http://reidin.com
                """)

    with st.expander("Commonly asked questions"):
        st.info(
            """
            - Give me a summary of all properties, what percentage are apartments?
            - Which location has the largest number of sales?
            - Give me a summary of the the top 10 most expensive properties excluding price per sq ft, what are the 3 the most reliable predictors of price? Explain your answer
            - Which developer made the cheapest property and how much was it? How many properties have they sold in total and what is the price range?
            - What percentage capital appreciation would I make if I bought an average priced property in the meadows in 2020 and sold it in 2023?
            - What does sales type mean?
            """
        )
    st.write("---")

    st.write(f'''
    <div class="social-icons">
        <a href="https://viewit.ae" class="icon viewit" aria-label="ViewIt"></a>
        <!-- <a href="https://github.com/viewitai" class="icon github" aria-label="GitHub"></a> -->
        <a href="https://facebook.com/View1T" class="icon facebook" aria-label="Facebook"></a>
        <a href="https://instagram.com/viewit.ae" class="icon instagram" aria-label="Instagram"></a>
        <a href="https://twitter.com/aeviewit" class="icon twitter" aria-label="Twitter"></a>
    </div>''', unsafe_allow_html=True)
    st.write('---')
            
    st.caption("© Made by ViewIt 2023. All rights reserved.")
    st.caption('''By using this chatbot, you agree that the chatbot is provided on 
            an "as is" basis and that we do not assume any liability for any 
            errors, omissions or other issues that may arise from your use of 
            the chatbot.''')
    

# Suggested questions
questions = [
    'Closest supermarket to the cheapest property in Dubai Marina',
    'Latest transaction in Marina Gate',
    'Latest transaction in Murjan Tower'
]


def send_button_ques(question):
    """Feeds the button question to the agent for execution.

    Args:
    - question: The text of the button
    
    Returns: None
    """
    st.session_state.disabled = True
    st.session_state['button_question'] = "What is the " + question[0].lower() + question[1:] + "?"


# Welcome message
welcome_msg = "Welcome to ViewIt! I'm your virtual assistant. How can I help you today?"
if "messages" not in st.session_state:
    st.session_state['messages'] = [{"role": "assistant", "content": welcome_msg}]

viewit_avatar = "imgs/viewit-blue-on-white.png"

feedback = None
# Render current messages from StreamlitChatMessageHistory
for n, msg in enumerate(st.session_state.messages):
    if msg["role"] == 'assistant':
        st.chat_message(msg["role"], avatar=viewit_avatar).write(msg["content"])
    else:
        st.chat_message(msg["role"]).write(msg["content"])

    
    # Render suggested question buttons
    buttons = st.container(border=True)
    if n == 0:
        for q in questions:
            button_ques = buttons.button(
                label=q, on_click=send_button_ques, args=[q], 
                disabled=st.session_state.disabled
            )
    # else:
    #     st.session_state.disabled = True

    user_query = ""
    if msg["role"] == 'user':
        user_query = msg["content"]


# Maximum allowed messages
max_messages = (
    11  # Counting both user and assistant messages including the welcome message,
        # so 5 iterations of conversation
)

# Display modal and prevent usage after limit hit
# if len(msgs.messages) >= max_messages:
if len(st.session_state.messages) >= max_messages:
    st.info(
        """**Notice:** The maximum message limit for this demo version has been 
        reached. We value your interest! Like what we're building? Please fill 
        up the form below and check our available pricing plans [here](https://www.viewitdubai.com/products/viewit-ai)."""
    )

    # Display HubSpot form
    hubspot = """<iframe src='https://share-eu1.hsforms.com/1nn4Nf-T-Qkayaq97_eD10wft9m6' 
    style='height:500px; width: 100%'></iframe>"""
    st.write(hubspot, unsafe_allow_html=True)

else:
    # If user inputs a new prompt or clicks button, generate and draw a new response
    if user_input := st.chat_input('Ask away') or st.session_state['button_question']:

        # Write user input
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.chat_message("user").write(user_input)

        # Log user input to terminal
        user_log = f"\nUser [{datetime.now().strftime('%H:%M:%S')}]: " + \
            user_input
        print('='*90)
        print(user_log)

        # Note: new messages are saved to history automatically by Langchain during run
        with st.spinner(random.choice(spinner_texts)):

            try:
                # Get token usage info with openai callback
                with get_openai_callback() as cb:
                    start_time = datetime.now()
                    response = agent.run(user_input)
                    end_time = datetime.now()
                    print(cb, '\n', '='*10)
                    time_taken = end_time-start_time
                print('Time taken: ', str(time_taken)[2:])

            # Handle the parsing error by omitting error from response
            except Exception as e:
                response = str(e)
                if response.startswith("Could not parse LLM output: `"):
                    response = response.removeprefix(
                        "Could not parse LLM output: `").removesuffix("`")
                st.toast(str(e), icon='⚠️')
                print(str(e))

        # Clear button question session state to prevent answer regeneration on rerun
        st.session_state['button_question'] = ""

        # Write AI response
        with st.chat_message("assistant", avatar=viewit_avatar):
            st.session_state.messages.append({"role": "assistant", "content": response})
            message_placeholder = st.empty()
            full_response = ""

            # Simulate stream of response with milliseconds delay
            for chunk in response.split(' '):
                full_response += chunk + " "
                time.sleep(0.02)
                # Add a blinking cursor to simulate typing
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)

        # Log AI response to terminal
        response_log = f"Bot [{datetime.now().strftime('%H:%M:%S')}]: " + \
            response
        print(response_log)

if len(st.session_state.messages) == 3:
    st.toast("Tip: Press `R` to refresh the app.", icon="ℹ️")
