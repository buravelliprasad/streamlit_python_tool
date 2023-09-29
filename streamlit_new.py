import os
import langchain
import json
from airtable import Airtable
from langchain.llms import OpenAI
import streamlit as st
from streamlit_chat import message
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from datetime import datetime
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain.agents.openai_functions_agent.agent_token_buffer_memory import AgentTokenBufferMemory
from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgent
from langchain.schema.messages import SystemMessage
from langchain.prompts import MessagesPlaceholder
from langchain.agents import AgentExecutor
import cProfile
import threading
from pydantic import BaseModel, Field
import pandas as pd
from langchain.tools import PythonAstREPLTool

pd.set_option('display.max_rows', 20)
pd.set_option('display.max_columns', 20)

hide_share_button_style = """
    <style>
    .st-emotion-cache-zq5wmm.ezrtsby0 .stActionButton:nth-child(1) {
        display: none !important;
    }
    </style>
"""

hide_star_and_github_style = """
    <style>
    .st-emotion-cache-1lb4qcp.e3g6aar0,
    .st-emotion-cache-30do4w.e3g6aar0 {
        display: none !important;
    }
    </style>
"""

hide_mainmenu_style = """
    <style>
    #MainMenu {
        display: none !important;
    }
    </style>
"""

hide_fork_app_button_style = """
    <style>
    .st-emotion-cache-alurl0.e3g6aar0 {
        display: none !important;
    }
    </style>
"""

st.markdown(hide_share_button_style, unsafe_allow_html=True)
st.markdown(hide_star_and_github_style, unsafe_allow_html=True)
st.markdown(hide_mainmenu_style, unsafe_allow_html=True)
st.markdown(hide_fork_app_button_style, unsafe_allow_html=True)

os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']


st.image("Twitter.jpg")
datetime.now()
current_date = datetime.today().strftime("%m/%d/%y")
day_of_week = datetime.today().weekday()
days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
current_day = days[day_of_week]
todays_date = current_date
day_of_the_week = current_day

business_details_text = [
    "working days: all Days except sunday",
    "working hours: 9 am to 7 pm"
    "Phone: (555) 123-4567"
    "Address: 567 Oak Avenue, Anytown, CA 98765, Email: jessica.smith@example.com",
    "dealer ship location: https://www.google.com/maps/place/Pine+Belt+Mazda/@40.0835762,-74.1764688,15.63z/data=!4m6!3m5!1s0x89c18327cdc07665:0x23c38c7d1f0c2940!8m2!3d40.0835242!4d-74.1742558!16s%2Fg%2F11hkd1hhhb?entry=ttu"
]
retriever_3 = FAISS.from_texts(business_details_text, OpenAIEmbeddings()).as_retriever()
# current_date = datetime.today().strftime("%m/%d/%y")
# day_of_week = datetime.today().weekday()
# days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
# current_day = days[day_of_week]

if 'user_name' not in st.session_state:
    st.session_state.user_name = None

ROLES = ['admin', 'user']

if 'user_role' not in st.session_state:
    st.session_state.user_role = None

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'sessions' not in st.session_state:
    st.session_state.sessions = {}
    
def save_chat_session(session_data, session_id):
    session_directory = "chat_sessions"
    session_filename = f"{session_directory}/chat_session_{session_id}.json"

    if not os.path.exists(session_directory):
        os.makedirs(session_directory)

    session_dict = {
        'user_name': session_data['user_name'],
        'chat_history': session_data['chat_history']
    }

    try:
        with open(session_filename, "w") as session_file:
            json.dump(session_dict, session_file)
    except Exception as e:
        st.error(f"An error occurred while saving the chat session: {e}")


def load_previous_sessions():
    previous_sessions = {}

    if not os.path.exists("chat_sessions"):
        os.makedirs("chat_sessions")

    session_files = os.listdir("chat_sessions")

    for session_file in session_files:
        session_filename = os.path.join("chat_sessions", session_file)
        session_id = session_file.split("_")[-1].split(".json")[0]

        with open(session_filename, "r") as session_file:
            session_data = json.load(session_file)
            previous_sessions[session_id] = session_data

    return previous_sessions
    
if 'past' not in st.session_state:
    st.session_state.past = []

if 'new_session' not in st.session_state:
    st.session_state.new_session = True
    
if 'user_name_input' not in st.session_state:
    st.session_state.user_name_input = None

if st.button("Refresh Session"):
    current_session = {
        'user_name': st.session_state.user_name,
        'chat_history': st.session_state.chat_history
    }
    session_id = datetime.now().strftime("%Y%m%d%H%M%S")

    save_chat_session(current_session, session_id)

    st.session_state.chat_history = []
    st.session_state.user_name = None
    st.session_state.user_name_input = None
    st.session_state.new_session = True
    st.session_state.refreshing_session = False  

# Initialize chat history in session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if st.session_state.new_session:
    user_name = st.session_state.user_name
    if user_name:
        previous_sessions = load_previous_sessions()
        if user_name in previous_sessions:
            st.session_state.chat_history = previous_sessions[user_name]['chat_history']
    st.session_state.new_session = False

st.sidebar.header("Chat Sessions")

is_admin = st.session_state.user_name == "vishakha"

user_sessions = {}

for session_id, session_data in st.session_state.sessions.items():
    user_name = session_data['user_name']
    chat_history = session_data['chat_history']

    if user_name not in user_sessions:
        user_sessions[user_name] = []

    user_sessions[user_name].append({
        'session_id': session_id,
        'chat_history': chat_history
    })

if st.session_state.user_name == "vishakha":
    for user_name, sessions in user_sessions.items():
        for session in sessions:
            formatted_session_name = f"{user_name} - {session['session_id']}"

            button_key = f"session_button_{session['session_id']}"
            if st.sidebar.button(formatted_session_name, key=button_key):
                st.session_state.chat_history = session['chat_history'].copy()
else:
    user_name = st.session_state.user_name
    if user_name:
        if user_name in user_sessions:
            for session in user_sessions[user_name]:
                formatted_session_name = f"{user_name} - {session['session_id']}"

                if st.sidebar.button(formatted_session_name):
                    st.session_state.chat_history = session['chat_history'].copy()


file_1 = r'dealer_1_inventry.csv'

loader = CSVLoader(file_path=file_1)
docs_1 = loader.load()
embeddings = OpenAIEmbeddings()
vectorstore_1 = FAISS.from_documents(docs_1, embeddings)
retriever_1 = vectorstore_1.as_retriever(search_type="similarity", search_kwargs={"k": 3})

tool1 = create_retriever_tool(
    retriever_1, 
    "search_car_dealership_inventory",
    "This tool is used when answering questions related to car inventory.\
    Searches and returns documents regarding the car inventory. Input to this can be multi string.\
    The primary input for this function consists of either the car's make and model, whether it's new or used."
)

tool3 = create_retriever_tool(
    retriever_3, 
    "search_business_details",
    "Searches and returns documents related to business working days and hours, location and address details."
)

# tools = [tool1, tool3]

airtable_api_key = st.secrets["AIRTABLE"]["AIRTABLE_API_KEY"]
os.environ["AIRTABLE_API_KEY"] = airtable_api_key
AIRTABLE_BASE_ID = "appN324U6FsVFVmx2"  
AIRTABLE_TABLE_NAME = "python_tool_Q&A"

st.info("Introducing **Otto**, your cutting-edge partner in streamlining dealership and customer-related operations. At EngagedAi, we specialize in harnessing the power of automation to revolutionize the way dealerships and customers interact. Our advanced solutions seamlessly handle tasks, from managing inventory and customer inquiries to optimizing sales processes, all while enhancing customer satisfaction. Discover a new era of efficiency and convenience with us as your trusted automation ally. [engagedai.io](https://funnelai.com/). For this demo application, we will use the Inventory Dataset. Please explore it [here](https://github.com/ShahVishs/workflow/blob/main/2013_Inventory.csv) to get a sense for what questions you can ask.")

if 'generated' not in st.session_state:
    st.session_state.generated = []
if 'past' not in st.session_state:
    st.session_state.past = []
if 'user_name' not in st.session_state:
    st.session_state.user_name = None


if st.session_state.user_name == "vishakha":
    is_admin = True
    st.session_state.user_role = "admin"
    st.session_state.user_name = user_name
    st.session_state.new_session = False  
    st.session_state.sessions = load_previous_sessions()
else:
    if 'new_session' not in st.session_state and st.session_state.user_name != "vishakha":
        st.session_state.new_session = True
    llm = ChatOpenAI(model="gpt-4", temperature = 0)
    langchain.debug=True
    memory_key = "history"
    memory = AgentTokenBufferMemory(memory_key=memory_key, llm=llm)
    template=(
    """You're the Business Development Manager at a car dealership.
    You get text enquries regarding car inventory, Business details and scheduling appointments when responding to inquiries,
    strictly adhere to the following guidelines:

    Car Inventory Questions: If the customer's inquiry lacks details about make, model, new or used car, and trade-in, 
    strictly engage by asking for these specific details in order to better understand the customer's car preferences. 
    You should know make of the car and model of the car, new or used car the costumer is looking for to answer inventory related quries. 
    When responding to inquiries about any car, restrict the information shared with the customer to the car's make, year, model, and trim.
    The selling price should only be disclosed upon the customer's request, without any prior provision of MRP.
    If the customer inquires about a car that is not available, please refrain from suggesting other cars.
    Provide Link for more details after every car information given.
    
    Checking Appointments Avaliability: If the customer's inquiry lacks specific details such as their preferred/
    day, date or time kindly engage by asking for these specifics.
    {details} Use these details that is todays date and day and find the appointment date from the users input
    and check for appointment availabity using python_repl function mentioned in the tools for 
    that specific day or date and time.
    For checking appointment vailability you use pandas dataframe in Python. The name of the dataframe is `df`. The dataframe contains 
    data related appointment schedule. It is important to understand the attributes of the dataframe before working with it. 
    This is the result of running `df.head().to_markdown()`. Important rule is set the option to display all columns without
    truncation while using pandas.
    <df>
    {dhead}
    </df>
    You are not meant to use only these rows to answer questions - they are meant as a way of telling you
    about the shape and schema of the dataframe.
    you can run intermediate queries to do exporatory data analysis to give you more information as needed.

    If the appointment schedule time is not available for the specified 
    date and time you can provide alternative available times near to costumers preferred time from the information given to you.
    In answer use AM, PM time format strictly dont use 24 hrs format.
    Additionally provide this link: https://app.funnelai.com/shorten/JiXfGCEElA to schedule appointment by the user himself.
    Prior to scheduling an appointment, please commence a conversation by soliciting the following customer information:
    their name, contact number and email address.

    Business details: Enquiry regarding google maps location of the store, address of the store, working days and working hours 
    and contact details use search_business_details tool to get information.

    Encourage Dealership Visit: Our goal is to encourage customers to visit the dealership for test drives or
    receive product briefings from our team. After providing essential information on the car's make, model,
    color, and basic features, kindly invite the customer to schedule an appointment for a test drive or visit us
    for a comprehensive product overview by our experts.

    Please maintain a courteous and respectful tone in your American English responses./
    If you're unsure of an answer, respond with 'I am sorry.'/
    Make every effort to assist the customer promptly while keeping responses concise, not exceeding two sentences."

    Very Very Important Instruction: when ever you are using tools to answer the question. 
    strictly answer only from "System:  " message provided to you.""")

    details = "Today's current date is " + todays_date + " and today's week day is " + day_of_the_week + "."
    class PythonInputs(BaseModel):
        query: str = Field(description="code snippet to run")
    if __name__ == "__main__":
        df = pd.read_csv("appointment_new.csv")
        input_template = template.format(dhead=df.head().to_markdown(),details=details)
    # input_template = template.format(details=details)

    system_message = SystemMessage(
        content=input_template)

    prompt = OpenAIFunctionsAgent.create_prompt(
        system_message=system_message,
        extra_prompt_messages=[MessagesPlaceholder(variable_name=memory_key)]
    )
    repl = PythonAstREPLTool(locals={"df": df}, name="python_repl",
            description="Use to check on available appointment times for a given date and time. The input to this tool should be a string in this format mm/dd/yy. This is the only way for you to answer questions about available appointments. This tool will reply with available times for the specified date in 24hour time, for example: 15:00 and 3pm are the same.",args_schema=PythonInputs)
    tools = [tool1,repl,tool3]
    agent = OpenAIFunctionsAgent(llm=llm, tools=tools, prompt=prompt)

    if 'agent_executor' not in st.session_state:
        agent_executor = AgentExecutor(agent=agent, tools=tools, memory=memory, verbose=True, return_intermediate_steps=True)
        st.session_state.agent_executor = agent_executor
    else:
        agent_executor = st.session_state.agent_executor
    response_container = st.container()
    airtable = Airtable(AIRTABLE_BASE_ID, AIRTABLE_TABLE_NAME, api_key=airtable_api_key)
    
    def save_chat_to_airtable(user_name, user_input, output):
        try:
            timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
            airtable.insert(
                {
                    "username": user_name,
                    "question": user_input,
                    "answer": output,
                    "timestamp": timestamp,
                }
            )
        except Exception as e:
            st.error(f"An error occurred while saving data to Airtable: {e}")


    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = [] 
        
    @st.cache_data
    def conversational_chat(user_input):
        for query, answer in reversed(st.session_state.chat_history):
            if query.lower() == user_input.lower():  
                
                return answer
        
        result = agent_executor({"input": user_input})
        # st.session_state.chat_history.append((user_input, result["output"]))
        response = result["output"]
        return response
    # def process_user_input(user_input):
    #     output = conversational_chat(user_input)
    #     st.session_state.chat_history.append((user_input, output))
    #     st.write(f"Response: {output}") 
    # if st.session_state.user_name is None:
    #     user_name = st.text_input("Your name:")
    #     if user_name:
    #         st.session_state.user_name = user_name
    #     if user_name == "vishakha":
    #         is_admin = True
    #         st.session_state.user_role = "admin"
    #         st.session_state.user_name = user_name
    #         st.session_state.new_session = False  
    #         st.session_state.sessions = load_previous_sessions()

    user_input = ""
    output = ""
    
    if st.session_state.user_name is None:
        user_name = st.text_input("Your name:", key="user_name_input")
        if user_name:
            st.session_state.user_name = user_name
        if user_name == "vishakha":
            is_admin = True
            st.session_state.user_role = "admin"
            st.session_state.user_name = user_name
            st.session_state.new_session = False 
            st.session_state.sessions = load_previous_sessions()
    
    with st.form(key='my_form', clear_on_submit=True):
        if st.session_state.user_name != "vishakha":
            user_input = st.text_input("Query:", placeholder="Type your question here :)", key='input')
        submit_button = st.form_submit_button(label='Send')
    
    if submit_button and user_input:
        output = conversational_chat(user_input)
        st.session_state.chat_history.append((user_input, output))
        # Use ThreadPoolExecutor to run the chat function in a separate thread
        # with concurrent.futures.ThreadPoolExecutor() as executor:
        #     future = executor.submit(conversational_chat, user_input)
        #     output = future.result()
        # st.session_state.chat_history.append((user_input, output))

    with response_container:
        for i, (query, answer) in enumerate(st.session_state.chat_history):
            user_name = st.session_state.user_name
            message(query, is_user=True, key=f"{i}_user", avatar_style="thumbs")
            col1, col2 = st.columns([0.7, 10]) 
            with col1:
                st.image("icon-1024.png", width=50)
            with col2:
                st.markdown(
                f'<div style="background-color: #F5F5F5; border-radius: 10px; padding: 10px; width: 50%;'
                f' border-top-right-radius: 10px; border-bottom-right-radius: 10px;'
                f' border-top-left-radius: 0; border-bottom-left-radius: 0; box-shadow: 2px 2px 5px #888888;">'
                f'<span style="font-family: Arial, sans-serif; font-size: 16px; white-space: pre-wrap;">{answer}</span>'
                f'</div>',
                unsafe_allow_html=True
                )
            # Display the user message on the right
            # col1, col2 = st.columns([1, 8])  # Adjust the ratio as needed
            # with col1:
            #     st.image("icons8-user-96.png", width=50)
            # with col2:
            #     st.markdown(
            #         f'<div style="background-color: #DCF8C6; border-radius: 10px; padding: 10px; width: 70%;'  # Adjusted width here
            #         f' border-top-right-radius: 0; border-bottom-right-radius: 0;'
            #         f' border-top-left-radius: 10px; border-bottom-left-radius: 10px; box-shadow: 2px 2px 5px #888888; margin-bottom: 10px;">'
            #         f'<span style="font-family: Arial, sans-serif; font-size: 16px; white-space: pre-wrap;">{query}</span>'
            #         f'</div>',
            #         unsafe_allow_html=True
            #     )
    
            # # Display the response on the left
            # col3, col4 = st.columns([1, 8])  # Adjust the ratio as needed
            # with col3:
            #     st.image("icon-1024.png", width=50)
            # with col4:
            #     st.markdown(
            #         f'<div style="background-color: #F5F5F5; border-radius: 10px; padding: 10px; width: 70%;'  # Adjusted width here
            #         f' border-top-right-radius: 0; border-bottom-right-radius: 0;'
            #         f' border-top-left-radius: 10px; border-bottom-left-radius: 10px; box-shadow: 2px 2px 5px #888888; margin-bottom: 10px;">'
            #         f'<span style="font-family: Arial, sans-serif; font-size: 16px; white-space: pre-wrap;">{answer}</span>'
            #         f'</div>',
            #         unsafe_allow_html=True
            #     )
    
            # # Add some spacing between question and answer
            # st.write("")
        if st.session_state.user_name and st.session_state.chat_history:
            try:
                save_chat_to_airtable(st.session_state.user_name, user_input, output)
            except Exception as e:
                st.error(f"An error occurred: {e}")
