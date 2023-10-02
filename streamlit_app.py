import os
import streamlit as st
from airtable import Airtable
from langchain.chains.router import MultiRetrievalQAChain
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.indexes import VectorstoreIndexCreator
from streamlit_chat import message
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from pytz import timezone
import datetime
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain.agents.agent_toolkits import create_conversational_retrieval_agent
from langchain.chat_models import ChatOpenAI
import langchain
from langchain.agents.openai_functions_agent.agent_token_buffer_memory import AgentTokenBufferMemory
from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgent
from langchain.schema.messages import SystemMessage
from langchain.prompts import MessagesPlaceholder
from langchain.agents import AgentExecutor
from langchain.smith import RunEvalConfig, run_on_dataset
from pydantic import BaseModel, Field
import pandas as pd
from langchain.tools import PythonAstREPLTool

pd.set_option('display.max_rows', 20)
pd.set_option('display.max_columns', 20)

os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']
st.image("socialai.jpg")

datetime.datetime.now()
# datetime.now()
# Get the current date in "%m/%d/%y" format
current_date = datetime.date.today().strftime("%m/%d/%y")
# current_date = datetime.today().strftime("%m/%d/%y")
# Get the day of the week (0: Monday, 1: Tuesday, ..., 6: Sunday)
day_of_week = datetime.date.today().weekday()
# day_of_week = datetime.today().weekday()

# Convert the day of the week to a string representation
days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
current_day = days[day_of_week]

# print("Current date:", current_date)
# print("Current day:", current_day)
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

file_1 = r'dealer_1_inventry.csv'

loader = CSVLoader(file_path=file_1)
docs_1 = loader.load()
embeddings = OpenAIEmbeddings()
vectorstore_1 = FAISS.from_documents(docs_1, embeddings)
retriever_1 = vectorstore_1.as_retriever(search_type="similarity", search_kwargs={"k": 3})#check without similarity search and k=8
# retriever_1 = vectorstore_1.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.65,"k": 3})

# Create the first tool
tool1 = create_retriever_tool(
    retriever_1, 
     "search_car_dealership_inventory",
     "This tool is used when answering questions related to car inventory.\
      Searches and returns documents regarding the car inventory. Input to this can be multi string.\
      The primary input for this function consists of either the car's make and model, whether it's new or used."
)


# Create the third tool
tool3 = create_retriever_tool(
    retriever_3, 
    "search_business_details",
    "Searches and returns documents related to business working days and hours, location and address details."
)

# Append all tools to the tools list

# airtable
airtable_api_key = st.secrets["AIRTABLE"]["AIRTABLE_API_KEY"]
os.environ["AIRTABLE_API_KEY"] = airtable_api_key
AIRTABLE_BASE_ID = "appN324U6FsVFVmx2"  
AIRTABLE_TABLE_NAME = "python_tool_Q&A"

# Streamlit UI setup
st.info(" Introducing **Otto**, your cutting-edge partner in streamlining dealership and customer-related operations. At EngagedAi, we specialize in harnessing the power of automation to revolutionize the way dealerships and customers interact. Our advanced solutions seamlessly handle tasks, from managing inventory and customer inquiries to optimizing sales processes, all while enhancing customer satisfaction. Discover a new era of efficiency and convenience with us as your trusted automation ally. [engagedai.io](https://funnelai.com/). For this demo application, we will use the Inventory Dataset. Please explore it [here](https://github.com/ShahVishs/workflow/blob/main/2013_Inventory.csv) to get a sense for what questions you can ask.")
# st.info(" We're developing cutting-edge conversational AI solutions tailored for automotive retail, aiming to provide advanced products and support. As part of our progress, we're establishing a environment to check offerings and also check Our website [engane.ai](https://funnelai.com/). This test application answers about Inventory, Business details, Financing and Discounts and Offers related questions. [here](https://github.com/buravelliprasad/streamlit_python_tool/blob/main/dealer_1_inventry.csv) is a inventry dataset to explore. Appointment dataset [here](https://github.com/buravelliprasad/streamlit_python_tool/blob/main/appointment_new.csv)")
# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'generated' not in st.session_state:
    st.session_state.generated = []
if 'past' not in st.session_state:
    st.session_state.past = []
# Initialize user name in session state
if 'user_name' not in st.session_state:
    st.session_state.user_name = None

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
If you're unsure of an answer, respond with 'I am sorry.'
Make every effort to assist the customer promptly while keeping responses concise, not exceeding two sentences."

Very Very Important Instruction: when ever you are using tools to answer the question. 
strictly answer only from "System:  " message provided to you.""")
details= "Today's current date is "+ todays_date +" todays week day is "+day_of_the_week+"."
class PythonInputs(BaseModel):
    query: str = Field(description="code snippet to run")
# if __name__ == "__main__":
df = pd.read_csv("appointment_new.csv")
input_templete = template.format(dhead=df.head().to_markdown(),details=details)


system_message = SystemMessage(
        content=input_templete)

prompt = OpenAIFunctionsAgent.create_prompt(
        system_message=system_message,
        extra_prompt_messages=[MessagesPlaceholder(variable_name=memory_key)]
    )

repl = PythonAstREPLTool(locals={"df": df}, name="python_repl",
        description="Use to check on available appointment times for a given date and time. The input to this tool should be a string in this format mm/dd/yy. This is the only way for you to answer questions about available appointments. This tool will reply with available times for the specified date in 24hour time, for example: 15:00 and 3pm are the same.",args_schema=PythonInputs)
tools = [tool1,repl,tool3]

agent = OpenAIFunctionsAgent(llm=llm, tools=tools, prompt=prompt)
print("this code block running every time")


if 'agent_executor' not in st.session_state:
	agent_executor = AgentExecutor(agent=agent, tools=tools, memory=memory, verbose=True, return_intermediate_steps=True)
	st.session_state.agent_executor = agent_executor
else:
	agent_executor = st.session_state.agent_executor

response_container = st.container()
container = st.container()

airtable = Airtable(AIRTABLE_BASE_ID, AIRTABLE_TABLE_NAME, api_key=airtable_api_key)

def save_chat_to_airtable(user_name, user_input, output):
    try:
        timestamp = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
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

chat_history=[]

def conversational_chat(user_input):
    result = agent_executor({"input": user_input})
    st.session_state.chat_history.append((user_input, result["output"]))
    return result["output"]

with container:
    if st.session_state.user_name is None:
        user_name = st.text_input("Your name:")
        if user_name:
            st.session_state.user_name = user_name
            
    with st.form(key='my_form', clear_on_submit=True):
        user_input = st.text_input("Query:", placeholder="Type your question here (:", key='input')
        submit_button = st.form_submit_button(label='Send')
    
    if submit_button and user_input:
       output = conversational_chat(user_input)
       
       with response_container:
           for i, (query, answer) in enumerate(st.session_state.chat_history):
               message(query, is_user=True, key=f"{i}_user", avatar_style="big-smile")
               message(answer, key=f"{i}_answer", avatar_style="thumbs")
   
           if st.session_state.user_name:
               try:
                   save_chat_to_airtable(st.session_state.user_name, user_input, output)
               except Exception as e:
                   st.error(f"An error occurred: {e}")
