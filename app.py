import streamlit as st
from crewai import Agent, Task, Crew, LLM,Process
import os
from crewai_tools import ScrapeWebsiteTool, SerperDevTool
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from io import BytesIO
import base64
from PIL import Image

# Load environment variables
load_dotenv()
# os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
os.environ["SERPER_API_KEY"] = os.getenv("SERPER_API_KEY")
# os.environ["MISTRAL_API_KEY"] = os.getenv("MISTRAL_API_KEY")



# Streamlit UI Configuration
st.set_page_config(page_title="AI Doctor Assistant", layout="wide", page_icon="🩺")

# Sidebar Navigation
with st.sidebar:
    st.title("🔍 Doctor Assistant")
    st.markdown("**Empowering doctors with AI insights.**")
    st.divider()
    st.markdown("## Features")
    st.markdown("✅ Diagnosis Recommendations")
    st.markdown("✅ Treatment Plans")
    st.markdown("✅ Download Reports")

# Main Section
st.title("🩺 AI-Powered Doctor's Assistant")
st.markdown("Get instant medical insights with AI-powered diagnosis and treatment recommendations.")
st.divider()

# Input Section
gender, age_col = st.columns(2)
with gender:
    gender = st.selectbox('Select Gender', ['Male', 'Female', 'Other'])
with age_col:
    age = st.number_input('Enter Age', min_value=0, max_value=120, value=25)

symptoms = st.text_area('Enter Symptoms', 'e.g., fever, cough, headache')
medical_history = st.text_area('Enter Medical History', 'e.g., diabetes, hypertension')

# Initialize Tools
search_tool = SerperDevTool()
scrape_tool = ScrapeWebsiteTool()

llm = LLM(model='gemini/gemini-1.5-flash',api_key=os.environ["GOOGLE_API_KEY"])

# Define Agents
diagnostician = Agent(
    role="Medical Diagnostician",
    goal="Analyze patient symptoms and medical history to provide a preliminary diagnosis.",
    backstory="This agent specializes in diagnosing medical conditions based on patient-reported symptoms and medical history. It uses advanced algorithms and medical knowledge to identify potential health issues.",
    llm=llm,
    verbose=True,
    allow_delegation=False,
    tools=[search_tool, scrape_tool],
    
)
treatment_advisor = Agent(
    role="Treatment Advisor",
    goal="Recommend appropriate treatment plans based on the diagnosis provided by the Medical Diagnostician.",
    backstory="This agent specializes in creating treatment plans tailored to individual patient needs. It considers the diagnosis, patient history, and current best practices in medicine to recommend effective treatments.",
    llm=llm,
    verbose=True,
    allow_delegation=False,
    tools=[search_tool, scrape_tool],
)

# Define Tasks
diagnose_task = Task(
    description=(
        "1. Analyze the patient's symptoms ({symptoms}) and medical history ({medical_history}).\n"
        "2. Provide a preliminary diagnosis with possible conditions based on the provided information.\n"
        "3. Limit the diagnosis to the most likely conditions."
    ),
    expected_output="A preliminary diagnosis with a list of possible conditions.",
    agent=diagnostician
)

treatment_task = Task(
    description=(
        "1. Based on the diagnosis, recommend appropriate treatment plans step by step.\n"
        "2. Consider the patient's medical history ({medical_history}) and current symptoms ({symptoms}).\n"
        "3. Provide detailed treatment recommendations, including medications, lifestyle changes, and follow-up care."
    ),
    expected_output="A comprehensive treatment plan tailored to the patient's needs.",
    agent=treatment_advisor
)



crew = Crew(
    agents=[diagnostician, treatment_advisor],
    tasks=[diagnose_task, treatment_task],
    verbose=True,
    process=Process.sequential
)


# Execution
if st.button("🧑‍⚕️ Generate Diagnosis and Treatment Plan"):
    with st.spinner('🩺 Analyzing patient data and generating recommendations...'):
        result = crew.kickoff(inputs={"symptoms": symptoms, "medical_history": medical_history})
        st.success("✅ Recommendations successfully generated!")

        st.markdown("### 📝 Results")
        st.info(result.raw)
