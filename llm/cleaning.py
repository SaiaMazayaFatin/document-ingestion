from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from settings import settings

_PROMPT = ChatPromptTemplate.from_messages([
    ("system","You clean ASR text. Remove fillers, fix casing/punctuation, keep meaning; no hallucinations."),
    ("user","Raw transcript:\n\n{raw}\n\nReturn the cleaned transcript.")
])

def get_clean_chain():
    llm = ChatOpenAI(model=settings.CLEAN_LLM_MODEL, temperature=0, api_key=settings.OPENAI_API_KEY)
    return _PROMPT | llm
