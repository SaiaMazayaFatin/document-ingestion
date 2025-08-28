from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from settings import settings
from models.schemas import ExtractionResult

_PROMPT = ChatPromptTemplate.from_messages([
    ("system","Extract entities (canonical+aliases) and relation triples (s,p,o) from text. Confidence 0-1. Only grounded facts."),
    ("user","Text:\n\n{chunk}\n\nReturn structured JSON.")
])

def get_extract_chain():
    llm = ChatOpenAI(model=settings.EXTRACT_LLM_MODEL, temperature=0, api_key=settings.OPENAI_API_KEY)
    return _PROMPT | llm.with_structured_output(ExtractionResult)
