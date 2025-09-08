
import os
from dotenv import load_dotenv

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

load_dotenv()
if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY missing in .env file")

from langchain_community.document_loaders import JSONLoader

def metadata_func(record, metadata):
    return {
        "title": record.get("title", ""),
        "codeblocks": record.get("codeblocks", [])
    }

loader = JSONLoader(
    file_path="projects.jsonl",
    jq_schema=".",
    content_key="content",
    metadata_func=metadata_func,
    text_content=False,
    json_lines=True
)

raw_documents = loader.load()

for doc in raw_documents:
    if doc.metadata.get("codeblocks"):
        doc.page_content += "\n\nCode Examples:\n" + "\n".join(doc.metadata["codeblocks"])



text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
documents = text_splitter.split_documents(raw_documents)

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
db = FAISS.from_documents(documents, embeddings)

llm = GoogleGenerativeAI(model="gemini-2.5-pro")


response_schemas = [
    ResponseSchema(name="theory", description="Beginner-friendly explanation of the project and why it works."),
    ResponseSchema(name="steps", description="Step-by-step build guide with clear instructions."),
    ResponseSchema(name="pinouts", description="Pin connections between Arduino and modules."),
    ResponseSchema(name="code", description="Arduino/C++ code snippets with comments."),
    ResponseSchema(name="troubleshooting", description="Common issues and fixes for students."),
]

output_parser = StructuredOutputParser.from_response_schemas(response_schemas)



prompt = ChatPromptTemplate.from_template("""
You are an expert Arduino mentor. Your job is to help students build real Arduino projects step-by-step.

<context>
{context}
</context>

Question: {input}

Return your answer strictly in JSON with these keys:
{format_instructions}

Rules:
- Make it **extremely detailed** (like a full project guide).
- Always include **step-by-step wiring instructions** and a **pin mapping table**.
- Provide **complete Arduino code** with comments.
- Add **troubleshooting tips** to help beginners.
- Do not skip any section, even if you have to infer from general Arduino knowledge.
""").partial(format_instructions=output_parser.get_format_instructions())


document_chain = create_stuff_documents_chain(llm, prompt)
retriever = db.as_retriever(search_kwargs={"k": 5})
retrieval_chain = create_retrieval_chain(retriever, document_chain)

query = "How to build Arduino Location Tracker using SIM800L GSM Module and NEO-6M GPS Module with pinouts, code, and troubleshooting?"
response = retrieval_chain.invoke({"input": query})

try:
    parsed = output_parser.parse(response["answer"])
    print("Structured Response:\n")
    for k, v in parsed.items():
        print(f"\n--- {k.upper()} ---\n{v}\n")
except Exception:
    print("Parsing failed, raw output:\n")
    print(response["answer"])
