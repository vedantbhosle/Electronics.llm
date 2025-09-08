import os
import langchain
from dotenv import load_dotenv

from langchain_community.document_loaders import JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.cache import InMemoryCache
from langchain.schema import Document


load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    raise ValueError("Missing GOOGLE_API_KEY in .env file")

langchain.llm_cache = InMemoryCache()


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


documents = []
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

for doc in raw_documents:
    # Split long content into chunks
    content_chunks = text_splitter.split_text(doc.page_content)
    for chunk in content_chunks:
        documents.append(Document(
            page_content=chunk,
            metadata={"title": doc.metadata.get("title", ""), "type": "content"}
        ))

    # Each code block = one full doc (don’t split code)
    for idx, code in enumerate(doc.metadata.get("codeblocks", [])):
        documents.append(Document(
            page_content=code,
            metadata={"title": doc.metadata.get("title", ""), "type": f"codeblock_{idx+1}"}
        ))


embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
db = FAISS.from_documents(documents, embeddings)


llm_rerank = Ollama(model="llama3.1:8b ")
compressor = LLMChainExtractor.from_llm(llm_rerank)

retriever = ContextualCompressionRetriever(
    base_retriever=db.as_retriever(search_kwargs={"k": 5}),
    base_compressor=compressor
)


response_schemas = [
    ResponseSchema(name="steps", description="Step-by-step instructions to build the project"),
    ResponseSchema(name="pinouts", description="Pinout mapping for components"),
    ResponseSchema(name="code", description="Relevant Arduino/C++ code snippets"),
    ResponseSchema(name="sources", description="List of project titles used in the answer"),
]

output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

prompt = ChatPromptTemplate.from_template(
    """
You are an expert mentor in Arduino and beginner-level electronics projects.
Your role is to help students build confidence in making Arduino projects.

<context>
{context}
</context>

When answering:
- Use simple step-by-step explanations.
- Suggest low-cost components.
- Provide relevant pinouts and code examples.
- Include the project titles (from metadata) you used.

Question: {input}

Return your answer strictly in JSON with the following keys:
{format_instructions}
"""
).partial(format_instructions=output_parser.get_format_instructions())



document_chain = create_stuff_documents_chain(Ollama(model="qwen3:32b"), prompt)
retrieval_chain = create_retrieval_chain(retriever, document_chain)


query = "How to build a Wearable Air Quality Monitor Pendant using STM32 & SGP40 sensor?"
response = retrieval_chain.invoke({"input": query})


try:
    parsed = output_parser.parse(response["answer"])
    print(" Structured Response:\n")
    print(parsed)
except Exception:
    print(" Parsing failed, raw output:\n")
    print(response["answer"])
