import os

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

def load_llm():
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.5,
        max_tokens=512,
        groq_api_key=GROQ_API_KEY
    )
    return llm

CUSTOM_PROMPT_TEMPLATE = """
            Use the pieces of information provided in the context to answer the user's question in detail.

            Instructions:
            - Answer in 4-6 clear, well-structured sentences or bullet points.
            - Include relevant clinical details, causes, symptoms, or treatment steps if present in the context.
            - If the answer is not in the context, say "I don't know" — do not make up information.
            - Do not include anything outside the given context.

            Context: {context}
            Question: {question}

            Answer in detail:
        """

def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(
        template=custom_prompt_template,
        input_variables=["context", "question"]
    )
    return prompt

# Load Database
DB_FAISS_PATH = "vectorstore/db_faiss"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# Create QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=load_llm(),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={'k': 10}),
    return_source_documents=True,
    chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
)

# Now invoke with a single query
user_query = input("Write Query Here: ")
response = qa_chain.invoke({'query': user_query})
print("RESULT: ", response["result"])
print("SOURCE DOCUMENTS: ", response["source_documents"])