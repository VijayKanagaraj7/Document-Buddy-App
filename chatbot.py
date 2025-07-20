# chatbot.py

import os
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
import streamlit as st

class ChatbotManager:
    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en",
        device: str = "cpu",
        encode_kwargs: dict = {"normalize_embeddings": True},
        llm_model: str = "gemini-2.5-flash", 
        llm_temperature: float = 0.7,
        persist_directory: str = "db",
    ):
        """
        Initializes the ChatbotManager with embedding models, LLM, and vector store.

        Args:
            model_name (str): The HuggingFace model name for embeddings.
            device (str): The device to run the model on ('cpu' or 'cuda').
            encode_kwargs (dict): Additional keyword arguments for encoding.
            llm_model (str): The Google Generative AI model name (e.g., "gemini-2.5-flash").
            llm_temperature (float): Temperature setting for the LLM.
            persist_directory (str): The directory to persist ChromaDB.
        """
        self.model_name = model_name
        self.device = device
        self.encode_kwargs = encode_kwargs
        self.llm_model = llm_model
        self.llm_temperature = llm_temperature
        self.persist_directory = persist_directory

        # Initialize Embeddings
        self.embeddings = HuggingFaceBgeEmbeddings(
            model_name=self.model_name,
            model_kwargs={"device": self.device},
            encode_kwargs=self.encode_kwargs,
        )

        # Initialize Google Generative AI LLM
        # Directly using the provided API key. For production, consider environment variables for security.
        self.llm = ChatGoogleGenerativeAI(
            model=self.llm_model,
            temperature=self.llm_temperature,
            google_api_key="AIzaSyC7wR8jHxaJ-NFz3DnQ9yO_qiKkflpw4E0"
        )

        # Define the prompt template
        self.prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer. Answer must be detailed and well explained.
Helpful answer:
"""

        # Initialize the Chroma vector store
        self.db = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings
        )

        # Initialize the prompt
        self.prompt = PromptTemplate(
            template=self.prompt_template,
            input_variables=['context', 'question']
        )

        # Initialize the retriever
        self.retriever = self.db.as_retriever(search_kwargs={"k": 1})

        # Define chain type kwargs
        self.chain_type_kwargs = {"prompt": self.prompt}

        # Initialize the RetrievalQA chain with return_source_documents=False
        self.qa = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=False,  # Set to False to return only 'result'
            chain_type_kwargs=self.chain_type_kwargs,
            verbose=False
        )

    def get_response(self, query: str) -> str:
        """
        Processes the user's query and returns the chatbot's response.

        Args:
            query (str): The user's input question.

        Returns:
            str: The chatbot's response.
        """
        try:
            response = self.qa.run(query)
            return response  # 'response' is now a string containing only the 'result'
        except Exception as e:
            st.error(f"⚠️ An error occurred while processing your request: {e}")
            return "⚠️ Sorry, I couldn't process your request at the moment."
