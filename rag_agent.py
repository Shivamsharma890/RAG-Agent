from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader, PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_community.tools import tool
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver       ## for provide memory to our agent--
import streamlit as st


st.subheader("🤖⚡Welcome in Super Intelligence RAG Agentic System")


#### Data in st session
if "document_uploaded" not in st.session_state:
    st.session_state.document_uploaded = False
    
if "agent" not in st.session_state:
    st.session_state.agent = None
    
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
    
if "messages" not in st.session_state:
    st.session_state.messages = []


def document_process(path):
    
    ## load the Document :-
    loader = PyPDFDirectoryLoader(path)
    docs_load = loader.load()


    ## split document into chunks :-
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs_chunk = splitter.split_documents(documents=docs_load)


    ## convert chunks into embeddings :-
    Embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")


    ## store Embeddings into vector store :-
    vector_Database = InMemoryVectorStore.from_documents(
        documents=docs_chunk, 
        embedding=Embeddings
    )


    ## create an agent for tool, llm, prompt
    llm = ChatGroq(model="openai/gpt-oss-20b")


    @tool
    def retriever_tool(query: str):
        """
        This tool can help you to retrieve the relevant data from knowledge base.
        """
        docs = vector_Database.similarity_search(query=query, k=3)
        
        context = ""
        for doc in docs:
            context = doc.page_content + "\n\n"

        return context


    system_prompt = """
        you are a helpful assistant that answer question using retrieved context.
        ALWAYS use the `retriever_tool` for question requiring external knowledge.
    """

    agent_memory = InMemorySaver()

    
    rag_agent = create_agent(
        model=llm,
        tools=[retriever_tool],
        system_prompt=system_prompt,
        checkpointer=agent_memory,
    )
    
    
    st.session_state.agent = rag_agent
    st.session_state.document_uploaded = True

### Upload UI --
if not st.session_state.document_uploaded:
    uploaded = st.file_uploader(label = "Select PDF Files", type = ["PDF"], accept_multiple_files = True)
    if uploaded:
        with st.spinner("Processing...."):
            path = "./doc_files/"
            for file in uploaded:
                with open(path + file.name, "wb") as f:
                    f.write(file.getvalue())
                    
            document_process(path)
            st.rerun()


## Chat UI --
if st.session_state.document_uploaded and st.session_state.agent:
    for message in st.session_state.messages:
        role = message["role"]
        content = message["content"]
        st.chat_message(role).markdown(content)
    
    
    query = st.chat_input("Ask me anything related to uploaded Documents.....")
    if query:
        st.session_state.messages.append({"role" : "user", "content" : query})
        st.chat_message("user").markdown(query)
        res = st.session_state.agent.invoke(
            {"messages" : [{"role" : "user", "content" : query}]},
            {"configurable" : {"thread_id" : 1}}
        )
        
        result = res["messages"][-1].content
        st.chat_message("ai").markdown(result)
        st.session_state.messages.append({"role" : "ai", "content" : result})
        
