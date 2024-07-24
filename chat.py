import os
from dotenv import load_dotenv
from langchain import hub
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough


load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")


def response(user_query, chat_history):
    llm = ChatOpenAI(model_name="gpt-3.5-turbo",
                     temperature=0.9,
                     openai_api_key=openai_api_key,
                     max_tokens=256)

    
    template = """You are a personal assistant. Greet the patient with a standard way of greeting and tell them you are here to help.
    Do conversation like a pro in the field in which the user is sending you a message. Do not send paragraphs, play in an interactive tone.
    Take into account the {chat_history} through {ConversationBufferMemory} to answer with respect to the previous messages.
    {context}

    Interactive Answer: {answer}
    """

    custom_rag_prompt = PromptTemplate.from_template(template)

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    
    memory = ConversationBufferMemory()

    
    for message in chat_history:
        memory.add_message(message)

    
    conversational_chain = ConversationChain(memory=memory, prompt=custom_rag_prompt, llm=llm)

    response = conversational_chain.run(user_query)
    chat_history.append(HumanMessage(content=user_query))
    chat_history.append(AIMessage(content=response))

    return response, chat_history 
