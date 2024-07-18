from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from ecommbot.ingest import ingest_data
from transformers import pipeline

def generation(vstore):
    retriever = vstore.as_retriever(search_kwargs={"k": 3})

    PRODUCT_BOT_TEMPLATE = """
    Your ecommercebot bot is an expert in product recommendations and customer queries.
    It analyzes product titles and reviews to provide accurate and helpful responses.
    Ensure your answers are relevant to the product context and refrain from straying off-topic.
    Your responses should be concise and informative.

    CONTEXT:
    {context}

    QUESTION: {question}

    YOUR ANSWER:
    """


    prompt = ChatPromptTemplate.from_template(PRODUCT_BOT_TEMPLATE)

    # Load a Hugging Face model for text generation
    model_name = "gpt2"  # GPT-2 model from Hugging Face
    generator = pipeline("text-generation", model=model_name)

    

    # Use the Hugging Face generator instead of the OpenAI model
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | generator
        | StrOutputParser()
    )

    return chain

if __name__=='__main__':
    vstore = ingest_data("done")
    chain  = generation(vstore)
    print(chain.invoke("can you tell me the best bluetooth buds?"))
