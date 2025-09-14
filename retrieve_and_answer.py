from pinecone_client import index
from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")


def retrieve_chunks(query: str, top_k: int = 3):
    query_vector = embeddings_model.embed_query(query)
    results = index.query(vector=query_vector, top_k=top_k, include_metadata=True)
    retrieved_text = "\n\n".join(match['metadata']['text'] for match in results['matches'])
    return retrieved_text

def answer_question(query: str) -> str:
    context_text = retrieve_chunks(query, top_k=3)
    prompt = PromptTemplate(
        template="""
You are a helpful assistant.
Answer ONLY from the provided transcript context.
If the context is insufficient, just say you don't know.
{context}
Question: {question}
""",
        input_variables=['context', 'question']
    )
    final_prompt = prompt.invoke({"context": context_text, "question": query})
    answer = llm.invoke(final_prompt)
    return answer.content


if __name__ == "__main__":
    question = "Write Lee Kuan Yew's strategy in two short points"
    response = answer_question(question)
    print("Answer:\n", response)
    
    