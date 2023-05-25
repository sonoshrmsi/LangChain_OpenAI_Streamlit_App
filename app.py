import streamlit as st
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain. vectorstores import ElasticVectorSearch, Pinecone, Weaviate, FAISS

from pdf2image import convert_from_bytes
from dotenv import load_dotenv
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI

# ENV variables
load_dotenv()
# Langchain
chain = load_qa_chain(OpenAI(), chain_type='stuff')

# Application
# sessions
if 'selected_doc' not in st.session_state:
    st.session_state['selected_doc'] = 'A Prompt Pattern Catalog to Enhance Prompt Engineering with ChatGPT'

# Title
st.title('Frontend Lighting Talks Demo')

# A brief information
st.write("""Large Language Models (LLMs) are highly useful in providing guidance for applications documentation. With their extensive language understanding and knowledge base, LLMs can analyze and interpret application requirements, offering insights and suggestions to improve documentation quality. They can assist in clarifying technical jargon, ensuring consistency in language and formatting, and even providing examples and templates for different sections of the documentation. LLMs enable developers and technical writers to streamline the application process, saving time and effort, and enhancing the overall user experience. By leveraging LLMs, organizations can ensure that their applications are accompanied by comprehensive and well-crafted documentation.""")

# Uploading the PDF file
# Select documenation
st.subheader("Let's play with the LLM")

st.session_state.selected_doc = st.selectbox("Please select the article:", ("streamlit app doc",
                                                                            "Data Leakage",
                                                                            "control engineering",
                                                                            "React"))

# updating the session for the selected pdf
pdf_path = "./docs/" + st.session_state.selected_doc + ".pdf"

# Create a PdfReader object to read the PDF file
pdf_reader = PdfReader(pdf_path)

# Prmopting the LLMs
st.subheader("Prompting")
prompt = st.text_input('Ask your question here')

# if the pdf is uploaded
if pdf_reader:
    reader = pdf_reader
    # reader = PdfReader(pdf)
    raw_text = ''
    # Looping through the pdf
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            raw_text += text

    # Splitting the text into chunks
    text_splitter = CharacterTextSplitter(
        separator='\n',
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    text = text_splitter.split_text(raw_text)
    # Embedding the texts
    embeddings = OpenAIEmbeddings()
    docsearch = FAISS.from_texts(text, embeddings)

# LLM creativity
llm = OpenAI(temperature=0.5)

# Prompting the PDF
if pdf_reader:
    if prompt:
        # prompt = prompt + " And at least provide 200 words"
        search = docsearch.similarity_search(prompt)
        answer = chain.run(input_documents=search, question=prompt)
        # response = llm(prompt)
        if answer == "I don't know.":
            st.write(
                "I need mnore information or the requested information is availabale to me!")
        else:
            st.header("from the book/article")
            st.write(answer)
            # # st.write(response)
            # st.header("from text-davinci")
            # # Prompting the language model for the examples
            # example_answer = llm("provide an example of this:" + answer)
            # # providing code if the python book is selected
            # if st.session_state.selected_doc == "Python book":
            #     st.write(example_answer)
            #     st.code(example_answer)

            # st.write(example_answer)
