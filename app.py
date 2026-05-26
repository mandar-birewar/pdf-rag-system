import os
import gradio as gr
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq

# ---------------- LOAD ENV ---------------- #

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

print("API KEY:", GROQ_API_KEY)

# ---------------- GLOBAL VARIABLES ---------------- #

vector_db = None
qa_chain = None

# ---------------- PDF PROCESSING ---------------- #

def process_pdf(pdf_file):

    global vector_db
    global qa_chain

    try:

        if pdf_file is None:
            return "Please upload a PDF first."

        print("STEP 1: Loading PDF")

        loader = PyPDFLoader(pdf_file.name)

        documents = loader.load()

        print("STEP 2: Splitting text")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        chunks = splitter.split_documents(documents)

        print("STEP 3: Creating embeddings")

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        print("STEP 4: Creating vector DB")

        vector_db = FAISS.from_documents(
            chunks,
            embeddings
        )

        print("STEP 5: Loading LLM")

        llm = ChatGroq(
            groq_api_key=GROQ_API_KEY,
            model_name="llama3-8b-8192"
        )

        print("STEP 6: Creating QA Chain")

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vector_db.as_retriever(),
            chain_type="stuff"
        )

        print("SUCCESS")

        return "PDF processed successfully."

    except Exception as e:

        print("FULL ERROR:")
        print(str(e))

        return f"FULL ERROR: {str(e)}"

# ---------------- QUESTION ANSWERING ---------------- #

def ask_question(question, history):

    global qa_chain

    if history is None:
        history = []

    if qa_chain is None:

        history.append(
            [question, "Please upload and process a PDF first."]
        )

        return history

    if question.strip() == "":
        return history

    try:

        response = qa_chain.run(question)

        history.append(
            [question, response]
        )

        return history

    except Exception as e:

        history.append(
            [question, f"Error: {str(e)}"]
        )

        return history

# ---------------- CUSTOM CSS ---------------- #

custom_css = """
body {
    background: linear-gradient(135deg, #020617, #0f172a, #111827);
}

.gradio-container {
    background: transparent !important;
    font-family: 'Poppins', sans-serif;
}

.main-title {
    text-align: center;
    font-size: 60px;
    font-weight: 900;
    background: linear-gradient(90deg, #38bdf8, #8b5cf6, #ec4899);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-top: 10px;
}

.sub-title {
    text-align: center;
    color: #d1d5db;
    font-size: 20px;
    margin-bottom: 30px;
}

.glass-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    backdrop-filter: blur(18px);
    border-radius: 25px;
    padding: 25px;
    box-shadow: 0 0 40px rgba(0,0,0,0.5);
}

button {
    background: linear-gradient(90deg,#06b6d4,#8b5cf6) !important;
    color: white !important;
    border: none !important;
    border-radius: 18px !important;
    font-weight: bold !important;
    transition: 0.3s ease;
}

button:hover {
    transform: scale(1.03);
}

footer {
    visibility: hidden;
}
"""

# ---------------- UI ---------------- #

with gr.Blocks(
    css=custom_css,
    theme=gr.themes.Soft()
) as demo:

    gr.HTML(
        """
        <div class="glass-card">
            <div class="main-title">
                PDF Intelligence Engine
            </div>

            <div class="sub-title">
                AI Powered Retrieval-Augmented Generation System
            </div>
        </div>
        """
    )

    with gr.Row():

        # LEFT SIDE

        with gr.Column(scale=1):

            gr.HTML(
                """
                <div class="glass-card">
                    <h2 style="color:white;text-align:center;">
                        Upload Document
                    </h2>
                </div>
                """
            )

            pdf_input = gr.File(
                label="Upload PDF",
                file_types=[".pdf"]
            )

            process_btn = gr.Button(
                "Process PDF"
            )

            process_output = gr.Textbox(
                label="System Status",
                interactive=False
            )

        # RIGHT SIDE

        with gr.Column(scale=2):

            gr.HTML(
                """
                <div class="glass-card">
                    <h2 style="color:white;text-align:center;">
                        AI Research Assistant
                    </h2>
                </div>
                """
            )

            chatbot = gr.Chatbot(
                height=500,
                label="Chat Window"
            )

            question_input = gr.Textbox(
                placeholder="Ask anything from your PDF...",
                lines=2
            )

            ask_btn = gr.Button(
                "Generate Answer"
            )

    # ---------------- BUTTON ACTIONS ---------------- #

    process_btn.click(
        fn=process_pdf,
        inputs=pdf_input,
        outputs=process_output
    )

    ask_btn.click(
        fn=ask_question,
        inputs=[question_input, chatbot],
        outputs=chatbot
    )

# ---------------- LAUNCH ---------------- #

if __name__ == "__main__":

    demo.launch(
        server_name="127.0.0.1",
        server_port=7860
    )