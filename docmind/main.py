from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from groq import Groq
from dotenv import load_dotenv
import os
import glob

load_dotenv()

DATA_DIR = "docmind/data"


def find_pdfs():
    pdfs = glob.glob(os.path.join(DATA_DIR, "*.pdf"))
    return pdfs


def select_pdf():
    pdfs = find_pdfs()

    if not pdfs:
        print(f"No PDFs found in {DATA_DIR}/. Drop a PDF there and try again.")
        exit(1)

    if len(pdfs) == 1:
        print(f"Found: {os.path.basename(pdfs[0])}")
        return pdfs[0]

    print("\nAvailable PDFs:")
    for i, pdf in enumerate(pdfs):
        print(f"  [{i + 1}] {os.path.basename(pdf)}")

    while True:
        choice = input("\nSelect a PDF (number): ")
        if choice.isdigit() and 1 <= int(choice) <= len(pdfs):
            return pdfs[int(choice) - 1]
        print("Invalid choice. Try again.")


def load_and_chunk(pdf_path):
    print(f"\nLoading: {os.path.basename(pdf_path)}")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"Loaded {len(documents)} pages")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")
    return chunks


def create_vector_store(chunks):
    print("Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vector_store = FAISS.from_documents(chunks, embeddings)
    print("Vector store ready!")
    return vector_store


def ask(question, vector_store, client, all_chunks):
    # Semantic search results
    relevant_chunks = vector_store.max_marginal_relevance_search(
        question, k=4, fetch_k=10
    )

    # Always inject page 1 chunks (abstract/intro)
    page_1_chunks = [c for c in all_chunks if c.metadata.get("page", -1) == 0]

    # Combine — page 1 first, then semantic results, remove duplicates
    seen = set()
    combined = []
    for chunk in page_1_chunks + relevant_chunks:
        if chunk.page_content not in seen:
            seen.add(chunk.page_content)
            combined.append(chunk)

    print("\n--- Retrieved chunks ---")
    for i, chunk in enumerate(combined):
        print(f"\nChunk {i + 1} (Page {chunk.metadata.get('page', 'unknown') + 1}):")
        print(chunk.page_content)
        print("---")

    context = "\n\n".join([chunk.page_content for chunk in combined])

    sources = sorted(
        list(
            set(
                [
                    f"Page {chunk.metadata.get('page', 'unknown') + 1}"
                    for chunk in combined
                ]
            )
        )
    )

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": """You are a research paper assistant. 
                Answer questions using ONLY the provided context from the paper.
                If the answer is not in the context, say 'This information is not in the provided context.'
                Always be precise and cite specific details from the context.""",
            },
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
    )

    answer = response.choices[0].message.content
    return answer, sources


if __name__ == "__main__":
    pdf_path = select_pdf()
    chunks = load_and_chunk(pdf_path)
    vector_store = create_vector_store(chunks)
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    print("\nDocMind ready! Ask questions about your paper.")
    print("Type 'quit' to exit\n")

    while True:
        question = input("Your question: ")
        if question.lower() == "quit":
            print("Goodbye!")
            break

        answer, sources = ask(question, vector_store, client, chunks)
        print(f"\nAnswer: {answer}")
        print(f"Sources: {', '.join(sources)}\n")
        print("-" * 50)
