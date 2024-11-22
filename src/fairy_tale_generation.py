import os
import json
import logging
import threading
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document

# 싱글톤 관련 글로벌 변수
llm = None
db = None
singleton_lock = threading.Lock()
is_initialized = False

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("app.log")]
)


def load_environment():
    try:
        load_dotenv('../.env')
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("API 키를 찾을 수 없습니다. .env 파일을 확인해 주세요.")
    except Exception as e:
        logging.error(f"환경 변수 로드 중 오류 발생: {e}")
        exit()


def load_processed_data(category):
    logging.info("데이터를 로드합니다...")
    file_path = f"../data/processed/{category}.json"
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        logging.info("데이터 로드 완료...")
        return data
    except FileNotFoundError:
        logging.error(f"파일을 찾을 수 없습니다: {file_path}")
        exit()
    except json.JSONDecodeError:
        logging.error(f"JSON 파일을 디코딩하는 중 오류가 발생했습니다: {file_path}")
        exit()


def split_text(data, chunk_size=2000, chunk_overlap=500):
    logging.info("데이터 분할 중...")
    rc_text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        separators=["\n\n", "\n", " "],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        encoding_name="o200k_base",
        model_name="gpt-4o"
    )
    texts = [Document(page_content=story.get("description", "")) for story in data]
    text_documents = rc_text_splitter.split_documents(texts)
    logging.info("데이터 분할 완료...")
    return text_documents


def create_embedding_model():
    logging.info("임베딩 모델 생성 중...")
    model_name = "jhgan/ko-sroberta-multitask"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': True, 'clean_up_tokenization_spaces': False}
    embedding_model = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    logging.info("임베딩 모델 생성 완료...")
    return embedding_model


def embed_documents(docs, model, save_directory="./chroma_db"):
    logging.info("벡터 데이터베이스 생성 중...")
    import shutil
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    if os.path.exists(save_directory):
        shutil.rmtree(save_directory)

    db = Chroma.from_documents(docs, model, persist_directory=save_directory)
    logging.info("벡터 데이터베이스 생성 완료...")
    return db


def create_llm(model_name="gpt-4", max_tokens=1500, temperature=0.7):
    logging.info("LLM을 생성 중…")
    llm = ChatOpenAI(
        model_name=model_name,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        temperature=temperature,
        max_tokens=max_tokens,
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()]
    )
    logging.info("LLM 생성 완료…")
    return llm


def gen_run():
    global llm, db, is_initialized

    with singleton_lock:
        if is_initialized:
            return llm, db

        # 환경 설정 로드
        load_environment()

        # 데이터 로드 및 처리
        data = load_processed_data("동화학습데이터")
        chunks = split_text(data)

        # 임베딩 모델 및 벡터 데이터베이스 생성
        embedding_model = create_embedding_model()
        db = embed_documents(chunks, embedding_model)

        # LLM 생성
        llm = create_llm()

        is_initialized = True
    return llm, db


def generate_story(keywords, readage, socketio):
    query = ", ".join(keywords)

    try:
        retriever = db.as_retriever(
            search_type="mmr",
            search_kwargs={'k': 2, 'fetch_k': 3}
        )
        context_docs = retriever.invoke(query)
    except Exception as e:
        logging.error(f"문서 검색 중 오류가 발생했습니다: {e}")
        return

    context_text = "\n".join(doc.page_content for doc in context_docs)
    max_context_length = 1000
    if len(context_text) > max_context_length:
        context_text = context_text[:max_context_length] + "..."

    prompt_text = f"""
    당신은 어린이를 위한 동화를 만드는 AI입니다.
    주어진 키워드를 모두 포함하여 교훈적이고 자연스러운 이야기를 간결하게 작성하세요.
    대상 연령층에 어울리는 동화로 작성해주세요.
    동화는 다음 요소를 포함해야 합니다:
    - 이야기의 주제와 관련된 교훈을 중심으로 간결하게 진행합니다.
    - 아이들이 공감할 수 있는 간단한 대화와 상황을 담아 주세요.
    - 부적절한 소리 묘사나 과도하게 복잡한 내용은 피하고, 교육적이면서 흥미로운 내용을 담아 주세요.

    제목을 포함하여 동화 전체 내용을 간결하게 작성해 주세요.
    
    대상 연령층 : {readage}
    키워드: {keywords}
    참고 자료:
    {context_text}
    """

    try:
        response = llm.invoke([{"role": "system", "content": prompt_text}])
        story = response.content
        logging.info("동화 생성 완료.")
        socketio.emit('generate_story', story)
        return story
    except Exception as e:
        logging.error(f"동화 생성 중 오류가 발생했습니다: {e}")
        return
