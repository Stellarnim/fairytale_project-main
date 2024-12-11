import os
import json
import glob
import logging
import threading
import time
import torch
import asyncio
from deep_translator import GoogleTranslator
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from diffusers import DiffusionPipeline
from huggingface_hub import login
from tts import generate_and_play_audio

# 싱글톤 관련 글로벌 변수
llm = None
db = None
singleton_lock = threading.Lock()
is_initialized = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()  # 콘솔 출력 추가
    ]
)

def load_environment():
    import os
    load_dotenv('../.env')
    if not os.getenv("OPENAI_API_KEY") or not os.getenv("HUGGINGFACE_API_KEY"):
        logging.error("API 키가 누락되었습니다.")
        raise EnvironmentError("Missing API keys in .env file.")

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
        raise FileNotFoundError(f"Cannot find the file: {file_path}")
    except json.JSONDecodeError:
        logging.error(f"JSON 파일을 디코딩하는 중 오류가 발생했습니다: {file_path}")
        raise ValueError(f"Invalid JSON format in file: {file_path}")


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
    if torch.cuda.is_available():
        device = 'cuda'  # GPU 사용
    else:
        device = 'cpu'  # GPU가 없는 경우 CPU 사용

    logging.info("임베딩 모델 생성 중... device: {}".format(device))
    model_name = "jhgan/ko-sroberta-multitask"
    model_kwargs = {'device': device}
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

    logging.debug(f"문서 개수: {len(docs)}")
    try:
        db = Chroma.from_documents(docs, model, persist_directory=save_directory)
        logging.info("벡터 데이터베이스 생성 완료...")
        return db
    except Exception as e:
        logging.error(f"벡터 데이터베이스 생성 중 오류 발생: {e}")
        raise


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


def load_lists(socketio):
    folder_path = './gen_stories'

    json_files = glob.glob(os.path.join(folder_path, "*.json"))
    sorted_files = sorted(json_files, key=os.path.getctime)

    file_names = [os.path.splitext(os.path.basename(sorted_file))[0] for sorted_file in sorted_files]

    socketio.emit('load_lists', {"file_names": file_names})


def load_fairytale(title, socketio):
    folder_path = './gen_stories'
    file_path = os.path.join(folder_path, title + '.json')

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

        print(f"Loaded JSON: {data}")  # 디버깅 출력

        # 'description' 필드를 읽어와서 스토리 부분으로 사용
        story_parts = data.get('description', ['내용 없음'])  # description이 없으면 기본값
        if not isinstance(story_parts, list):
            story_parts = [str(story_parts)]  # 리스트가 아닌 경우 리스트로 변환

        print(f"Story parts: {story_parts}")  # 디버깅 출력
        socketio.emit("load_fairytale", {"story_title": title, "story_parts": story_parts})

    except FileNotFoundError:
        print(f"File not found: {file_path}")
        socketio.emit("load_fairytale_error", {"error": "File not found"})
    except json.JSONDecodeError:
        print(f"Error decoding JSON: {file_path}")
        socketio.emit("load_fairytale_error", {"error": "Invalid JSON format"})

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

    story_title = None
    story_parts = []

    try:
        response = llm.invoke([{"role": "system", "content": prompt_text}])

        contents = response.content.split("\n\n")

        if contents:
            story_title = contents[0].replace("제목: ", "").replace('\"', "").strip()

        story_parts = [line.strip() for line in contents[1:] if line.strip()]

        logging.info("동화 생성 완료.")

        output_dir = "gen_stories"
        os.makedirs(output_dir, exist_ok=True)

        file_path = os.path.join(output_dir, story_title + ".json")

        story_data = {
            "title": story_title,
            "readAge": readage,
            "description": story_parts
        }

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(story_data, f, ensure_ascii=False, indent=4)

        logging.info("동화가 " + str(story_title) + ".json에 저장되었습니다.")

        logging.info("이미지와 TTS 생성중...")
        asyncio.run(generate_and_play_audio(story_title, story_parts))
        generate_illustrations_from_story(story_title, story_parts)


        socketio.emit('generate_story', {"story_title": story_title, "story_parts": story_parts})

    except Exception as e:
        logging.error(f"동화 생성 중 오류가 발생했습니다: {e}")
        socketio.emit('story_error')
        return


def generate_illustrations_from_story(story_title, story_parts):
    """
    각 문단마다 품질 높은 삽화를 생성하는 함수
    """
    os.makedirs("illustrations", exist_ok=True)  # 삽화를 저장할 디렉토리 생성

    hf_api_key = os.getenv("HUGGINGFACE_API_KEY")
    if not hf_api_key:
        logging.error("HUGGINGFACE_API_KEY 환경 변수를 설정해야 합니다.")
        return

    try:
        login(token=hf_api_key)
    except Exception as e:
        logging.error(f"Hugging Face 로그인 중 오류 발생: {e}")
        return

    try:
        pipe = DiffusionPipeline.from_pretrained("black-forest-labs/FLUX.1-dev")
        pipe.load_lora_weights("Shakker-Labs/FLUX.1-dev-LoRA-One-Click-Creative-Template")
        pipe.to("cuda" if torch.cuda.is_available() else "cpu")
    except Exception as e:
        logging.error(f"모델 로드 중 오류 발생: {e}")
        return

    translated_paragraphs = []  # 번역된 문단 저장

    try:
        translated_title = GoogleTranslator(source="auto", target="en").translate(story_title)
        translated_paragraphs.append(translated_title)

        for part in story_parts:
            translated_paragraph = GoogleTranslator(source='ko', target='en').translate(part)
            translated_paragraphs.append(translated_paragraph)

    except Exception as e:
        logging.error(f"번역 중 오류 발생: {e}")
        return

    for i, paragraph in enumerate(translated_paragraphs, start=1):
        # 각 문단을 기반으로 한 삽화 생성 프롬프트 설정
        prompt = (
            f"Create a charming, detailed children's storybook illustration for the following scene. "
            f"The scene should be consistent with the overall fairy tale, focusing on creating a warm, friendly, and magical atmosphere. "
            f"Scene: {paragraph} "
            "Illustrate the emotions and interactions between the characters to reflect the story's narrative. "
            "Use soft pastel colors, gentle lighting, and simple yet inviting backgrounds, such as nature elements like clouds, trees, and rainbows. "
            "Ensure a cohesive storybook style across all illustrations. "
            "Do not include text in the image. Focus on visually telling the story through expressions and details that children can easily understand and connect with."
        )

        print(f"Generating illustration {i} with prompt: {prompt}") # 삽화 생성 시작 메시지
        try:
            image = pipe(prompt).images[0]
            image_path = f"../static/illustrations/{story_title}_{i}.png"
            image.save(image_path)
            print(f"Generated illustration saved as {image_path}")
        except Exception as e:
            print(f"Illustration generation error for part {i}: {e}")  # 예외 발생 시 에러 메시지 출력

        time.sleep(1)  # 다음 요청 전 대기