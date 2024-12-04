import os
import json

from mpmath import extend


def load_and_process_data(base_dir, output_dir):
    """
    모든 하위 폴더의 JSON 파일을 읽어 연령대별로 전처리하여 저장하는 함수
    :param base_dir: 최상위 데이터 폴더 경로
    :param output_dir: 전처리된 데이터를 저장할 폴더 경로
    """
    categories = {
        '유아': [],
        '초등_저학년': []
    }

    # 모든 하위 폴더 탐색
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)

                # 파일 경로에 연령대 키워드가 포함되어 있는지 확인
                for category in categories.keys():
                    if category in root:
                        # JSON 파일 읽기
                        with open(file_path, 'r', encoding='utf-8') as f:
                            try:
                                data = json.load(f)
                                title = data.get('title', '제목 없음')
                                readAge = data.get('readAge', [])
                                paragraph_info = data.get('paragraphInfo', [])
                                content = ' '.join([paragraph.get('srcText', '') for paragraph in paragraph_info])

                                categories[category].append({
                                    'title': title.strip(),
                                    'readAge': readAge,
                                    'description': content.strip()
                                })

                            except json.JSONDecodeError as e:
                                print(f"JSON 디코딩 오류: {e}, 파일: {file_path}")

    # 각 범주별로 데이터를 저장
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"동화학습데이터.json")

    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            try:
                existing_data = json.load(f)
            except json.JSONDecodeError:
                existing_data = []
    else:
        existing_data = []

    existing_titles = {item['title'] for item in existing_data}
    new_data = [item for item in categories[category] if item['title'] not in existing_titles]
    combined_data = existing_data + new_data

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(combined_data, f, ensure_ascii=False, indent=4)

    print(f"{output_file}에 {len(combined_data)}개의 동화 데이터가 저장되었습니다.")
    print(f"새로 추가된 데이터: {len(new_data)}개")

# 실행 경로 설정
base_dir = '../data/raw'  # 원본 데이터 폴더 경로
output_dir = '../data/processed'  # 전처리된 데이터를 저장할 폴더 경로

# 데이터 로드 및 저장
if __name__ == "__main__":
    load_and_process_data(base_dir, output_dir)
