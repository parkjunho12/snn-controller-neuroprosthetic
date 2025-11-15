import pandas as pd
import scipy.io
import os
import numpy as np

def convert_mat_to_csv(mat_file_path, output_dir):
    """
    하나의 .mat 파일을 로드하여 그 안의 각 변수를 별개의 CSV 파일로 저장합니다.

    :param mat_file_path: 변환할 .mat 파일의 경로
    :param output_dir: CSV 파일을 저장할 폴더 경로
    """
    try:
        # .mat 파일 불러오기
        mat_data = scipy.io.loadmat(mat_file_path)
        print(f"'{mat_file_path}' 파일 로드 완료. 포함된 변수: {list(mat_data.keys())}")

        # .mat 파일의 기본 이름 (확장자 제외)을 가져옴
        base_filename = os.path.basename(mat_file_path).replace('.mat', '')

        # mat 파일 내의 각 변수에 대해 반복
        for key, value in mat_data.items():
            # 메타데이터(예: '__header__')는 건너뛰기
            if key.startswith('__'):
                continue

            # 변수의 데이터가 numpy 배열이고, CSV로 저장할 의미가 있는지 확인
            if isinstance(value, np.ndarray) and value.ndim > 0:
                
                # DataFrame으로 변환 가능한 형태로 만들기 (필요시)
                # 만약 데이터가 1차원이라면 2차원으로 변경
                if value.ndim == 1:
                    value = value.reshape(-1, 1) # 열벡터로 변환
                
                # Numpy 배열을 Pandas DataFrame으로 변환
                df = pd.DataFrame(value)

                # 저장할 CSV 파일 이름 설정 (예: S1_A1_E1_emg.csv)
                output_csv_path = os.path.join(output_dir, f"{base_filename}_{key}.csv")

                # DataFrame을 CSV 파일로 저장 (인덱스는 저장하지 않음)
                df.to_csv(output_csv_path, index=False)
                print(f"  -> 변수 '{key}'를 '{output_csv_path}' 파일로 저장했습니다.")

    except Exception as e:
        print(f"'{mat_file_path}' 파일 변환 중 오류 발생: {e}")


def batch_convert(input_dir, output_dir):
    """
    특정 폴더에 있는 모든 .mat 파일을 찾아 CSV로 변환합니다.

    :param input_dir: .mat 파일이 있는 폴더 경로
    :param output_dir: 변환된 CSV 파일을 저장할 폴더 경로
    """
    # 출력 폴더가 없으면 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 입력 폴더 내의 모든 파일을 확인
    for filename in os.listdir(input_dir):
        # 파일이 .mat 확장자로 끝나는 경우에만 처리
        if filename.endswith('.mat'):
            mat_file_path = os.path.join(input_dir, filename)
            convert_mat_to_csv(mat_file_path, output_dir)

# --- 메인 코드 실행 부분 ---
if __name__ == "__main__":
    # 1. .mat 파일이 저장된 폴더 경로를 지정하세요.
    # 예: 'C:/Users/YourUser/Desktop/Ninapro_Data'
    # '.'은 현재 스크립트가 있는 폴더를 의미합니다.
    INPUT_DIRECTORY = '/Users/mac/Documents/Sheffiled University/Dissertation/AI-ML-Portfolio/NinaPro/data/DB6/DB6_s1_a' 

    # 2. 변환된 CSV 파일을 저장할 폴더 경로를 지정하세요.
    # 'mat_to_csv_output' 이라는 이름의 폴더가 생성됩니다.
    OUTPUT_DIRECTORY = '../NinaPro/data/csv'

    print(f"'{INPUT_DIRECTORY}' 폴더의 .mat 파일을 '{OUTPUT_DIRECTORY}' 폴더로 변환합니다...")
    batch_convert(INPUT_DIRECTORY, OUTPUT_DIRECTORY)
    print("모든 변환 작업이 완료되었습니다.")