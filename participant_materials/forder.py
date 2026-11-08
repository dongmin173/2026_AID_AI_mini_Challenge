import os

def save_folder_structure_with_content(target_path, output_txt_path):
    """
    지정한 폴더의 구조와 파일 내용을 텍스트 파일로 저장합니다.
    (무거운 가상환경 폴더 및 대용량/바이너리 확장자는 내용 읽기를 제외합니다)
    """
    # 1. 탐색에서 완전히 제외할 폴더 목록
    EXCLUDE_DIRS = {'.venv', 'venv', 'env', '.git', '__pycache__', '.vscode', '.idea'}
    
    # 2. 구조만 표시하고 '파일 내용은 읽지 않을' 확장자 목록 (소문자 기준)
    BINARY_EXTENSIONS = {
        '.pt', '.pth', '.ckpt', '.bin', '.onnx',     # AI/딥러닝 모델 파일
        '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.ico', # 이미지 파일
        '.mp4', '.avi', '.mkv', '.mov',                # 영상 파일
        '.zip', '.tar', '.gz', '.7z', '.rar',          # 압축 파일
        '.exe', '.dll', '.so', '.dylib',               # 실행/바이너리 파일
        '.pyc', '.pyd'                                 # 파이썬 컴파일 파일
    }

    with open(output_txt_path, 'w', encoding='utf-8') as f:
        
        def traverse_dir(current_path, depth=0):
            indent = "    " * depth
            try:
                items = sorted(os.listdir(current_path))
            except PermissionError:
                f.write(f"{indent}⚠️ [접근 권한이 없는 폴더입니다]\n")
                return

            for item in items:
                item_path = os.path.join(current_path, item)
                
                # 결과물(.txt) 자체가 다시 기록되는 것을 방지
                if os.path.abspath(item_path) == os.path.abspath(output_txt_path):
                    continue
                
                if os.path.isdir(item_path):
                    # 제외 대상 폴더인 경우 스킵
                    if item.lower() in EXCLUDE_DIRS:
                        f.write(f"{indent}📁 {item}/ [탐색 제외 폴더]\n")
                        continue
                        
                    f.write(f"{indent}📁 {item}/\n")
                    traverse_dir(item_path, depth + 1)
                else:
                    f.write(f"{indent}📄 {item}\n")
                    
                    # 파일 확장자 확인
                    _, ext = os.path.splitext(item.lower())
                    
                    # 바이너리 파일인 경우 내용 읽기 건너뛰기
                    if ext in BINARY_EXTENSIONS:
                        f.write(f"{"    " * (depth + 1)}[바이너리 파일: 내용 생략]\n\n")
                        continue
                    
                    # 텍스트 파일만 내용 읽기 시도
                    try:
                        with open(item_path, 'r', encoding='utf-8', errors='replace') as file_content:
                            content = file_content.read()
                            if content.strip():
                                content_indent = "    " * (depth + 1)
                                f.write(f"{content_indent}--- 파일 내용 시작 ---\n")
                                for line in content.splitlines():
                                    f.write(f"{content_indent}{line}\n")
                                f.write(f"{content_indent}--- 파일 내용 끝 ---\n")
                            else:
                                f.write(f"{"    " * (depth + 1)}[빈 파일]\n")
                    except Exception:
                        f.write(f"{"    " * (depth + 1)}[읽을 수 없는 파일 형식입니다]\n")
                    
                    f.write("\n")

        f.write(f"📂 루트 폴더: {os.path.abspath(target_path)}\n")
        f.write("=" * 50 + "\n\n")
        traverse_dir(target_path)

if __name__ == "__main__":
    target_folder = "." 
    output_file = "folder_structure.txt"
    
    save_folder_structure_with_content(target_folder, output_file)
    print(f"폴더 구조와 내용이 '{output_file}'에 성공적으로 저장되었습니다.")
