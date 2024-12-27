import cv2
import numpy as np
import sounddevice as sd
import wave
import threading
import queue
from datetime import datetime
import os
import time
import subprocess


output_filename = ''

output_filename = ''
class AudioRecorder:
    def __init__(self, filename, samplerate=44100, channels=2, dtype=np.int16):
        self.filename = filename
        self.samplerate = samplerate
        self.channels = channels
        self.dtype = dtype
        self.recording = False
        self.audio_queue = queue.Queue()

        # 사용 가능한 오디오 장치 출력
        print("\n사용 가능한 오디오 장치:")
        print(sd.query_devices())

        # CABLE Output 장치 찾기
        self.device = None
        for i, device in enumerate(sd.query_devices()):
            if 'CABLE Output' in device['name']:
                self.device = i
                print(f"\n'CABLE Output' 장치를 찾았습니다: {device['name']}")
                break

        if self.device is None:
            print("\n경고: 'CABLE Output'을 찾을 수 없습니다. 기본 입력 장치를 사용합니다.")
            self.device = sd.default.device[0]

    def audio_callback(self, indata, frames, time, status):
        """오디오 콜백 함수"""
        if status:
            print(status)
        self.audio_queue.put(indata.copy())

    def start(self):
        """오디오 녹음을 시작합니다."""
        self.recording = True
        self.stream = sd.InputStream(
            device=self.device,
            channels=self.channels,
            samplerate=self.samplerate,
            callback=self.audio_callback,
            dtype=self.dtype
        )
        self.stream.start()

    def stop(self):
        """오디오 녹음을 중지하고 파일로 저장합니다."""
        if self.recording:
            self.recording = False
            self.stream.stop()
            self.stream.close()

            # WAV 파일로 저장
            with wave.open(self.filename, 'wb') as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(self.samplerate)

                # 모든 오디오 데이터를 파일에 쓰기
                while not self.audio_queue.empty():
                    wf.writeframes(self.audio_queue.get().tobytes())


def capture_av(video_device_index=0, resolution=(1920, 1080), fps=60.0):
    """비디오와 시스템 사운드를 동시에 녹화합니다."""
    # 파일 이름 설정
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    video_filename = f"video_{timestamp}.avi"
    audio_filename = f"audio_{timestamp}.wav"
    global output_filename
    # output_filename = f"recording_{timestamp}.mp4"

    # 비디오 캡처 설정
    cap = cv2.VideoCapture(video_device_index, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
    cap.set(cv2.CAP_PROP_FPS, fps)

    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"\n실제 설정된 해상도: {actual_width}x{actual_height}")
    print(f"실제 설정된 FPS: {actual_fps}")

    if not cap.isOpened():
        print("캡쳐카드를 열 수 없습니다.")
        return

    # 비디오 writer 설정
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(video_filename, fourcc, actual_fps,
                          (actual_width, actual_height))

    # 오디오 레코더 설정
    audio_recorder = AudioRecorder(audio_filename)

    print("\n녹화를 시작합니다. 종료하려면 'q'를 누르세요.")
    start_time = time.time()
    frames = 0

    try:
        # 오디오 녹음 시작
        audio_recorder.start()

        while True:
            ret, frame = cap.read()
            if not ret:
                print("프레임을 읽을 수 없습니다.")
                break

            frames += 1
            elapsed_time = time.time() - start_time
            current_fps = frames / elapsed_time

            # 프레임 저장
            out.write(frame)

            # 녹화 중인 화면 표시 (축소된 크기로)
            display_frame = cv2.resize(frame, (960, 540))

            # FPS와 녹화 시간 정보 추가
            cv2.putText(display_frame, f"FPS: {current_fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(display_frame, f"Time: {elapsed_time:.1f}s", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow('Recording (Press q to stop)', display_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\n녹화가 중단되었습니다.")

    finally:
        # 녹화 중지 및 자원 해제
        audio_recorder.stop()
        cap.release()
        out.release()
        cv2.destroyAllWindows()

        duration = time.time() - start_time
        average_fps = frames / duration

        print("\n녹화 완료. 파일을 병합하는 중...")

        # FFmpeg를 사용하여 비디오와 오디오 병합
        try:
            subprocess.run([
                'ffmpeg', '-i', video_filename,
                '-i', audio_filename,
                '-c:v', 'copy',
                '-c:a', 'aac',
                output_filename
            ], check=True)

            # 임시 파일 삭제
            os.remove(video_filename)
            os.remove(audio_filename)

            print(f"\n녹화 통계:")
            print(f"- 녹화 시간: {duration:.1f}초")
            print(f"- 평균 FPS: {average_fps:.1f}")
            print(f"- 총 프레임 수: {frames}")
            print(f"- 파일 저장 경로: {os.path.abspath(output_filename)}")

        except subprocess.CalledProcessError:
            print("\nFFmpeg 오류: 비디오와 오디오를 병합할 수 없습니다.")
            print(f"개별 파일이 저장된 경로:")
            print(f"- 비디오: {os.path.abspath(video_filename)}")
            print(f"- 오디오: {os.path.abspath(audio_filename)}")
        except FileNotFoundError:
            print("\nFFmpeg가 설치되어 있지 않습니다.")
            print("FFmpeg를 설치하거나 개별 파일을 사용하세요:")
            print(f"- 비디오: {os.path.abspath(video_filename)}")
            print(f"- 오디오: {os.path.abspath(audio_filename)}")
            
            
def set_category():
    category_list = ['의', '식', '주', '소비', '가족', '교육론']
    for idx, category in enumerate(category_list):
        print(f'{idx}: {category}')
        if not os.path.exists(category):
            # 디렉토리가 없으면 생성
            os.makedirs(category)
    category_index = int(input('영역 번호를 입력하세요 : '))
    return category_list[category_index]

def set_filename(category_name):
    global output_filename
    video_number = input('강의 번호를 입력하세요 : ')
    path_temp = os.path.join(category_name, category_name+'영역_'+video_number+'강').replace('\\', '/')
    output_filename = f"{path_temp}.mp4"
    

if __name__ == "__main__":
    # 캡쳐보드 설정
    device_index = 1  # 기본 캡쳐 장치
    resolution = (1920, 1080)  # Full HD
    fps = 60.0  # 60fps
    
    category_name = set_category()
    set_filename(category_name)

    print("캡쳐 장치 설정:")
    print(f"- 장치 번호: {device_index}")
    print(f"- 목표 해상도: {resolution[0]}x{resolution[1]}")
    print(f"- 목표 FPS: {fps}")

    # 3초 대기 후 녹화 시작
    print("\n3초 후에 녹화가 시작됩니다...")
    time.sleep(3)

    # 녹화 시작
    capture_av(device_index, resolution, fps)