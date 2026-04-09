import cv2
from inference import InferencePipeline
from inference.core.interfaces.camera.entities import VideoFrame

# 결과를 화면에 보여주는 함수
def render_callback(predictions: dict, video_frame: VideoFrame):
    # 워크플로우에서 출력으로 설정한 'final_image'를 가져옵니다.
    annotated_image = predictions["final_image"].numpy_image
    
    # 화면에 출력
    cv2.imshow("K-Ride Safety Monitor", annotated_image)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        exit()

# 인퍼런스 파이프라인 초기화
pipeline = InferencePipeline.init_with_workflow(
    api_key="YOUR_ROBOFLOW_API_KEY",      # 여기에 API 키 입력
    workspace_name="minas-workspace-fnwe0", # 워크스페이스 이름
    workflow_id="k-ride-bicycle-safety-pipeline-1775705723209", # 워크플로우 ID
    video_reference="my_bicycle_ride.mp4", # 0으로 설정하면 웹캠 사용
    on_prediction=render_callback,
)

pipeline.start() # 시작
pipeline.join()  # 종료 대기
