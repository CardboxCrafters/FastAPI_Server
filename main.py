from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
import cv2
import numpy as np
from io import BytesIO

app = FastAPI()

def crop_image(image):
    # 이미지를 numpy 배열로 변환
    nparr = np.frombuffer(image, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # 이미지 전처리
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 200)

    # 윤곽선 찾기
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        if len(approx) == 4:  # 근사 다각형이 4개의 꼭지점을 갖는 경우
            cardContour = approx
            break
    else:
        print("No business card detected.")
        return None

    # 카드 영역 자르기
    src_pts = cardContour.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")

    # 좌표 정렬
    s = src_pts.sum(axis=1)
    rect[0] = src_pts[np.argmin(s)]
    rect[2] = src_pts[np.argmax(s)]

    diff = np.diff(src_pts, axis=1)
    rect[1] = src_pts[np.argmin(diff)]
    rect[3] = src_pts[np.argmax(diff)]

    (tl, tr, br, bl) = rect

    # 최대 너비와 높이 계산
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst_pts = np.array([[0, 0],
                        [maxWidth - 1, 0],
                        [maxWidth - 1, maxHeight - 1],
                        [0, maxHeight - 1]], dtype="float32")

    # 변환 행렬 계산
    perspectiveTransform = cv2.getPerspectiveTransform(rect, dst_pts)
    croppedImage = cv2.warpPerspective(img, perspectiveTransform, (maxWidth, maxHeight))

    # 이미지를 byte 배열로 변환하여 반환
    _, img_encoded = cv2.imencode('.jpg', croppedImage)
    return img_encoded.tobytes()


@app.post("/cropImage")
async def upload_file(file: UploadFile = File(...)):
    # 파일 데이터 읽기
    contents = await file.read()

    # 이미지 자르기
    cropped_image = crop_image(contents)

    if cropped_image is None:
        return {"message": "No business card detected."}

    # 바이트 스트림으로 반환
    return StreamingResponse(BytesIO(cropped_image), media_type="image/jpeg")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
