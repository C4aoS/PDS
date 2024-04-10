import cv2
import numpy as np
import docScannerUtils as utils
import pytesseract

def capture_image_from_camera(camera_index=0):
    """
    Captures a single image from the specified camera.
    """
    cap = cv2.VideoCapture(camera_index)
    ret, img = cap.read()
    cap.release()
    if not ret:
        print("Failed to capture image.")
        return None
    return img

def preprocess_image(img):
    """
    Applies preprocessing steps to the image to prepare it for contour detection.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 1)
    threshold_img = cv2.Canny(blur, 200, 200)
    return threshold_img

def main():
    img = capture_image_from_camera()
    if img is None:
        return

    threshold_img = preprocess_image(img)
    contours, _ = cv2.findContours(threshold_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    biggest, _ = utils.find_biggest_contour(contours)

    if biggest.size != 0:
        biggest = utils.reorder(biggest)
        warped_img = utils.four_point_transform(img, biggest.reshape(4, 2))
        rotated_img = utils.rotate_image(warped_img, -90)
        
        # OCR with pytesseract
        extracted_text = pytesseract.image_to_string(rotated_img, lang='eng')
        print(extracted_text)
        
        # Display the processed image and wait for a key press
        cv2.imshow("Scanned Document", rotated_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No document found.")

if __name__ == "__main__":
    main()
