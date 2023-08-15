import onnxruntime
import numpy as np
import cv2

input_factor = np.array([1, 1, 4, 4], dtype=np.float32) 

input_img = cv2.imread('../face.png').astype(np.float32) 
input_img = np.transpose(input_img, [2, 0, 1]) 
input_img = np.expand_dims(input_img, 0) 

ort_session = onnxruntime.InferenceSession("srcnn.onnx", providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']) 
ort_inputs = {'input': input_img} 
ort_output = ort_session.run(['output'], ort_inputs)[0] 
 
ort_output = np.squeeze(ort_output, 0) 
ort_output = np.clip(ort_output, 0, 255) 
ort_output = np.transpose(ort_output, [1, 2, 0]).astype(np.uint8) 
cv2.imwrite("face_ort.png", ort_output)