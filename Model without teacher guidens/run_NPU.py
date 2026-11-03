import tflite_runtime.interpreter as tflite
import numpy as np
import cv2

# 1. Încărcarea delegatului NPU (specific NXP)
delegate = tflite.load_delegate('/usr/lib/libvx_delegate.so')
interpreter = tflite.Interpreter(
    model_path="student_depth_int8.tflite",
    experimental_delegates=[delegate]
)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 2. Preprocesarea imaginii
img = cv2.imread('picture1.jpg')
img_resized = cv2.resize(img, (224, 224))
img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

input_scale, input_zero_point = input_details[0]['quantization']
img_int8 = (img_rgb / input_scale + input_zero_point).astype(np.int8)
img_input = np.expand_dims(img_int8, axis=0)

# 3. Execuția inferenței pe NPU
interpreter.set_tensor(input_details[0]['index'], img_input)
interpreter.invoke()

# 4. Postprocesarea și salvarea
output_data = interpreter.get_tensor(output_details[0]['index'])[0]
output_scale, output_zero_point = output_details[0]['quantization']
depth_map = (output_data.astype(np.float32) - output_zero_point) * output_scale

depth_normalized = cv2.normalize(depth_map, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)

cv2.imwrite('depth_map_picture1_npu.png', depth_colored)
print("Generat cu succes pe NPU: depth_map_picture1_npu.png")