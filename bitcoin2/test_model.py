import onnxruntime as ort

print('ONNX Runtime version:', ort.__version__)

try:
    session = ort.InferenceSession('model/rf_model.onnx')
    print('Model loaded successfully')
    print('Input names:', [inp.name for inp in session.get_inputs()])
    print('Output names:', [out.name for out in session.get_outputs()])
    print('Input shapes:', [inp.shape for inp in session.get_inputs()])
    print('Output shapes:', [out.shape for out in session.get_outputs()])
except Exception as e:
    print('Error loading model:', str(e))
