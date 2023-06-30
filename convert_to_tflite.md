To convert the model to tflite, run pt-to-tflite.py, which converts the model at a specific path to a tflite model. to inference the tflite model, follow the example inference code in tf_lite_sample_inference.py.

both of the mentioned scripts assume an experiment file has been added to the base directory of the repository. to change the file path, simply change the file paths within the script.

the script also requires at least one sample image to run, as onnx needs a sample input to trace the graph.

