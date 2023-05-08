from predict import predict_from_args
import os, sys
import scipy.stats as st
experiment="experiment_9"
test_path = f'{experiment}\\test'
types = ["good", "crack", "leakage"]
for t in types:
    sub_path = os.path.join(test_path, t)
    output_directory = os.path.join(sub_path, "results")
    output = ""
    images = os.listdir(sub_path)
    for image in images:
        name = image[:-4]
        output_name = name+"_vis"
        outlier_score = predict_from_args(os.path.join(sub_path, image), output_directory, output_name, experiment_dir=experiment, topk=0.25)
        output += name + (" Anomaly Score: %.4f" % outlier_score) + ("    Anomaly Probability: {:.2f} %".format((1-(2*(1-st.norm.cdf(outlier_score))))*100)) +"\n\n"
    with open(os.path.join(output_directory, "scores.txt"), "w+") as f:
        f.write(output)

        
test_path = f'{experiment}\\test2'
types = ["crack", "leakage"]
for t in types:
    sub_path = os.path.join(test_path, t)
    output_directory = os.path.join(sub_path, "results")
    output = ""
    images = os.listdir(sub_path)
    for image in images:
        name = image[:-4]
        output_name = name+"_vis"
        outlier_score = predict_from_args(os.path.join(sub_path, image), output_directory, output_name, experiment_dir=experiment, topk=0.25)
        output += name + (" Anomaly Score: %.4f" % outlier_score) + ("    Anomaly Probability: {:.2f} %".format((1-(2*(1-st.norm.cdf(outlier_score))))*100)) +"\n\n"
    with open(os.path.join(output_directory, "scores.txt"), "w+") as f:
        f.write(output)