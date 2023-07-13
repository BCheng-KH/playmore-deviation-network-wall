from predict import predict_with_model, predict_score_with_model, load_model, generate_args
import os, sys
import scipy.stats as st
import time
experiment="experiment_14"
test_path = f'{experiment}\\test3'

output_directory = os.path.join(test_path, "results")
output = ""
images = os.listdir(test_path)
img_size = 260
model_args = generate_args(experiment_dir=experiment, topk=0.25, no_cuda=True, img_size=img_size)
crack_args = generate_args(experiment_dir=experiment, topk=0.25, weight_name='crack.pkl', no_cuda=True, img_size=img_size)
leakage_args = generate_args(experiment_dir=experiment, topk=0.25, weight_name='leakage.pkl', no_cuda=True, img_size=img_size)
model = load_model(model_args)
crack_model = load_model(crack_args)
leakage_model = load_model(leakage_args)

for image in images[:10]:
    name = image[:-4]
    output_name = name+"_vis"
    start = time.time()
    outlier_score = predict_with_model(model, os.path.join(test_path, image), output_directory, output_name)
    crack_score = predict_score_with_model(crack_model, os.path.join(test_path, image))
    leakage_score = predict_score_with_model(leakage_model, os.path.join(test_path, image))
    print(time.time()-start)
    output += name + (" Anomaly Score: %.4f" % outlier_score) + ("    Anomaly Probability: {:.2f} %    Crack Probability: {:.2f} %    Leakage Probability: {:.2f} %".format((1-(2*(1-st.norm.cdf(outlier_score))))*100, (1-(2*(1-st.norm.cdf(crack_score))))*100, (1-(2*(1-st.norm.cdf(leakage_score))))*100)) +"\n\n"
with open(os.path.join(output_directory, "scores.txt"), "w+") as f:
    f.write(output)