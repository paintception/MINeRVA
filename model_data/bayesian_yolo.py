"""
The idea is to add dropout layers before the convolutional layers that precede the predictions of the network.
In total there is 3 output layers, adding the dropout layer requires to redefine the original computational graph.
We assert that the newly defined model has 3 extra layers (dropout layers) properly located in the graph and that its
3 main outputs are kept.

The script redefines the computational graph properly by adding dropout where needed.
"""

from keras.models import load_model
from keras.layers import Dropout
from keras.models import Model, Sequential
from keras.utils import plot_model

original_yolo = load_model('yolo.h5')

plot_model(original_yolo, 'original_yolo.png')

print(len(original_yolo.layers))
print(len(original_yolo.output))

output_1 = original_yolo.get_layer('up_sampling2d_1').output

output_1 = original_yolo.get_layer('leaky_re_lu_58').output
output_2 = original_yolo.get_layer('conv2d_59')
bayesian_layer_1 = Dropout(0.6)
bayesian_layer_1 = bayesian_layer_1(output_1)
new_output_1 = output_2(bayesian_layer_1)

output_1 = original_yolo.get_layer('leaky_re_lu_65').output
output_2 = original_yolo.get_layer('conv2d_67')
bayesian_layer_2 = Dropout(0.6)
bayesian_layer_2 = bayesian_layer_2(output_1)
new_output_2 = output_2(bayesian_layer_2)

output_1 = original_yolo.get_layer('leaky_re_lu_72').output
output_2 = original_yolo.get_layer('conv2d_75')
bayesian_layer_2 = Dropout(0.6)
bayesian_layer_2 = bayesian_layer_2(output_1)
new_output_3= output_2(bayesian_layer_2)

bayesian_model = Model(input=original_yolo.input, output=[new_output_1, new_output_2, new_output_3])
plot_model(bayesian_model, 'bayesian_yolo.png')

print(len(bayesian_model.layers))
print(len(bayesian_model.output))

bayesian_model.save('bayesian_yolo.h5')