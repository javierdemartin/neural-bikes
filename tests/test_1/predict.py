from keras.models import load_model

model = load_model('test_1.h5')

weights = model.get_weights()

print weights
