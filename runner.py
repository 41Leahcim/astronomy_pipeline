import tensorflow as tf

model = tf.keras.models.load_model("model.keras")
print(model.summary())
