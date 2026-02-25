import tensorflow as tf

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = (x_train / 255.0).reshape(-1, 28*28)
x_test  = (x_test  / 255.0).reshape(-1, 28*28)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28*28,)),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax"),
])

model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

model.fit(x_train, y_train, epochs=2, batch_size=256, verbose=2)
loss, acc = model.evaluate(x_test, y_test, verbose=0)
print("test accuracy:", acc)