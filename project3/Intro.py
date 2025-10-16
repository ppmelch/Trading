import tensorflow as tf 
import mlflow

mlflow.set_experiment("CNN Tuning")

#Load and preprocess the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0


def build_model(params):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input(shape=(32, 32, 3)))

    num_filters = params.get("conv_filters", 32)
    conv_layers = params.get("conv_layers", 2)
    activation = params.get("activation", "relu")

    for i in range(conv_layers):
        model.add(tf.keras.layers.Conv2D(num_filters, (3, 3), activation=activation))
        model.add(tf.keras.layers.MaxPooling2D((2, 2)))
        num_filters *= 2

    dense_units = params.get("dense_units", 64)
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(dense_units, activation=activation))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))
    optimizer = params.get("optimizer", "adam")

    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
        
    return model


params_space = [
    {"conv_layers":2 , "conv_filters":32, "activation":"relu", "dense_units":64},
    {"conv_layers":3 , "conv_filters":32, "activation":"relu", "dense_units":32},
    {"conv_layers":2 , "conv_filters":32, "activation":"sigmoid", "dense_units":64},
]    


print("Starting MLflow run...")
for params in params_space:
    with mlflow.start_run() as run:
        run_name = f"conv{params['conv_layers']}_filters{params['conv_filters']}"
        run_name += f"_dense{params['dense_units']}_activation{params['activation']}"

        mlflow.log_params(params)

        model = build_model(params)
        hist = model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test), verbose=2)


        final_metrics = {
            "val_accuracy": hist.history['val_accuracy'][-1],
            "val_loss": hist.history['val_loss'][-1]
        }

        mlflow.log_metrics(final_metrics)
        print(f"Final metrics: {final_metrics}")

        mlflow.tensorflow.log_model(model,"model")
        