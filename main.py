MODEL_INPSHAPE = (28, 28, 1)

def get_model():
    import keras

    model = keras.models.Sequential([
        keras.layers.Conv2D(filters=8, kernel_size=(3, 3), input_shape=MODEL_INPSHAPE),
        keras.layers.Activation('relu'),
        keras.layers.MaxPool2D((2, 2), (2, 2)),
        keras.layers.Conv2D(filters=16, kernel_size=(3, 3)),
        keras.layers.Activation('relu'),
        keras.layers.MaxPool2D((2, 2), (2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(10),
        keras.layers.Activation('softmax'),
        ])

    return model

def inference(
    model_path: str
):
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt

    model = get_model()
    print(f"Loading model from {model_path}")
    model.load_weights(model_path)

    cam = cv2.VideoCapture(0)
    cv2.namedWindow(model_path)

    w = cam.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = cam.get(cv2.CAP_PROP_FRAME_HEIGHT)
    s = int(h) if h < w else int(w)
    print(f"Image H: {h}, Image W: {w}. Croping to {s}.")
    img_center = np.empty((s, s, 3), np.uint8)

    x0 = int((s-w)/2)
    x1 = int(s-x0)
    y0 = int((s-h)/2)
    y1 = int(s-y0)

    x0 = abs(x0)
    x1 = abs(x1)
    y0 = abs(y0)
    y1 = abs(y1)

    print("Done. Press space to capture.")

    while True:
        k = cv2.waitKey(0)

        # SPACE press
        if k == 32:
            print("Capturing image...")

            _, img = cam.read()
            if img.size == 0:
                print("Could not capture image")
                return

            img_center = img[y0:y1, x0:x1]
            img_resized = cv2.cvtColor(img_center, cv2.COLOR_BGR2GRAY)
            img_resized = cv2.resize(img_resized, MODEL_INPSHAPE[0:2])
            img_resized = np.expand_dims(img_resized, 2)

            pred = model.predict_proba(np.expand_dims(img_resized, 0))[0]
            max_idx = np.argmax(pred)
            max_prb = pred[max_idx]

            cv2.putText(img, f"{max_idx}, Prob: {max_prb}", (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imshow(model_path, img)

        # ESC press
        if k == 27:
            print("Exit")
            cam.release()
            cv2.destroyAllWindows()
            return

def train(
    model_fpath: str,
    model_load: bool,
    epochs: int,
    batch: int
):
    import numpy as np
    import keras
    from keras.datasets import mnist

    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
    print(f"Original X shape: {X_train.shape}, Original Y shape: {Y_train.shape}")
    X_train = np.expand_dims(X_train, 3)
    X_test = np.expand_dims(X_test, 3)
    Y_test  = keras.utils.to_categorical(Y_test, num_classes=10)
    Y_train = keras.utils.to_categorical(Y_train, num_classes=10)

    model = get_model()
    model.compile(optimizer='Adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
        )

    # Load a model if specified
    if model_load:
        print(f"Loading model from {model_fpath}")
        model.load_weights(model_fpath)

    model.fit(X_train, Y_train, batch, epochs, validation_data=(X_test, Y_test))
    model.save_weights(model_fpath)

if __name__ == "__main__":
    import argparse

    arg_p = argparse.ArgumentParser()
    arg_p.add_argument("cmd", type=str, help="One of 'train' or 'inference'")
    arg_p.add_argument("--model_path", type=str, help="Model filename")
    arg_p.add_argument("--load", action="store_true", help="Load a previous model instead of training from scratch")
    arg_p.add_argument("--epochs", type=int, help="Number of train epochs")
    arg_p.add_argument("--batch", type=int, help="Image batch size")

    args = arg_p.parse_args()

    if args.cmd == "train":
        if args.model_path == None:
            print("'--model_path' is required. Set it to the output/trained model path")
            exit()

        train(model_fpath=args.model_path,
            model_load=args.load,
            epochs=args.epochs,
            batch=args.batch
            )

    if args.cmd == "inference":
        if args.model_path == None:
            print("'--model_path' is required. Set it to the trained model path")
            exit()
        inference(args.model_path)