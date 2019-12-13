MODEL_INPSHAPE = (28, 28, 1)

def get_model():
    import tensorflow.keras as keras

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

            img_resized = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            
            cnts,_ = cv2.findContours(img_resized.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            
            for c in cnts:
                # compute the center of the contour, then detect the name of the
                # shape using only the contour
                    
                # multiply the contour (x, y)-coordinates by the resize ratio,
                # then draw the contours and the name of the shape on the image
                
                x,y,w,h = cv2.boundingRect(c)
                
                ROI = img_resized[y:y+h, x:x+w]
                
            ROI = cv2.resize(ROI, MODEL_INPSHAPE[0:2]).reshape(28,28,1)
            
            pred = model.predict_proba(np.expand_dims(ROI, 0))[0]
            max_idx = np.argmax(pred)
            max_prb = pred[max_idx]

            cv2.putText(img, f"{max_idx}, Prob: {max_prb}", (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.rectangle(img, (x, y), (x+w, y+h), (255,0,0), 2)
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
    import tensorflow.keras as keras
    from tensorflow.keras.datasets import mnist

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

    model.fit(X_train, Y_train, batch, epochs, shuffle = True, validation_data=(X_test, Y_test))
    model.save_weights(model_fpath)


train(model_fpath="model.h5",
      model_load=False,
      epochs=100,
      batch=10000
      )

inference("model.h5")