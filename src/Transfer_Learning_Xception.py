import keras
from keras.models import Model
from keras.layers import Dense, Conv2D, GlobalAveragePooling2D
from preprocessing import *
from keras.optimizers import Adam



# from keras import metrics
# metrics.categorical_accuracy()

class Xception_Model():
    def __init__(self, batch_size, num_classes, trainable=True):
        self.batch_size = batch_size
        self.model = keras.applications.Xception(include_top=False,weights=None,input_shape=(96,96,3))
        if trainable:
            self.model.load_weights('./pretrained_model/xception_weights_tf_dim_ordering_tf_kernels_notop.h5')

        for layer in self.model.layers[:-20]:
            layer.trainable = False

        x = self.model.output
        x = GlobalAveragePooling2D()(x)
        # add a fully-connected layer
        x = Dense(1024, activation='relu')(x)
        self.predictions = Dense(num_classes, activation='softmax', name='out_put')(x)
        # model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)
        self.model = Model(inputs=self.model.input, outputs=self.predictions)
        self.model.compile(optimizer=Adam(lr=0.0005), loss='categorical_crossentropy', metrics=['accuracy'])

        self.earlyStopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.001, patience=3, verbose=1)
        self.tensorBoard = keras.callbacks.TensorBoard('./logs',batch_size=batch_size, write_grads=True,write_images=True)
        self.checkpoint = keras.callbacks.ModelCheckpoint('./out_model/weights.{epoch:02d}-{val_loss:.2f}.hdf5',
                                                          monitor='val_loss', verbose=0, save_best_only=True,
                                                          save_weights_only=False, mode='auto', period=1)
        self.callBackList = [self.earlyStopping, self.tensorBoard, self.checkpoint]

        [self.train_loss, self.train_metrics] = 2*[None]
        self.history = None


    def load_model(self, path):
        self.model.load_weights(path)

    def fit(self, x, y, x_val=None, y_val=None):
        if x_val == None:
            self.history = self.model.fit(x, y, batch_size=self.batch_size, epochs=200, verbose=1,
                                          callbacks=self.callBackList, validation_split=0.1)
        else:
            self.history = self.model.fit(x, y, batch_size=self.batch_size, epochs=200, verbose=1,
                                          callbacks=self.callBackList, validation_data=(x_val, y_val))
        self.train_loss, self.train_metrics = self.model.evaluate(x=x, y=y, batch_size=512, verbose=1)
        return self.history

    def sumary(self):
        return self.model.summary()

    def predict(self, x):
        return self.model.predict(x)

def accuracy(test_x, test_y, model):
    results = model.predict(test_x)
    predicted_labels = np.argmax(results, axis=1)
    true_labels = np.argmax(test_y, axis=1)
    num_correct = np.sum(predicted_labels == true_labels)
    accuracy = float(num_correct)/results.shape[0]
    return (accuracy * 100)

def run():

    dir_test = r'E:\datasets\animal_BTL_ML\test'

    X, Y = load_image(dir_test, W=96, H=96)

    model = Xception_Model(16, 5, trainable=False)
    model.sumary()

    model.load_model("weights.05-0.50.hdf5")
    # model.fit(X,Y)


    print("accuracy on testset: ", accuracy(X,Y,model))
    print(1)
    print(2)

if __name__ == '__main__':
    run()
