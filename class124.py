import cv2
import tensorflow as tf
import numpy as np

Video=cv2.VideoCapture(0)
model=tf.keras.models.load_model("keras_model.h5")
print(model)

while True:
    dummy,frame=Video.read()
    #resizing captured img to 224x224 pix
    img=cv2.resize(frame,(224,224))
    #print(img.size)

    #converting img data to array

    test_img=np.array(img,dtype=np.float32)
    #print(test_img)

    #converting 3d array to 4d

    array_test_img=np.expand_dims(test_img,axis=0)
    #print(array_test_img)

    #normalizing img data to 0 and 1 format

    normalized_img=array_test_img/255.0
    # print(normalized_img)

    prediction=model.predict(normalized_img)
    print(prediction)
    cv2.imshow("mask detection",frame)
    key=cv2.waitKey(0)
    if key==32:
        print("closing")
        break
Video.release()
cv2.destroyAllWindows()