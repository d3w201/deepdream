import tensorflow as tf
import numpy as np
import PIL.Image
import time

# Download an image and read it into a NumPy array.
from deep import DeepDream


def download(param_url, max_dim=None):
    name = url.split('/')[-1]
    image_path = tf.keras.utils.get_file(name, param_url)
    target_img = PIL.Image.open(image_path)
    if max_dim:
        target_img.thumbnail((max_dim, max_dim))
    return np.array(target_img)


# Normalize an image
def de_process(target_img):
    target_img = 255 * (target_img + 1.0) / 2.0
    return tf.cast(target_img, tf.uint8)


# Display an image
def show(target_img):
    img_to_show = PIL.Image.fromarray(np.array(target_img))
    img_to_show.show()


def run_deep_dream_simple(target_img, steps=100, step_size=0.01):
    # Convert from uint8 to the range expected by the model.
    target_img = tf.keras.applications.inception_v3.preprocess_input(target_img)
    target_img = tf.convert_to_tensor(target_img)
    step_size = tf.convert_to_tensor(step_size)
    steps_remaining = steps
    step = 0
    while steps_remaining:
        if steps_remaining > 100:
            run_steps = tf.constant(100)
        else:
            run_steps = tf.constant(steps_remaining)
        steps_remaining -= run_steps
        step += run_steps

        loss, target_img = deepdream(target_img, run_steps, tf.constant(step_size))

        # show(de_process(target_img))
        print("Step {}, loss {}".format(step, loss))

    result = de_process(target_img)
    # show(result)

    return result


print('s t a r t i n g . . .')

url = "file:///C:/Users/utente/Pictures/Camera%20Roll/boos_smurf.jpg"

print('searching for file :: ' + url)

# Downsizing the image makes it easier to work with.
original_img = download(url, 500)
# show(original_img)

# input("Press Enter to continue...")

# c o n v o l u t i o n a l - n e u r a l - n e t w o r k

# b a s e - m o d e l
base_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')

# s e l e c t - l a y e r s
layers = [base_model.get_layer(name).output for name in ['mixed7', 'mixed0']]

# e x t r a c t i o n - m o d e l
dream_model = tf.keras.Model(base_model.input, layers)

# p r e p a r e
deepdream = DeepDream(dream_model)

# d r e a m
dream_img = run_deep_dream_simple(target_img=original_img, steps=100, step_size=0.01)

# input("Press Enter to continue...")

# p o s t - p r o c e s s i n g
start = time.time()

OCTAVE_SCALE = 1.30

img = tf.constant(np.array(original_img))
base_shape = tf.shape(img)[:-1]
float_base_shape = tf.cast(base_shape, tf.float32)

for n in range(-2, 3):
    new_shape = tf.cast(float_base_shape * (OCTAVE_SCALE ** n), tf.int32)

    img = tf.image.resize(img, new_shape).numpy()

    img = run_deep_dream_simple(target_img=img, steps=50, step_size=0.01)

# r e s i z i n g
img = tf.image.resize(img, base_shape)
img = tf.image.convert_image_dtype(img / 255.0, dtype=tf.uint8)
show(img)
end = time.time()
time_elapsed = end - start
print("task finished in " + str(time_elapsed))
