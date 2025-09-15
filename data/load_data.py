# Set target image size and paths
target = 256
dataPath = 'yourpath'
batch_size = 8

def addNoise(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.uint8)
    ret, mask = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY)

    randStd = random.uniform(0, 10.0)
    gaussian = np.random.normal(randStd*-1, randStd, (target, target,3))
    noisy_image = image + gaussian
    image = np.clip(noisy_image, 0, 255).astype(np.uint8)

    image[mask == 0] = [0,0,0]
    #image = preprocess_input(image)
    return image

# Data generators
trainDataGen = ImageDataGenerator(
    preprocessing_function=addNoise,
    horizontal_flip=True,
    vertical_flip=True,
    channel_shift_range=10,
    width_shift_range=0.05,
    height_shift_range=0.05,
    rotation_range=360,
    shear_range=30,
    brightness_range=(0.95, 1.05),
    fill_mode='constant',
    cval=0,
    zoom_range=0.05
)

trainGen = trainDataGen.flow_from_directory(
    batch_size=batch_size,
    shuffle=True,
    class_mode="binary",
    target_size=(target, target),
    directory=os.path.join(dataPath, 'train'),
    color_mode='rgb'
)

valDataGen = ImageDataGenerator(rescale=1.0 / 255.0)
valGen = valDataGen.flow_from_directory(
    batch_size=1,
    class_mode="binary",
    target_size=(target, target),
    directory=os.path.join(dataPath, 'validation'),
    color_mode='rgb'
)

testDataGen = ImageDataGenerator(rescale=1.0 / 255.0)
testGen = testDataGen.flow_from_directory(
    batch_size=1,
    class_mode="binary",
    target_size=(target, target),
    directory=os.path.join(dataPath, 'test'),
    color_mode='rgb'
)
