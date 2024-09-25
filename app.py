from flask import Flask, render_template, request, send_file
import cv2
import numpy as np

app = Flask(__name__)

def semantic_segmentation(image):
    # Create a single-channel mask
    height, width = image.shape[:2]
    mask = np.zeros((height, width), dtype='uint8')  # Create a single-channel mask

    # Create a mask for the middle portion of the image
    mask[height // 4: height * 3 // 4, width // 4: width * 3 // 4] = 255  # White mask for the foreground
    return mask

def colorize_targeted(image, mask):
    # Convert the original grayscale image to BGR color
    colorized_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)  # Convert grayscale to BGR color

    # Colorize the selected region based on the mask
    colorized_image[mask == 255] = [0, 255, 0]  # Colorize the selected region in green
    return colorized_image

@app.route('/')
def upload_form():
    return render_template('upload.html')

@app.route('/', methods=['POST'])
def upload_image():
    file = request.files['image']
    
    # Save the uploaded image
    file_path = 'uploaded_image.jpg'
    file.save(file_path)
    
    # Read the image in grayscale
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

    # Check if the image was loaded properly
    if image is None:
        return "Error: Could not read the image."

    # Perform semantic segmentation
    mask = semantic_segmentation(image)
    
    # Check the shape of the mask
    print("Mask shape:", mask.shape)

    # Colorize the image based on the mask
    result_image = colorize_targeted(image, mask)
    
    # Save result
    result_path = 'result_image.jpg'
    cv2.imwrite(result_path, result_image)

    return send_file(result_path, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True)



