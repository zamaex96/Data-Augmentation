from PIL import Image

def bicubic_interpolation(image, target_size):
    # Apply bicubic interpolation
    resized_image = image.resize(target_size, Image.BICUBIC)
    return resized_image

def convert_to_binary(image, threshold):
    # Convert the image to grayscale
    grayscale_image = image.convert('L')

    # Apply binary thresholding
    binary_image = grayscale_image.point(lambda pixel: 0 if pixel < threshold else 255, 'L')

    return binary_image

def main():
    # Input image path
    input_image_path = 'C:/Users/USPLa/OneDrive/Desktop/dummy 2/00.jpg'

    # Load the original image
    original_image = Image.open(input_image_path)

    # Resize the original image to 150x110
    target_size = (28, 28)
    resized_original_image = original_image.resize(target_size, Image.BICUBIC)

    # Display the resized original image
    resized_original_image.show(title='Resized Original Image')

    # Bicubic interpolation on the resized image
    scale_factor = 1  # Adjust the scale factor as needed
    bicubic_image = bicubic_interpolation(resized_original_image, (int(target_size[0] * scale_factor), int(target_size[1] * scale_factor)))

    # Display the bicubic interpolated image
    bicubic_image.show(title='Bicubic Interpolated Image')

    # Convert to binary image using thresholding
    threshold_value = 200  # Adjust the threshold as needed
    binary_image = convert_to_binary(bicubic_image, threshold_value)

    # Display the binary image
    binary_image.show(title='Binary Image')
    # Save images with compression to compare file sizes
    resized_original_image.save('resized_image.jpg', quality=95)
    bicubic_image.save('bicubic_interpolated_image.jpg', quality=85)  # Adjust quality as needed
    binary_image.save('binary_image.jpg', quality=95)

if __name__ == "__main__":
    main()