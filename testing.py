from PIL import Image,ImageFilter
import pytesseract
import cv2

# Open an image file
im = Image.open("/home/krishnadev/Pictures/data.png")

# Convert the image to grayscale
im_gray = im.convert("L")

# Perform some preprocessing (e.g., denoising, thresholding)
im_processed = im_gray.filter(ImageFilter.GaussianBlur(radius=2))

# Save the processed image (for visualization, optional)
im_processed.save("processed_imag.jpg")

# Use pytesseract to extract text from the processed image
text = pytesseract.image_to_string(im_processed, lang='eng')

# Print the extracted text
print(text)
