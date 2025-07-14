from PIL import Image, ImageDraw, ImageFont
import io

def combine_pil_images_to_bytes(images, labels, font_path=None):
    """
    Combines multiple PIL images horizontally, adds labels beneath each image, and returns the resultar as a byte array.

    Args:
        images (list): List of PIL Image objects (1024x1024).
        labels (list): List of labels corresponding to each image.
        font_path (str, optional): Path to a TTF font file. If None, uses default PIL font.

    Returns:
        bytes: Byte array of the combined image in PNG format.
    """
    # Verify inputs
    if len(images) != len(labels):
        raise ValueError("Number of images must match number of labels")
    if len(images) == 0:
        raise ValueError("At least one image and label must be provided")

    # Verify all images are 1024x1024 and convert to RGB
    images = [img.convert('RGB') for img in images]
    for img in images:
        if img.size != (1024, 1024):
            raise ValueError("All images must be 1024x1024 pixels")

    # Calculate dimensions
    num_images = len(images)
    image_width, image_height = 1024, 1024
    label_height = 250  # Space for label (adjustable)
    total_width = image_width * num_images
    total_height = image_height + label_height

    # Create new blank image
    combined_image = Image.new('RGB', (total_width, total_height), (255, 255, 255))

    # Load font (use default if font_path not provided)
    try:
        font = ImageFont.truetype(font_path, 200) if font_path else ImageFont.load_default(size=200)
    except:
        font = ImageFont.load_default(size=200)

    draw = ImageDraw.Draw(combined_image)

    # Paste images and add labels
    for i, (img, label) in enumerate(zip(images, labels)):
        # Paste image
        combined_image.paste(img, (i * image_width, 0))

        # Add label
        # Calculate text position to center it below the image
        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_x = (i * image_width) + (image_width - text_width) // 2
        text_y = image_height + 10  # Small padding below image
        draw.text((text_x, text_y), label, fill=(0, 0, 0), font=font)

    # Save to byte array
    byte_io = io.BytesIO()
    combined_image.save(byte_io, format='PNG')
    return byte_io.getvalue()

# Example usage:
# from PIL import Image
# images = [Image.open("image1.jpg"), Image.open("image2.jpg"), Image.open("image3.jpg")]
# labels = ["Image 1", "Image 2", "Image 3"]
# byte_array = combine_pil_images_to_bytes(images, labels, font_path="arial.ttf")
# with open("combined_output.png", "wb") as f:
#     f.write(byte_array)