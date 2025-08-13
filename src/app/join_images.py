from PIL import Image, ImageDraw, ImageFont
import io

def combine_pil_images_to_bytes(images, labels, font_path=None, width=1024, height=1024):
    """
    Combines multiple PIL images horizontally, adds labels to the top-right corner of each image, and returns the result as a byte array.

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
    # Calculate dimensions
    num_images = len(images)
    image_width, image_height = images[0].size
    total_width = image_width * num_images
    total_height = image_height

    # Create new blank image
    combined_image = Image.new('RGB', (total_width, total_height), (255, 255, 255))

    # Load font (use default if font_path not provided)
    try:
        font = ImageFont.truetype(font_path, 100) if font_path else ImageFont.load_default(size=100)
    except:
        font = ImageFont.load_default(size=100)

    draw = ImageDraw.Draw(combined_image)

    # Paste images and add labels
    for i, (img, label) in enumerate(zip(images, labels)):
        # Paste image
        combined_image.paste(img, (i * image_width, 0))

        # Add label
        # Calculate text position for top-right corner
        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        text_x = (i * image_width) + image_width - text_width - 10  # 10px padding from right
        text_y = 10  # 10px padding from top
        draw.text((text_x, text_y), label, fill=(255, 255, 255), font=font)

    # Save to byte array
    byte_io = io.BytesIO()
    combined_image.save(byte_io, format='JPEG', quality=80, optimize=True, progressive=True)
    return byte_io.getvalue()