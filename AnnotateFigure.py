import os
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw, ImageFont


def add_text_box(img, text, position, font_size=18, box_padding=10, box_color="black", text_color="white"):
    """
    Adds a text box with text to the image.

    Parameters:
    - img: The PIL Image object.
    - text: The text to be added inside the box.
    - position: The (x, y) tuple indicating the top-left corner where the text box starts.
    - font_size: The size of the font (default is 30).
    - box_padding: The padding around the text inside the box (default is 10).
    - box_color: The color of the background rectangle (text box) (default is "lightblue").
    - text_color: The color of the text (default is "black").

    Returns:
    - The PIL Image object with the text box added.
    """

    font_path = "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"

    # Create ImageDraw object to draw on the image
    draw = ImageDraw.Draw(img)

    # Load a custom font
    font = ImageFont.truetype(font_path, size=font_size)

    # Calculate text bounding box (x1, y1, x2, y2)
    bbox = draw.textbbox((0, 0), text, font=font)

    # Calculate text width and height from the bounding box
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    # Define the rectangle coordinates based on the text size and padding
    box_coordinates = (position[0] - box_padding, position[1] - box_padding, position[0] + text_width + box_padding,
                       position[1] + text_height + box_padding)

    # Draw the rectangle (background for the text box)
    draw.rectangle(box_coordinates, fill=box_color)

    # Draw the text inside the box
    draw.text(position, text, fill=text_color, font=font)

    return img


working_dir = os.getcwd()

# Load the image
img = Image.open(os.path.join(working_dir, 'plots', 'Female Regions with significantly altered power in post-COVID rsMEG gamma band.png'))

img_array = np.array(img)

img_1d = img_array[:, :, 1]

# Define the region you want to move (x1, y1, x2, y2)
region_box = (0, 106, 799, 692)  # adjust these values
region = img.crop(region_box)

# Define where you want to paste it (new top-left corner)
new_position = (0, 170)  # x, y position lower down

# Blank out the original area (fill with background color)
draw = Image.new('RGB', (region_box[2] - region_box[0], region_box[3] - region_box[1]), color=(0, 0, 0))
img.paste(draw, (region_box[0], region_box[1]))

# Paste the cropped region into the new location
img.paste(region, new_position)

# Add text box
add_text_box(img, 'Right Hemisphere Lateral', [35, 130])
add_text_box(img, 'Right Hemisphere Medial', [550, 130])
add_text_box(img, 'Left Hemisphere Lateral', [35, 450])
add_text_box(img, 'Left Hemisphere Medial', [550, 450])

# View image
# Convert the image to a NumPy array
img_array = np.array(img)
plt.figure(figsize=(10, 10))
plt.imshow(img_array)
plt.axis('off')  # Turn off axis for a clean view
plt.show()

# Save the result
# img.save("moved_image.png")