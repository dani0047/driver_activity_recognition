from PIL import Image, ImageTk, ImageDraw

def create_rounded_rectangle_mask(width, height, radius):
    """Create a mask with rounded corners."""
    mask = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(mask)
    draw.rounded_rectangle((0, 0, width, height), radius, fill=255)
    return mask

def masked_image(image, mask_dim = [320, 240, 10]):
    npimg = Image.fromarray(image)
    mask = create_rounded_rectangle_mask(mask_dim[0], mask_dim[1], mask_dim[2])
    npimg.putalpha(mask)
    masked_img = ImageTk.PhotoImage(image=npimg)
    return masked_img



