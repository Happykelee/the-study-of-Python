import os
from PIL import Image, ImageDraw, ImageFont

class AddNum():
    def __init__(self):
        self.img = None
        self.num = None
        self.img_path = None

    def open(self, img_path):
        self.img = Image.open(img_path)
        self.img_path = os.path.abspath(img_path)
        # return True

    def draw(self, num = 1):
        # --Initialize the number--
        num = int(num)
        self.num = num
        num_str = str(num) if num < 100 else '99+'

        # --Draw the number--
        font_size = max(self.img.size[0], self.img.size[1]) // 5
        # img_width = self.img.size[0];img_height = self.img.size[1]
        font = ImageFont.truetype("arial.ttf", font_size)
        text_x = self.img.size[0] - font.getsize(num_str)[0]
        text_y = 0
        text_color = (255, 0, 0)
        draw = ImageDraw.Draw(self.img)
        draw.text(
            xy = (text_x, text_y),
            text = num_str,
            fill = text_color,
            font =  font)

        # --Save the new picture--
        left, right = self.img_path.rsplit(".", 1)
        new_img_path = left + "_" + num_str + "." + right
        self.img.save(new_img_path)
        # return True

if __name__ == '__main__':
    solver = AddNum()
    solver.open('./02_sample.jpg')
    solver.draw(4)
