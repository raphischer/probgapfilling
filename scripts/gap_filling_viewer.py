"""viewer application which allows to interactively view spatio-temporal gap filling results"""
import os
import argparse
from datetime import datetime, timedelta
from tkinter import Canvas, Tk, Button, RAISED, DISABLED, SUNKEN, NORMAL
import numpy as np
from PIL import Image, ImageTk
import probgf.media as media


class MainWindow():


    def next(self, event=None):
        self.curr_img = (self.curr_img + 1) % len(self.imgs_orig)
        self.refresh()


    def prev(self, event=None):
        self.curr_img = (self.curr_img - 1) % len(self.imgs_orig)
        self.refresh()


    def click_wheel(self, event):
        self.start_drag = (event.x + self.shift_x, event.y + self.shift_y)


    def click_left(self, event):
        if not self.over_button:
            self.prev()


    def click_right(self, event):
        if not self.over_button:
            self.next()


    def refresh(self):
        zoom = float(self.zoom) / 100
        self.start_x = int(self.img_w_f / 2 - self.img_w_f / zoom / 2) + self.shift_x
        self.end_x = int(self.start_x + self.img_w_f / zoom)
        self.start_y = int(self.img_w_f / 2 - self.img_w_f / zoom / 2) + self.shift_y
        self.end_y = int(self.start_y + self.img_w_f / zoom)
        if not self.mask_toggle:
            self.b_masks.config(relief=RAISED)
            img1 = self.imgs_orig[self.curr_img]
            img2 = self.imgs_pred[self.curr_img]
        else:
            self.b_masks.config(relief=SUNKEN)
            img1 = self.imgs_orig_m[self.curr_img]
            img2 = self.imgs_pred_m[self.curr_img]
        img1 = img1.crop((self.start_x, self.start_y, self.end_x, self.end_y)).resize((self.img_w, self.img_w), Image.ANTIALIAS)
        img2 = img2.crop((self.start_x, self.start_y, self.end_x, self.end_y)).resize((self.img_w, self.img_w), Image.ANTIALIAS)
        self.imgs_orig_v[self.curr_img] = ImageTk.PhotoImage(img1)
        self.imgs_pred_v[self.curr_img] = ImageTk.PhotoImage(img2)
        self.canvas.itemconfig(self.i_left, image = self.imgs_orig_v[self.curr_img])
        self.canvas.itemconfig(self.i_right, image = self.imgs_pred_v[self.curr_img])
        self.canvas.itemconfig(self.i_list, image = self.imagelists[self.curr_img])
        self.canvas.itemconfig(self.day_info, text='{} - cloud cover {:06.2f}% - estimated MAE {}'.format(self.dates[self.curr_img],
                                                                                                          self.cc[self.curr_img] * 100,
                                                                                                          self.errors[self.curr_img]))
        if self.zoom == 100:
            self.canvas.itemconfig(self.zoom, text='')
            self.b_reset.config(state=DISABLED)
        else:
            self.canvas.itemconfig(self.zoom, text='ZOOM: {:3d}%'.format(self.zoom))
            self.b_reset.config(state=NORMAL)


    def zoomer(self, event):
        if event.num == 4 or event.delta == 120 or event.keysym == 'plus':
            self.zoom += 20
        elif event.delta == 240:
            self.zoom += 40
        elif event.delta == 360:
            self.zoom += 60
        else:
            if self.zoom - 20 >= 100:
                self.zoom -= 20
                if self.zoom == 100:
                    self.reset_transform()
        self.refresh()


    def drag_roi(self, event):
        self.shift_x = min(max(self.start_drag[0] - event.x, 0 - int(self.img_w_f / 2 - self.img_w_f / self.zoom / 2)), 
                           int(self.img_w_f / 2 - self.img_w_f / self.zoom / 2))
        self.shift_y = min(max(self.start_drag[1] - event.y, 0 - int(self.img_w_f / 2 - self.img_w_f / self.zoom / 2)),
                           int(self.img_w_f / 2 - self.img_w_f / self.zoom / 2))
        self.refresh()


    def toggle_mask(self, event=None):
        self.mask_toggle = not self.mask_toggle
        self.refresh()


    def reset_transform(self, event=None):
        self.mask_toggle = False
        self.zoom = 100
        self.shift_x = 0
        self.shift_y = 0
        self.refresh()


    def button_enter(self, event):
        self.over_button = True


    def button_leave(self, enter):
        self.over_button = False


    def __init__(self, root, w, h, imgs_p, imgs_o, imgs_m, dates, errors, logos):
        self.dates = dates
        self.errors = errors
        # setup images
        self.img_w = int(h * 0.68) # width of each displayed image
        self.imgs_orig_m = [] # masked full images
        self.imgs_pred_m = []
        self.imgs_orig = [] # unmasked full images
        self.imgs_pred = []
        self.cc = []
        for index, img in enumerate(imgs_p):
            self.imgs_orig.append(imgs_o[index].resize((self.img_w, self.img_w), resample=0))
            self.imgs_pred.append(img.resize((self.img_w, self.img_w), resample=0))
            self.imgs_orig_m.append(Image.blend(self.imgs_orig[-1], imgs_m[index].convert(mode='RGB').resize((self.img_w, self.img_w), resample=0), alpha=.5))
            self.imgs_pred_m.append(Image.blend(self.imgs_pred[-1], imgs_m[index].convert(mode='RGB').resize((self.img_w, self.img_w), resample=0), alpha=.5))
            self.cc.append(1 - np.count_nonzero(np.array(imgs_m[index])) / np.array(imgs_m[index]).size)
        self.curr_img = 0
        # text labels and logos
        h_logos = int(h / 17)
        b_logos = int(w / 100)
        self.canvas = Canvas(root, width=w, height=h)
        self.canvas.pack()
        self.canvas.configure(background='white')
        self.logo1 = ImageTk.PhotoImage(logos[0].resize((int(h_logos / logos[0].size[1] * logos[0].size[0]), h_logos), Image.ANTIALIAS))
        self.logo2 = ImageTk.PhotoImage(logos[1].resize((int(h_logos / logos[1].size[1] * logos[1].size[0]), h_logos), Image.ANTIALIAS))
        self.logo3 = ImageTk.PhotoImage(logos[2].resize((int(h_logos / logos[2].size[1] * logos[2].size[0]), h_logos), Image.ANTIALIAS))
        self.canvas.create_image(int(self.logo1.width() / 2 + b_logos), int(self.logo1.height() / 2 + b_logos), image=self.logo1)
        self.canvas.create_image(int(w - self.logo2.width() / 2 - b_logos), int(self.logo2.height() / 2 + b_logos), image=self.logo2)
        self.canvas.create_image(int(w - self.logo3.width() / 2 - b_logos), int(h - (self.logo3.height() / 2 + b_logos)), image=self.logo3)
        self.canvas.create_text(w / 2, h * 0.06, font=("Courier", int(h / 25)), text='Gap Filling Viewer')
        self.canvas.create_text(w / 3.9, h * 0.19, font=("Courier", int(h / 35)), text='Observed')
        self.canvas.create_text(w - w / 3.9, h * 0.19, font=("Courier", int(h / 35)), text='Predicted')
        self.day_info = self.canvas.create_text(w / 2, h * 0.13, font=("Courier", int(h / 30)), text='')
        self.zoom = self.canvas.create_text(w * 0.12, h * 0.94, font=("Courier", int(h / 50)), text='')
        # image timeline
        imagelist_h = int(self.img_w / len(self.imgs_pred)) + 1
        imagelist_a = np.zeros((len(self.imgs_pred), imagelist_h, imagelist_h, 3), dtype='uint8')
        for index in range(len(self.imgs_pred)):
            imagelist_a[index, :, :, :] = np.array(self.imgs_pred[index].resize((imagelist_h, imagelist_h), Image.ANTIALIAS))
        self.imagelists = []
        for index in range(len(self.imgs_pred)):
            c_list = np.array(imagelist_a)
            c_list[index, :int(w / 600), :, :] = 255
            c_list[index, (imagelist_h - int(w / 600)):, :, :] = 255
            c_list[index, :, :int(w / 600), :] = 255
            c_list[index, :, (imagelist_h - int(w / 600)):, :] = 255
            self.imagelists.append(ImageTk.PhotoImage(Image.fromarray(c_list.reshape(len(self.imgs_pred) * imagelist_h, imagelist_h, 3))))
        self.i_list = self.canvas.create_image(w * 0.5, h * 0.56, image=self.imagelists[self.curr_img])
        # images and buttons
        self.img_w_f = self.imgs_orig[0].size[0] # full image width
        self.imgs_orig_v = [ImageTk.PhotoImage(img.resize((self.img_w, self.img_w), Image.ANTIALIAS)) for img in self.imgs_orig] # images for visualization
        self.imgs_pred_v = [ImageTk.PhotoImage(img.resize((self.img_w, self.img_w), Image.ANTIALIAS)) for img in self.imgs_pred]
        self.i_left = self.canvas.create_image(w / 3.9, h * 0.56, image=self.imgs_orig_v[self.curr_img])
        self.i_right = self.canvas.create_image(w - w / 3.9, h * 0.56, image=self.imgs_pred_v[self.curr_img])
        self.b_masks = Button(root, font=("Courier", int(h / 50)), text = "Show masks", command=self.toggle_mask)
        self.b_reset = Button(root, font=("Courier", int(h / 50)), text = "Reset view", command=self.reset_transform, state=DISABLED)
        self.b_quit = Button(root, font=("Courier", int(h / 50)), text = "Quit", command=self.canvas.master.destroy)
        self.reset_transform()
        self.canvas.create_window(w * 0.30, h * 0.94, window=self.b_masks)
        self.canvas.create_window(w * 0.50, h * 0.94, window=self.b_reset)
        self.canvas.create_window(w * 0.70, h * 0.94, window=self.b_quit)
        # bind buttons and keys
        root.bind("q", lambda e: self.canvas.master.destroy())
        root.bind("r", self.reset_transform)
        root.bind("m", self.toggle_mask)
        root.bind("<Right>", self.next)
        root.bind("<Left>", self.prev)
        root.bind("<Down>", self.next)
        root.bind("<Up>", self.prev)
        root.bind("<Button-3>", self.click_right)
        root.bind("<Button-1>", self.click_left)
        root.bind("<Button-2>", self.click_wheel)
        root.bind("<Button-4>", self.zoomer)
        root.bind("<Button-5>", self.zoomer)
        root.bind("<MouseWheel>", self.zoomer)
        root.bind("<B2-Motion>", self.drag_roi)
        root.bind("+", self.zoomer)
        root.bind("-", self.zoomer)
        self.over_button = False
        self.b_masks.bind("<Enter>", self.button_enter)
        self.b_masks.bind("<Leave>", self.button_leave)
        self.b_reset.bind("<Enter>", self.button_enter)
        self.b_reset.bind("<Leave>", self.button_leave)
        self.b_quit.bind("<Enter>", self.button_enter)
        self.b_quit.bind("<Leave>", self.button_leave)


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('-l', '--left', default='imgs/original/',
                    help='directory with images which are shown on the left')
parser.add_argument('-r', '--right', default='imgs/pred_outline_lin_spatial_clouds0_2/',
                    help='directory with images which are shown on the right')
parser.add_argument('-m', '--masks', default='imgs/mask/',
                    help='directory with mask images')
parser.add_argument('-R', '--report', default='report_lin_spatial_clouds0_2.csv',
                    help='report containing date and error information for the right hand images')
parser.add_argument('-y', '--year', type=int, default=2018,
                    help='year of data acquisition')
parser.add_argument('-W', '--width', type=int, default=1280,
                    help='window width')
parser.add_argument('-H', '--height', type=int, default=720,
                    help='window height')

args = parser.parse_args()

imgs_o = [Image.open(img) for img in sorted([os.path.join(args.left, img) for img in os.listdir(args.left)])]
imgs_p = [Image.open(img) for img in sorted([os.path.join(args.right, img) for img in os.listdir(args.right)])]
imgs_m = [Image.open(img) for img in sorted([os.path.join(args.masks, img) for img in os.listdir(args.masks)])]

report = np.genfromtxt(args.report, delimiter=',', dtype=float)[1:-1]
dates = [(datetime(args.year, 1, 1) + timedelta(int(report[day, 1]) - 1)).strftime('%b %d %Y') for day in range(report.shape[0])]
errors = ['{:4.1f}'.format(error) if error != 0.0 else 'n.a. ' for error in report[:, 5]]
logos = [media.logo1, media.logo2, media.logo3]

if len(imgs_o) != len(dates):
    raise RuntimeError('Different number of images in {} than days in the report {}!'.format(args.left, args.report))
if len(imgs_p) != len(dates):
    raise RuntimeError('Different number of images in {} than days in the report {}!'.format(args.right, args.report))
if len(imgs_m) != len(dates):
    raise RuntimeError('Different number of images in {} than days in the report {}!'.format(args.masks, args.report))

root = Tk()
root.title('Gap Filling Viewer')
root.geometry("%dx%d+0+0" % (args.width, args.height))
MainWindow(root, args.width, args.height, imgs_p, imgs_o, imgs_m, dates, errors, logos)
root.focus_set()
root.mainloop()
