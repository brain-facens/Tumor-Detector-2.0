# libraries
from utils import load_model, image_dataset, applyMask, estimateResults, tf
from PIL import Image, ImageTk
from tqdm import tqdm

import tkinter as tk
import numpy as np


# configurations
SIZES = {
    0: {
        'window_size': '750x500',
        'icon_size': (50,50),
        'text_size': 10,
        'image_size': (400,400)
    },
    1: {
        'window_size': '1050x700',
        'icon_size': (65,65),
        'text_size': 12,
        'image_size': (600,600)
    },
    2: {
        'window_size': '1400x950',
        'icon_size': (80, 80),
        'text_size': 14,
        'image_size': (800,800)
    }
}
BACKGROUND_COLOR = '#272727'
STYLE_COLOR_1 = '#00D3FD'
STYLE_COLOR_2 = '#00D1AE'
TEXT_COLOR_1 = '#FFFFFF'
TEXT_COLOR_2 = '#000000'
IMAGE_TEST_FOLDER = 'data'



# main screen
class Screen(tk.Tk):

    def __init__(self, window_size: int, version: str, device: str, *args, **kwargs) -> None:
        tk.Tk.__init__(self, *args, **kwargs)


        # variables
        self.__current_index = 0


        # changing device
        if device == 'cpu':
            tf.config.set_visible_devices([], 'GPU')


        # loading model and images
        __model = load_model(version=version)
        __images = image_dataset(IMAGE_TEST_FOLDER)
        self.__data = []
        self.__labels = []
        self.__predict = []
        # inference
        for iter, (__image, __mask) in enumerate(tqdm(iterable=__images, desc='Running inferences')):
            # converting images from tensor to pillow
            __pred = np.asarray(tf.keras.utils.array_to_img(
                        x=__model(__image)[0], data_format='channels_last', dtype='float32'
                    ).resize(SIZES[window_size]['image_size'])) * 255
            __image = np.asarray(tf.keras.utils.array_to_img(
                        x=__image[0], data_format='channels_last', dtype='float32'
                    ).resize(SIZES[window_size]['image_size']))
            __mask = np.asarray(tf.keras.utils.array_to_img(
                x=__mask[0], data_format='channels_last', dtype='float32'
            ).resize(SIZES[window_size]['image_size']))

            if __pred.sum() > 500000:
                self.__predict.append(1)
            else:
                self.__predict.append(0)
            
            if __mask.sum():
                self.__labels.append(1)
            else:
                self.__labels.append(0)

            # overlapping results
            __segmentation = applyMask(__image, __mask, __pred)
            self.__data.append(ImageTk.PhotoImage(image=Image.fromarray(__segmentation)))
            if iter > 30:
                break
        self.__n_elements = self.__data.__len__()
        # getting data to be displayed
        __precision, __recall, __accuracy = estimateResults(self.__labels, self.__predict)
        self.__current_image = self.__data[self.__current_index]

        # setting up configurations of main screen
        # title
        self.wm_title(string='Tumor Detector 2.0 - Application')
        # size
        self.geometry(newGeometry=SIZES[window_size]['window_size'])
        self.resizable(width=False, height=False)
        # background color
        self.config(background=BACKGROUND_COLOR)
        # logo
        # - loading image
        __brain_image = Image.open(fp='docs/images/brain_logo.png')
        # - resizing image
        __brain_image = __brain_image.resize(size=SIZES[window_size]['icon_size'])
        # - applying image
        __brain_image = ImageTk.PhotoImage(image=__brain_image)
        __brain_frame = tk.Label(self, image=__brain_image)
        __brain_frame.image = __brain_image
        # - logo localization
        __brain_frame.place(relx=0.4, rely=0.0, relwidth=0.2, relheight=0.1)
        # - background color
        __brain_frame.config(background=BACKGROUND_COLOR)

        # setting up configurations of image frame
        self.__image_frame = tk.Label(master=self, image=self.__current_image)
        # style
        self.__image_frame.config(
            background='black',
            highlightthickness=2,
            highlightbackground=STYLE_COLOR_1
        )
        # localization
        self.__image_frame.place(relx=0.01, rely=0.1, relwidth=0.6, relheight=0.85)
        # -- buttons
        button_next = tk.Button(master=self.__image_frame)
        button_previous = tk.Button(master=self.__image_frame)
        # -- buttons style
        button_previous.config(
            text='<<',
            fg=TEXT_COLOR_2,
            background=STYLE_COLOR_2,
            font=('Arial', SIZES[window_size]['text_size'], 'bold'),
            command=self.previous_next
        )
        button_next.config(
            text='>>',
            fg=TEXT_COLOR_2,
            background=STYLE_COLOR_2,
            font=('Arial', SIZES[window_size]['text_size'], 'bold'),
            command=self.next_image
        )
        # -- buttons localization
        button_next.place(relx=0.94, rely=0.94, relwidth=0.05, relheight=0.05)
        button_previous.place(relx=0.01, rely=0.94, relwidth=0.05, relheight=0.05)

        # setting up configurations of settings frame
        self.__informations_frame = tk.Frame(master=self)
        # style
        self.__informations_frame.config(
            background=BACKGROUND_COLOR,
            highlightthickness=2,
            highlightbackground=STYLE_COLOR_1
        )
        # localization
        self.__informations_frame.place(relx=0.65, rely=0.1, relwidth=0.3, relheight=0.85)
        # text
        self.__informations_text_1 = tk.Label(master=self.__informations_frame)
        # - style
        self.__informations_text_1.config(
            text='Informations',
            fg=TEXT_COLOR_1,
            background=BACKGROUND_COLOR,
            font=('Arial', SIZES[window_size]['text_size'], 'bold')
        )
        # - localization
        self.__informations_text_1.place(relx=0.01, rely=0.01, relwidth=0.978, relheight=0.1)
        # - setting up configurations of results frame
        self.__results_frame = tk.Label(master=self.__informations_frame)
        # -- style
        self.__results_frame.config(
            text=f'Precision: {__precision}\nRecall: {__recall}\nAccuracy: {__accuracy}',
            fg=TEXT_COLOR_1,
            background=BACKGROUND_COLOR,
            font=('Arial', SIZES[window_size]['text_size']-2, 'bold')
        )
        # -- localization
        self.__results_frame.place(relx=0.01, rely=0.2, relwidth=0.95, relheight=0.3)

    def next_image(self) -> None:
        self.__current_index += 1
        if self.__current_index == self.__n_elements:
            self.__current_index = 0
        self.__image_frame.configure(image=self.__data[self.__current_index])
        self.__image_frame.image = self.__data[self.__current_index]

    def previous_next(self) -> None:
        self.__current_index -= 1
        if self.__current_index < 0:
            self.__current_index = self.__n_elements - 1
        self.__image_frame.configure(image=self.__data[self.__current_index])
        self.__image_frame.image = self.__data[self.__current_index]