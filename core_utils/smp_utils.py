import segmentation_models_pytorch as smp
import numpy as np
import cv2
import os


class Segmentation_Runner(object):

    def __init__(
        self,
        model_name='UnetPlusPlus',
        # Unet, UnetPlusPlus, MAnet, Linknet, FPN, PSPNet, PAN, DeepLabV3, DeepLabV3Plus
        # See more in https://smp.readthedocs.io/en/latest/models.html#id2
        encoder_name='resnet152',
        encoder_weights='imagenet',
        # resnet18, resnet34, resnet50, resnet101, resnet152
        # See more in https://smp.readthedocs.io/en/latest/encoders.html
        classes=10,
        movie_filename='',
    ):

        self.model_name = model_name
        self.encoder_name = encoder_name
        self.encoder_weights = encoder_weights
        self.classes = classes

        self.model_class = getattr(smp, self.model_name)(
            encoder_name=self.encoder_name,
            encoder_depth=5,
            encoder_weights=self.encoder_weights,
            decoder_use_batchnorm=True,
            decoder_channels=(256, 128, 64, 32, 16),
            decoder_attention_type=None,
            in_channels=3,
            classes=self.classes,
            activation=None,
            aux_params=None
        )

        if movie_filename != '':
            self.movie_filename = movie_filename
            self.video_handler = cv2.VideoCapture(self.movie_filename)

    def process_movie(self):

        currentframe = 0

        while(True):

            # reading from frame
            ret, frame = self.video_handler.read()

            if ret:

                # print(np.shape(frame))

                # writing the extracted images
                cv2.imshow('test', frame)

                currentframe += 1
            else:
                break

            cv2.waitKey(30)

        # Release all space and windows once done
        self.video_handler.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':

    filename = './data/out2.avi'

    segment_class = Segmentation_Runner(
        model_name='UnetPlusPlus',
        encoder_name='resnet152',
        movie_filename=filename
    )

    segment_class.process_movie()
