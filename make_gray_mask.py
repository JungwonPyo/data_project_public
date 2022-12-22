from utils.detectron2_dataloader import *

if __name__ == '__main__':

    total_color_mask_path = '/media/asura/T7_Shield_1/for_mid/20221216/mask'
    total_gray_mask_path = '/media/asura/T7_Shield_1/for_mid/20221216/gray_mask'

    # make_gray_masks_from_color_masks(total_color_mask_path, total_gray_mask_path)
    make_gray_masks_from_color_masks_parallel_pool(
        total_color_mask_path, 
        total_gray_mask_path,
        16
        )
