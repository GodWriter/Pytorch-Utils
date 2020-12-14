import argparse


def parse_args():
    """
    parsing and configuration
    :return: parse_args
    """
    desc = "Pytorch implementation of Unit"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--epoch', type=int, default=0, help="start epochs")
    parser.add_argument('--n_epochs', type=int, default=200, help="training epochs")
    parser.add_argument('--batch_size', type=int, default=4, help='batch size')
    parser.add_argument('--dataset', type=str, default='data/coco2014', help='path of dataset')
    parser.add_argument('--style_img', type=str, default='images/style.jpg', help='path of style image')
    parser.add_argument('--style_name', type=str, default='style', help='path to save checkpoints')
    parser.add_argument('--lr', type=float, default=0.0001, help='adam: learning rate')
    parser.add_argument('--decay_epoch', type=int, default=50, help='epoch from which to start lr decay')
    parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads during batch generation')
    parser.add_argument('--img_height', type=int, default=256, help='size of image height')
    parser.add_argument('--img_width', type=int, default=256, help='size of image width')
    parser.add_argument('--channels', type=int, default=3, help='number of image channels')
    parser.add_argument('--lambda_adv', type=float, default=10.0, help='adversarial loss weight')
    parser.add_argument('--lambda_st', type=float, default=100, help='style transfer loss weight')
    parser.add_argument('--lambda_style', type=float, default=100.0, help='style loss weight')
    parser.add_argument('--n_downsample', type=int, default=2, help='number downsampling layers in encoder')
    parser.add_argument('--n_upsample', type=int, default=2, help='number sampling layers in decoder')
    parser.add_argument('--n_residual', type=int, default=9, help='number residual block')
    parser.add_argument('--dim', type=int, default=64, help='number of filters in first encoder layer')
    parser.add_argument('--sample_interval', type=int, default=100, help='interval between image samples')
    parser.add_argument('--checkpoint_interval', type=int, default=5000, help='interval between saving models')
    parser.add_argument('--load_model', type=str, default='checkpoints/apple2orange/*_done.pth', help='model to load')
    parser.add_argument('--test_img', type=str, default='images/test.jpg', help='image to test')

    opt = parser.parse_args()
    print(opt)

    return opt
