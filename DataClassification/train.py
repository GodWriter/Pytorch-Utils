import os

from models import Vgg


def main(args):
    model = Vgg(args)

    dataset = DataLoader(args.train_path)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=opt.batch_size,
                                             shuffle=True,
                                             num_workers=0,
                                             pin_memory=False)
    
    for batch_i, (img_path, images, labels) in enumerate(dataloader):
        print("batch_i: ", batch_i)
        print("images: ", images)
        print("labels: ", labels)

        print("=" * 50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=3, help="size of each image batch")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--log", type=str, default="logs/Vgg")
    parser.add_argument("--train_path", type=str, default="config/train.txt", help="txt path saving image paths")
    args = parser.parse_args()
    print(args)

    main(args)