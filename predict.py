import sys
import os
import argparse
import torch
import logging
import time
from tqdm import tqdm
from image import *
from Models.lightUnetPlusPlus import lightUnetPlusPlus


def predict(model,
            threshold,
            device,
            dataset,
            output_paths,
            color):

    with tqdm(desc=f'Prediction', unit=' img') as progress_bar:
        for i, (image, _) in enumerate(dataset):
            image = image[0, ...]
            image = image.to(device)
            with torch.no_grad():
                mask_predicted = model(image)
            placeholder_path(output_paths[i])
            save_predicted_mask(mask_predicted, device, color=color, filename=(output_paths[i]+"/predicted.png"), threshold=threshold)
            progress_bar.update()


if __name__ == '__main__':
    t_start = time.time()
    current_path = sys.argv[0]
    current_path = current_path.replace("predict.py", "")


    # Hyperparameters
    batch_size = 1
    num_classes = 2
    n_channels = 6

    # Arg parse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i")
    parser.add_argument("--output", "-o")                        
    parser.add_argument("--threshold", "-t", type=float)
    parser.add_argument("--color", "-c") # red blue black

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cpu' if not torch.cuda.is_available() else 'cuda')
    logging.info(f'Using {device}')

    instance_names = [i for i in os.walk(args.input)][0][1]
    dataset, output_paths = load_dataset_predict(args.input, args.output, instance_names, batch_size)

    logging.info(f'Data loaded : {len(output_paths)} instances found')

    # Network creation, uncomment the one you want to use
    # model = BasicUnet(n_channels= n_channels, n_classes=num_classes)
    # model = modularUnet(n_channels=n_channels, n_classes=num_classes, depth=2)
    # model = unetPlusPlus(n_channels=n_channels, n_classes=num_classes)
    model = lightUnetPlusPlus(n_channels=n_channels, n_classes=num_classes)
    model.to(device)
    model.load_state_dict(torch.load('Weights/last.pth',map_location=torch.device(device)))
    model.eval()
    logging.info(f'Model loaded\n')

    try:
        predict(model=model,
                threshold=args.threshold,
                device=device,
                dataset=dataset,
                output_paths=output_paths,
                color=args.color)


    except KeyboardInterrupt:
        logging.info(f'Interrupted by Keyboard')
    finally:
        t_end = time.time()
        print("\nDone in " + str(int((t_end - t_start))) + " sec")