"""Script to plot layer activations.
"""

import random
import torch
import torchvision
import absl.flags
import absl.app
import matplotlib.pyplot as plt
import numpy as np

from src import segmentations
from src import model_utils
from src import mask_utils
from src import utils
from src import settings
from src import constants as C

# user flags
absl.flags.DEFINE_string(
    "subset", "ade20k", "subset to use. Values: [ade20k, pascal]"
)
absl.flags.DEFINE_string(
    "model",
    "resnet18",
    "model to use. Values: [resnet18, alexnet, resnet50, vgg16, densenet161]",
)
absl.flags.DEFINE_string(
    "pretrained",
    "places365",
    "whether to use pretrained weights. Values: [imagenet, places365, None]",
)
absl.flags.DEFINE_string("device", "cuda", "device to use to store the model")
absl.flags.DEFINE_integer("random_units", 0, "number of units")
absl.flags.DEFINE_string(
    "root_models", "data/model/", "root directory for models"
)
absl.flags.DEFINE_string(
    "root_datasets", "data/dataset/", "root directory for datasets"
)
absl.flags.DEFINE_string(
    "root_segmentations",
    "data/cache/segmentations/",
    "root directory for segmentations",
)
absl.flags.DEFINE_string(
    "root_activations",
    "data/cache/activations/",
    "root directory for activations",
)
absl.flags.DEFINE_string(
    "root_results", "data/results/", "root directory for results"
)

FLAGS = absl.flags.FLAGS


def main(argv):
    generator = utils.set_seed(0)

    # Parameters
    cfg = settings.Settings(
        subset=FLAGS.subset,
        model=FLAGS.model,
        pretrained=FLAGS.pretrained,
        num_clusters=1,
        beam_limit=1,
        device=FLAGS.device,
        root_models=FLAGS.root_models,
        root_datasets=FLAGS.root_datasets,
        root_segmentations=FLAGS.root_segmentations,
        root_activations=FLAGS.root_activations,
        root_results=FLAGS.root_results
    )
    sparse_segmentation_directory = cfg.get_segmentation_directory()
    mask_shape = cfg.get_mask_shape()

    # Load data
    dataset = segmentations.BrodenDataset(
        cfg.dir_datasets,
        subset=cfg.index_subset,
        resolution=cfg.get_img_size(),
        broden_version=1,
        transform_image=torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(cfg.get_img_size()),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    cfg.get_image_mean(), cfg.get_image_stdev()
                ),
            ]
        ),
    )
    segmentation_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=C.BATCH_SIZE,
        worker_init_fn=utils.seed_worker,
        generator=generator,
    )

    # Load Model
    model = model_utils.load_model_from_settings(cfg, device=cfg.device)

    # Loop over all the selected layers
    for _, layer_name in enumerate(cfg.get_feature_names()):
        # Get the number of units in the layer
        num_units = model_utils.get_number_of_units(model, layer_name, cfg)
        
        layer_str = model_utils.get_layer_str(layer_name)

        # Get activations
        activations = model_utils.get_layer_activations(
            segmentation_loader,
            model,
            layer_name,
            range(num_units),
            cfg.get_activation_directory(),
        )

        if torch.all(activations >= 0):
            activations = activations[activations > 0]

        x_min, x_max = np.percentile(activations, [0.1, 99.9])

        # Plot activations
        plt.hist(activations.flatten().cpu().numpy(), bins=1000)
        plt.xlabel("Activation")
        plt.ylabel("Frequency")
        plt.title(f"{cfg.model} Total Activations for {layer_str}")
        plt.xlim(x_min, x_max)
        plt.show()

        # Select units
        if FLAGS.random_units == 0:
            selected_units = range(num_units)
        else:
            selected_units = random.sample(
                range(num_units), FLAGS.random_units
            )

        for unit in selected_units:
            unit_activations = activations[unit]

            x_min, x_max = np.percentile(unit_activations, [0.1, 99.9])

            # Plot unit activations
            plt.hist(unit_activations.flatten().cpu().numpy(), bins=1000)
            plt.xlabel("Activation")
            plt.ylabel("Frequency")
            plt.title(f"Unit {unit} Activations")
            plt.xlim(x_min, x_max)
            plt.show()


if __name__ == "__main__":
    with torch.no_grad():
        absl.app.run(main)