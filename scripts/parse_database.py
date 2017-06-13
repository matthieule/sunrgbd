import argparse

from sunrgbd.sunrgbd_dataset import SunRGBDDataset


def main(args):
    """Main"""

    database = SunRGBDDataset(
        args.path_2_database, args.path_2_toolbox, args.output_dir,
        resize=args.resize, nx=args.nx, ny=args.ny, grayscale=args.grayscale
    )
    database.curate_database()


def parse_args():
    """Parse command line arguments"""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--path_2_database', help='path to the SUNRGBD database',
        type=str
    )
    parser.add_argument(
        '--path_2_toolbox', help='path to the SUNRGBD toolbox',
        type=str
    )
    parser.add_argument(
        '--output_dir', help='Destination for the curated database',
        type=str
    )
    parser.add_argument(
        '--resize', action='store_true', help='Resize the images'
    )
    parser.add_argument(
        '--nx', help='Resized length of the x axis',
        type=int
    )
    parser.add_argument(
        '--ny', help='Resized length of the y axis',
        type=int
    )
    parser.add_argument(
        '--grayscale', action='store_true', help='Turn the images grayscale'
    )

    args = parser.parse_args()

    return args


if __name__=='__main__':
    """Main"""

    args = parse_args()
    main(args)

