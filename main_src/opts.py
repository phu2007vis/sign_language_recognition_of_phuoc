import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    '--model_path',
    type=str,
    default='model/model_rgb.pth',
    help='Path to rgb model state_dict')

parser.add_argument(
    '--folder_data_path',
    type=str,
    default='data',
    help='Path to kinetics rgb numpy sample')

# Flow arguments
parser.add_argument(
    '--flow', action='store_true', help='Evaluate flow pretrained network')
parser.add_argument(
    '--flow_weights_path',
    type=str,
    default='model/model_flow.pth',
    help='Path to flow model state_dict')

# Class argument
parser.add_argument(
    '--classes_path',
    type=str,
    default='data/classes.txt',
    help='Path of the file containing classes names')

# Sample arguments
parser.add_argument(
    '--sample_num',
    type=int,
    default='16',
    help='The number of the output frames after the sample, or 1/sample_rate frames will be chosen.')

parser.add_argument(
    '--out_fps',
    type=int,
    default='5',
    help='The fps of the output video.')

parser.add_argument(
    '--out_frame_num',
    type=int,
    default='32',
    help='The number of frames sent to model finally.')
