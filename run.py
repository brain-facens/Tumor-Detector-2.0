# libraires
from src.interface import Screen

import argparse
import os



# init variables
PATH = os.path.dirname(os.path.realpath(__file__))
VERSIONS = ['v1']



# init arguments
parser = argparse.ArgumentParser()
parser.add_argument('--screen_size', '-sz', type=int, default=1, help='Specify the size of screen. Options available: [0: small, 1: medium, 2: larger]')
parser.add_argument('--weights', '-w', type=str, default='v1', help='Specify which version of model do you want to use.')
parser.add_argument('--device', type=str, default='gpu', help='Switch the device to be used. Options available: [cpu, gpu].')
arg = parser.parse_args()


# validating init arguments
assert arg.screen_size in [0, 1, 2], '--screen_size | -sz must be one of them: [0: small, 1: medium, 2: larger]'
assert arg.weights in VERSIONS, f'--weights | -w must be available to be used. Version availables: {VERSIONS}'
assert arg.device in ['cpu', 'gpu'], '--device must be one of them: [cpu, gpu]'


if __name__=='__main__':
    Screen(
        window_size=arg.screen_size,
        version=arg.weights,
        device=arg.device
    ).mainloop()