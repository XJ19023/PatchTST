import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    '--smooth_module',
    nargs='+',
    default=['qkv', 'to_out', 'ff.0', 'ff.3'],
    help='Modules to apply smooth'
)
parser.add_argument('--random_seed', type=int, default=2021, help='random seed')
args = parser.parse_args()

print(args.smooth_module)
print(args.random_seed)