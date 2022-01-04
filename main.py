from src.CoreManagement import CoreComponent

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='default', choices=['default'], help='type of models')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='type of device')
    parser.add_argument('--random_seed', type=int, default=11, help='choose a random seed for our project')
    parser.add_argument('--log_to_disk', type=bool, default=False, choices=[True, False],
                        help='If you hope to get a log file after a running ended, choose this as true')
    parser.add_argument('--run_type', type=str, default='both',
                        choices=['fractal_generation', 'style_generation', 'style_transfer', 'both'],
                        help='Only run fractal generation, or style transfer, or both. style_generation is deprecated')

    args = parser.parse_args()
    param_dict = {'model': args.model,
                  'device': args.device,
                  'random_seed': args.random_seed,
                  'log_to_disk': args.log_to_disk,
                  'run_type': args.run_type}

    core_managemnet = CoreComponent(param_dict=param_dict)
    core_managemnet.initialization()
    core_managemnet.run()
    core_managemnet.kill()
