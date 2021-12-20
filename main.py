from src.CoreManagement import CoreComponent

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='nnet', choices=['default'], help='type of models')
    parser.add_argument('--device', type=str, default='cpu', choices=['cuda', 'cpu'], help='type of device')
    parser.add_argument('--random_seed', type=int, default=0, help='choose a random seed for our project')
    parser.add_argument('--log_to_disk', type=bool, default=False, choices=[True, False],
                        help='If you hope to get a log file after a running ended, choose this as true')

    args = parser.parse_args()
    param_dict = {'model': args.model,
                  'device': args.device,
                  'random_seed': args.random_seed,
                  'log_to_disk': args.log_to_disk}

    core_managemnet = CoreComponent(param_dict=param_dict)
    core_managemnet.initialization()
    core_managemnet.run()
    core_managemnet.kill()
