import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-e", "--env_id", help="Environment name for your experiments",
                        default="CartPole-v1", type=str, required=False)
    parser.add_argument("-s", "--seed", help="Seed to your experiments",
                        default=777, type=int, required=False)
    parser.add_argument("-t", "--global_steps", help="Global timesteps to pass",
                        default=1_000_000, type=int, required=False)
    parser.add_argument("-g", "--gamma", help="Discount for return",
                        default=0.99, type=float, required=False)
    parser.add_argument("-n", "--num_envs", help="Number of parallel environments",
                        default=2, type=int, required=False)
    parser.add_argument("--track", action="store_true", help="Whether to track wandb")
    parser.add_argument("--project_name", help="Name of the project on wandb",
                        default="JimmyRL", type=str, required=False)
    parser.add_argument("--entity", help="Wandb entity",
                        default="myxik", type=str, required=False)
    return parser.parse_args()
