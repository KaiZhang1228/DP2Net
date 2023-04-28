import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Run DP2Net.")
    
    parser.add_argument('--seed', nargs='?', default='1',
                        help='Choose a seed')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='Gpu id.')

    parser.add_argument('--data_path', nargs='?', default='./dataset/polyvore_U/',
                        help='Input data path.')    
    parser.add_argument('--dataset', nargs='?', default='Polyvore_519',
                        help='Choose a dataset: Polyvore_630 or Polyvore_519.')  
    parser.add_argument('--save_dir', nargs='?', default='./model',
                        help='Model and log data saved path.')
    
    parser.add_argument('--epoch', type=int, default=100,
                        help='Number of epoch.')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='Batch size.')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate.')
    
    parser.add_argument('--input_size', type=int, default=1000,
                        help='Input item feature size (D).')
    parser.add_argument('--embed_size', type=int, default=64,
                        help='Embedding size (d).')
    parser.add_argument('--drop_m', type=int, default=0.2,
                        help='The dropout rate of user-outfit interaction matirx (M).')
    parser.add_argument('--layer_AIP_num', type=int, default=2,
                        help='Attentive information propogation layers number (l_g).')
    parser.add_argument('--layer_RU_num', type=int, default=2,
                        help='Representation updated layers number (l_s).')
    parser.add_argument('--layer_RF_num', type=int, default=2,
                        help='Representation fusion layers number (l_f).')    
    parser.add_argument('--num_heads', type=int, default=4,
                        help='Number of heads for multi-head self-attention (h).')   
    
    parser.add_argument('--lamda2', type=int, default=1,
                        help='User-specific preference perception loss rate.')
    parser.add_argument('--lamda3', type=int, default=0.2,
                        help='User-general preference perception loss rate.')
    parser.add_argument('--lamda4', type=int, default=1e-4,
                        help='L2 loss rate.')    
    
    parser.add_argument('--log_interval', type=int, default=10,
                        help='Print num.')
    parser.add_argument('--early_stopping_patience', type=int, default=5, 
                        help='Patience of decline epoch.') 
    
    return parser.parse_args()