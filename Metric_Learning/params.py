import argparse


parser = argparse.ArgumentParser()

parser.add_argument('--train_json', type=str, default='./configs/train_covid19_new.json')
parser.add_argument('--test_json', type=str, default='./configs/test_covid19_new.json')
parser.add_argument('--weight_dir', type=str, default='./weights/')
parser.add_argument('--experiment_name', type=str, default='resnet50')

parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--n_epochs', type=int, default=100)

parser.add_argument('--model_name', type=str, default='resnet50')
parser.add_argument('--freeze_encoder', type=int, default=1)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight_decay', type=float, default=1e-4)

parser.add_argument('--img_size', type=int, default=224)

parser.add_argument('--n_splits', type=int, default=5)

parser.add_argument('--output_dim', type=int, default=256)
parser.add_argument('--n_classes', type=int, default=2)

args, unknown = parser.parse_known_args()
# args = parser.parse_args()