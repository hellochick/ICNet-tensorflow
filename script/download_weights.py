from google_drive_downloader import GoogleDriveDownloader as gdd
import argparse

parser = argparse.ArgumentParser(description="Reproduced ICNet")

parser.add_argument("--dataset", type=str, default='',
                        choices=['ade20k', 'cityscapes'],
                        required=True)
args = parser.parse_args()

if args.dataset == 'cityscapes':
    ## Download cityscapes weight
    gdd.download_file_from_google_drive(file_id='15S_vZoZZwBsORxtRAMcbdsI99o6Cvo5x',
                                        dest_path='./model/cityscapes/icnet_cityscapes_train_30k_bnnomerge.npy',
                                        unzip=False)
    gdd.download_file_from_google_drive(file_id='17ZILbQ7Qazg7teb567CIPJ30FD57bVVg',
                                        dest_path='./model/cityscapes/icnet_cityscapes_train_30k.npy',
                                        unzip=False)
    gdd.download_file_from_google_drive(file_id='1Z-slNrKYJpfpELeuh2UlueQG1krF9I4a',
                                        dest_path='./model/cityscapes/icnet_cityscapes_trainval_90k_bnnomerge.npy',
                                        unzip=False)
    gdd.download_file_from_google_drive(file_id='1tZIHpppPcleamBlXKSzjOqL93gNjWGec',
                                        dest_path='./model/cityscapes/icnet_cityscapes_trainval_90k.npy',
                                        unzip=False)
elif args.dataset == 'ade20k':
    ## Download ade20k weight
    gdd.download_file_from_google_drive(file_id='1vh_JWy4lBM3A7QggQMIQLoJYZHEtUDi3',
                                        dest_path='./model/ade20k/model.ckpt-27150.data-00000-of-00001',
                                        unzip=False)
    gdd.download_file_from_google_drive(file_id='1_YAwOiBlSxu9ynlopaLQ4yjuhVTiz_-f',
                                        dest_path='./model/ade20k/model.ckpt-27150.index',
                                        unzip=False)
    gdd.download_file_from_google_drive(file_id='1ZeRnqKIoc3r6e8pezzT4xT-9FT0tudhd',
                                        dest_path='./model/ade20k/model.ckpt-27150.meta',
                                        unzip=False)
    gdd.download_file_from_google_drive(file_id='1RjIuPJ-Vhs1EuXgK9sHeT-XFoO_Sba69',
                                        dest_path='./model/ade20k/checkpoint',
                                        unzip=False)
