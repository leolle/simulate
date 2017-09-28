# -*- coding: utf-8 -*-
import argparse
from os import listdir, path, mkdir
import os
import zipfile
import glob


def complete_dir_path(dir_path):
    if not dir_path.endswith(r'/'):
        return dir_path + r'/'
    else:
        return dir_path


def main():
    arg_parser = argparse.ArgumentParser(
        prog='get_latest_gs', description="get & unzip latest gs.")
    arg_parser.add_argument(
        '-v', '--version', action='version', version='%(prog)s 0.1')
    arg_parser.add_argument(
        '-d',
        '--dir',
        type=str,
        default=r'\\192.168.1.100\public\GS',
        help="GS program directory.")
    arg_parser.add_argument(
        '-e',
        '--export_dir',
        type=str,
        default=r'D:\Wuwei\Software\GS',
        help="unzip output directory.")
    args = arg_parser.parse_args()

    app_dir = args.dir
    export_dir = args.export_dir
    path = r'\\192.168.1.100\public\GS'
    if not os.path.exists(export_dir):
        os.mkdir(export_dir)

    list_of_files = glob.glob(app_dir + '\*.zip')
    latest_file = max(list_of_files, key=os.path.getctime)

    zip = zipfile.ZipFile(latest_file)
    zip.extractall(export_dir)


if __name__ == '__main__':
    main()
