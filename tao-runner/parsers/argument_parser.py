import sys
from argparse import ArgumentParser, SUPPRESS, Namespace
from typing import List


class Parser(object):
    """Argument parser that can handle arguments with our special
    placeholder.
    """

    def __init__(self):
        self._parser = ArgumentParser(
            prog='tao-runner', description='Run different tao commands according to your experiments.yml', add_help=False)
        self._add_arguments()

    def _add_arguments(self):
        """Adds arguments to parser."""
        subparsers = self._parser.add_subparsers(
            help='sub-command help', dest='command')

        # Top-level parser
        self._parser.add_argument('project', help='The name of the project')
        self._parser.add_argument('experiments', nargs='+')
        self._parser.add_argument('--overwrite',
                                  action='store_true',
                                  help='If this flag is set, the affected dirs will be deleted and recreated')

        # 'split' command
        parser_split = subparsers.add_parser(
            'split', help='Split a dataset into train / val subsets')
        parser_split.add_argument(
            '--subset', required=True, help='The dataset to split')
        parser_split.add_argument(
            '--val', default=0.2, help='Percentage of the dataset to use for validation')

        # 'convert' command
        parser_convert = subparsers.add_parser(
            'convert', help='Convert a dataset to tfrecords')

        # 'train' command
        parser_train = subparsers.add_parser('train', help='Train a model')
        parser_train.add_argument(
            '-s', '--stop', help='Stop all running training sessions', action='store_true')

        # 'export' command
        parser_export = subparsers.add_parser('export', help='Export a model')
        parser_export.add_argument('-m', '--model',
                                   help='The Filename of the model to export')

    def parse(self, args=None) -> Namespace:
        return self._parser.parse_args(args)

    def print_usage(self):
        self._parser.print_usage(sys.stderr)

    def print_help(self):
        self._parser.print_help(sys.stderr)
