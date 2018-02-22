"""
Application entry-point.
"""
import click
from pathlib import Path


@click.group()
def cli():
    """
    Create a main command group
    """
    pass


@cli.command()
@click.option('--work-dir', '-w', type=Path)
@click.option('--epochs', '-e', type=int)
def train(**kwargs):
    """
    Collect the command line options and arguments we declared with click and
    pass to the method doing the actual training as kwargs.
    """
    from tf_mnist.train import do_train
    do_train(**kwargs)


@cli.command('get-data')
def get_data(**kwargs):
    """
    Download and unpack MNIST data
    """
    from tf_mnist.data import get_data
    get_data(**kwargs)


if __name__ == '__main__':
    cli()
