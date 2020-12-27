from invoke import task

from src.data import cookpad, msmarco
from src.data.cookpad import generate_n_listwise
from src.data.msmarco import generate_n_listwise
from src.utils.seed import set_seed

@task
def generate_listwise_cookpad(_ctx):
    set_seed()
    cookpad.generate_n_listwise.generate()

@task
def generate_listwise_msmarco(_ctx):
    set_seed()
    msmarco.generate_n_listwise.generate()
