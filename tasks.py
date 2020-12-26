from invoke import task

from src.data.cookpad.generate_n_listwise import generate
from src.utils.seed import set_seed

@task
def generate_listwise(_ctx):
    set_seed()
    generate()
