import json
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Set, Dict

project_dir = Path(__file__).resolve().parents[3]


def load_raw_recipes() -> Dict:
    recipes = defaultdict(dict)
    with open(f'{project_dir}/data/raw/recipes.json') as file:
        keys = [
            'recipe_id',
            'user_id',
            'title',
            'story_or_description',
            'country',
            'ingredients',
            'steps',
        ]
        raw_recipes = json.load(file)
        for recipe_id in raw_recipes:
            for key in keys:
                recipes[int(recipe_id)][key] = raw_recipes[recipe_id][key]
    return recipes


def load_recipes(size) -> Dict:
    if size not in ['small', 'medium', 'large']:
        raise KeyError
    with open(f'{project_dir}/data/processed/recipes.{size}.pkl', 'rb') as file:
        recipes = pickle.load(file)
    return recipes


def load_available_recipe_ids() -> Set[int]:
    recipe_ids = set()
    with open(f'{project_dir}/data/raw/recipes.json') as file:
        recipes = json.load(file)
        for recipe in recipes:
            recipe_ids.add(recipe['recipe_id'])
    return recipe_ids
