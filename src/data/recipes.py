import json
import pickle
import os
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2]


def load_raw_recipes():
    recipes = {}
    with open(os.path.join(project_dir, 'data', 'raw', 'recipes.json')) as file:
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
        for raw_recipe in raw_recipes:
            data = {}
            for key in keys:
                data[key] = raw_recipe[key]
            recipes[data['recipe_id']] = data
    return recipes


def load_recipes(size):
    if size not in ['small', 'medium', 'large']:
        raise KeyError
    with open(os.path.join(project_dir, 'data', 'processed', f'recipes.{size}.pkl'), 'rb') as file:
        recipes = pickle.load(file)
    return recipes


def load_available_recipe_ids() -> set:
    recipe_ids = set()
    with open(os.path.join(project_dir, 'data', 'raw', 'recipes.json')) as file:
        recipes = json.load(file)
        for recipe in recipes:
            recipe_ids.add(recipe['recipe_id'])
    return recipe_ids


if __name__ == '__main__':
    recipes = load_recipes(size='small')
    print(len(recipes))
