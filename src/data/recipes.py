import json
import os
from pathlib import Path


def load_recipes():
    project_dir = Path(__file__).resolve().parents[2]
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
