file_path = ".venv/lib/python3.10/site-packages/gym/utils/passive_env_checker.py"

with open(file_path, "r") as f:
    content = f.read()

content = content.replace("np.bool8", "np.bool")

with open(file_path, "w") as f:
    f.write(content)
