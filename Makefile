build:
	uv sync
	python src/fix_gym.py

clean:
	rm -rf .venv
	rm -rf results