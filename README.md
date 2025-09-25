# AI Hedge Fund 

# 0 Install uv (if not installed)
# macOS (Homebrew):  brew install uv
# Linux/macOS (script): curl -LsSf https://astral.sh/uv/install.sh | sh
# Windows (PowerShell): iwr https://astral.sh/uv/install.ps1 -UseBasicParsing | iex

# 1 Clone the repo
git clone https://github.com/VladimirDmitriev1997/ai-hedge-fund
cd ai-hedge-fund

# 2 Create an isolated environment and install dependencies from uv.lock
uv venv .venv
uv sync --frozen  # strictly from the lock file

# 3 Install notebook dependencies (optional extra)
uv sync --extra notebooks --frozen

# 4 Register the Jupyter kernel
uv run python -m ipykernel install --user --name ai-hedge-fund

# 5 Launch Jupyter Lab in the notebooks directory
uv run jupyter lab --notebook-dir=notebooks

