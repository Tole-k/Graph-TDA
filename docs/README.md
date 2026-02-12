
# Docs

## View documentation

```sh
docs/serve.sh
```

will serve the documentation files on [localhost:8080](http://localhost:8080). It will also build the documentation automatically if no html files are found (you will need to manually rebuild on changes).

## Build

```sh
docs/build.sh
```

will build the documentation.

## Requirements

Requirements needed to build the docs are available in `requirements-dev.txt`, install them with `pip install -r requirements.txt`

**PSA**: The `pandoc` library in python is only a wrapper for the program. You need to manually install the program on your computer:

- Linux: `sudo apt-get install pandoc`
- Mac: `brew install pandoc`
- Windows:
  - `choco install pandoc`
  - `winget install --source winget --exact --id JohnMacFarlane.Pandoc`
  - or use a windows installer (have fun :))

## Customization

- Change the logo by replacing `_static/logo.svg`. The logo will be visible on the main page as well as the favicon.
- To change the text, you can edit `.rst` files present in this directory. ! Editing files in `API/` will not do anything, as these are auto-generated from code. You can edit files in `_templates/` to change the look of those.
- To add new tutorials, place `.ipynb` files inside `tutorials/`, then link them like in the `tutorials.rst` file.