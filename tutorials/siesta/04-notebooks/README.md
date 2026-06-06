# Interactive Jupyter Notebooks

Interactive tutorials for hands-on learning with atomate2siesta. These notebooks allow you to experiment with parameters, visualize results in real-time, and learn by doing.

## Why Notebooks?

- **Interactive**: Modify code and see results immediately
- **Visual**: Embedded plots and visualizations
- **Educational**: Explanations alongside executable code
- **Exploratory**: Try different parameters easily
- **Self-paced**: Work through at your own speed

## Installation

To use these notebooks, you need Jupyter:

```bash
# Install Jupyter and visualization tools
pip install jupyter matplotlib ipywidgets
```

Optional recommended packages:
```bash
pip install plotly nglview  # For advanced visualization
```

## Running Notebooks

### Local Execution
```bash
# Navigate to notebooks directory
cd tutorials/notebooks

# Start Jupyter
jupyter notebook

# Or use JupyterLab (recommended)
jupyter lab
```

### Google Colab
Upload notebooks to Google Colab for cloud execution (no local installation needed).

## Available Notebooks

### 00_interactive_quickstart.ipynb
**First Steps with atomate2siesta**
- Your first calculation in notebook format
- Interactive parameter exploration
- Real-time result visualization
- Estimated time: 15 minutes

### 01_basics_interactive.ipynb
**Core Workflows**
- Relaxation calculations
- Band structure
- Electronic properties
- Interactive examples with widgets
- Estimated time: 30 minutes

### 02_convergence_visualization.ipynb
**Parameter Convergence Studies**
- Interactive convergence testing
- k-point and cutoff convergence
- Real-time plotting
- Parameter recommendation widgets
- Estimated time: 25 minutes

### 03_phonon_analysis.ipynb
**Vibrational Properties**
- Phonon calculations setup
- Band structure visualization
- Mode analysis and animation
- Thermal properties plotting
- Estimated time: 35 minutes

### 04_surface_exploration.ipynb
**Surface Energy Calculations**
- Surface termination generation
- Energy calculations
- Interactive Wulff construction
- Surface stability plots
- Estimated time: 30 minutes

### 05_recipe_playground.ipynb
**RecipeBook Experimentation**
- Try all recipes interactively
- Compare different recipes
- Parameter widgets for customization
- Result comparison and visualization
- Estimated time: 40 minutes

### 06_troubleshooting_guide.ipynb
**Interactive Debugging**
- Diagnose common errors
- Apply fixes interactively
- Test solutions
- Learn systematic debugging
- Estimated time: 25 minutes

## Features

### Interactive Widgets
Many notebooks include interactive widgets for:
- Parameter adjustment (sliders, dropdowns)
- Structure visualization
- Plot customization
- Real-time calculation updates

### Visualization
- Band structure plots
- DOS plots
- Convergence curves
- Structure viewers (3D)
- Phonon band structures
- Surface models

### Exercises
Each notebook includes:
- Guided exercises
- Challenge problems
- Solutions (hidden in separate cells)
- Extension activities

## Notebook Structure

All notebooks follow a consistent structure:

1. **Introduction**: Learning objectives and prerequisites
2. **Setup**: Import statements and configuration
3. **Theory**: Brief conceptual overview
4. **Examples**: Working code cells
5. **Exercises**: Hands-on practice
6. **Visualization**: Plotting and analysis
7. **Summary**: Key takeaways
8. **Next Steps**: Related topics and resources

## Tips for Using Notebooks

### Getting Started
1. Run cells sequentially (Shift+Enter)
2. Modify parameters and re-run to see effects
3. Try exercises before looking at solutions
4. Experiment freely - you can always reload

### Troubleshooting
- **Kernel issues**: Restart kernel if code doesn't run
- **Import errors**: Check atomate2siesta installation
- **SIESTA not found**: Set SIESTA_CMD in config
- **Plots not showing**: Try `%matplotlib inline`

### Best Practices
- Save your work frequently
- Create copies to preserve original
- Take notes in markdown cells
- Export to HTML/PDF for sharing

## Dry-Run Mode

Most notebooks use **dry-run mode** by default:
- Generates input files without running SIESTA
- Much faster for learning
- No SIESTA installation required for initial exploration
- Enable actual calculations by setting `dry_run=False`

Example:
```python
# Dry-run mode (default, fast)
flow = RecipeBook.quick_characterization(structure)

# Actual calculation (requires SIESTA)
flow = RecipeBook.quick_characterization(structure, dry_run=False)
```

## Advanced Features

### Connecting to HPC
Run notebooks locally while submitting jobs to HPC:
```python
from jobflow import run_remote

# Configure remote execution
response = run_remote(flow, resources={"nodes": 2, "cores": 48})
```

### Database Integration
Store results directly from notebooks:
```python
from jobflow import run_locally

# Results automatically saved to MongoDB
response = run_locally(flow, store=True)
```

### Custom Workflows
Build complex workflows interactively:
```python
from atomate2.siesta.jobs.core import RelaxMaker
from jobflow import Flow

# Create custom workflow
maker1 = RelaxMaker(...)
maker2 = RelaxMaker(...)

flow = Flow([maker1.make(struct1), maker2.make(struct2)])
```

## Learning Paths

### For Beginners
1. Start with `00_interactive_quickstart.ipynb`
2. Progress to `01_basics_interactive.ipynb`
3. Try `05_recipe_playground.ipynb`

### For Convergence Testing
1. `02_convergence_visualization.ipynb`
2. Then apply to your system in `01_basics_interactive.ipynb`

### For Specific Properties
- **Phonons**: `03_phonon_analysis.ipynb`
- **Surfaces**: `04_surface_exploration.ipynb`
- **Troubleshooting**: `06_troubleshooting_guide.ipynb`

### For Advanced Users
- All notebooks as reference
- Customize and extend
- Create your own notebooks using these as templates

## Converting to Scripts

To convert a notebook to a Python script:
```bash
jupyter nbconvert --to python notebook.ipynb
```

To generate HTML documentation:
```bash
jupyter nbconvert --to html notebook.ipynb
```

## Sharing Notebooks

### Export Options
- **HTML**: For viewing without Jupyter
- **PDF**: For printing or archival
- **Slides**: For presentations
- **Script**: For automation

### Sharing Code
- Share `.ipynb` files directly
- Upload to GitHub (renders automatically)
- Use nbviewer for public viewing
- Google Colab for collaborative editing

## Contributing

To contribute a new notebook:

1. Use consistent structure (see existing notebooks)
2. Include:
   - Clear learning objectives
   - Explanatory text in markdown cells
   - Working code examples
   - Exercises with solutions
   - Visualizations where appropriate
3. Test thoroughly
4. Clear all outputs before committing
5. Update this README

## Notebook Template

A blank template is available: `NOTEBOOK_TEMPLATE.ipynb`

Use it to create new tutorials following project standards.

## FAQ

**Q: Do I need SIESTA installed to use notebooks?**
A: No, most notebooks use dry-run mode by default. You can explore workflows without SIESTA installation.

**Q: Can I run calculations from notebooks?**
A: Yes, set `dry_run=False` and ensure SIESTA is configured properly.

**Q: How do I save my modified notebooks?**
A: File → Save or Ctrl+S (Cmd+S on Mac)

**Q: Why aren't my plots showing?**
A: Add `%matplotlib inline` at the top of your notebook.

**Q: Can I convert notebooks to regular Python scripts?**
A: Yes, use `jupyter nbconvert --to python notebook.ipynb`

**Q: Are these notebooks compatible with Google Colab?**
A: Yes, upload the `.ipynb` file to Colab. You may need to install atomate2siesta first.

## Further Reading

- Jupyter Documentation: https://jupyter.org/documentation
- IPython tutorials: https://ipython.readthedocs.io/
- Matplotlib gallery: https://matplotlib.org/stable/gallery/
- Main atomate2siesta tutorials: `tutorials/README.md`

## Support

- Issues with notebooks: https://github.com/arsalan-akhtar/atomate2siesta/issues
- General questions: Check `tutorials/QUICKSTART.md`
- Troubleshooting: See `tutorials/05-troubleshooting/`
