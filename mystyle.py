def mystyle():
  """
  Create custom plotting style.

  Returns
  -------
  my_style : dict
      Dictionary with matplotlib parameters.

  """
  # color pallette
  style = {
      # Use LaTeX to write all text
      "text.usetex": False,
      "font.family": "Arial", #"Open Sans",
      "mathtext.fontset": "custom",
      #"font.weight": "normal",
      # Use 16pt font in plots, to match 16pt font in document
      "axes.labelsize": 16,
      "axes.titlesize": 18,
      "font.size": 16,
      # Make the legend/label fonts a little smaller
      "legend.fontsize": 14,
      "xtick.labelsize": 14,
      "ytick.labelsize": 14,
      "axes.linewidth": 1.5,
      "lines.markersize": 3.0,
      "lines.linewidth": 1.5,
      "xtick.major.width": 1.,
      "ytick.major.width": 1.,
      "axes.labelweight": "normal",
      "axes.spines.right": False,
      "axes.spines.top": False,
      "axes.edgecolor": 'white',
      "axes.facecolor": '#EAEAF2',
      'xtick.color': 'white',
      'ytick.color': 'white',
      "axes.facecolor": "black",
      'figure.facecolor':'black'
  }

  return style
