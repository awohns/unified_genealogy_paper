import warnings

import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.gridspec
import matplotlib.pyplot as plt
import mpld3
from mpld3 import plugins, utils
#mpld3.enable_notebook() # if in a jupyter notebook - also needs %matplotlib inline

# Get data and massage it to the right format
base_filename = "all-data/merged_hgdp_1kg_sgdp_high_cov_ancients_chr20.dated.binned.historic.20nodes.tmrcas"
mean_tmrca_df = pd.read_csv(base_filename + ".csv", index_col=0)
# Flatten the upper triangular means and log (also matches the hist data)
mean_tmrcas = np.log(mean_tmrca_df.values[np.triu_indices(mean_tmrca_df.shape[0])])
histdata_tmrcas = np.load(base_filename + ".npz") # Rows are arranged as [(0, 0), (0, 1), (0, 2), (0, 3), (1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3)]
dist_tmrcas = histdata_tmrcas['histdata']
bins = histdata_tmrcas['bins']


for row in range(mean_tmrca_df.shape[1]):
    for col in range(row+1, mean_tmrca_df.shape[0]):
       mean_tmrca_df.iloc[col, row] = mean_tmrca_df.iloc[row, col]


class LinkedView(plugins.PluginBase):
    """A plugin showing how multiple axes can be linked"""

    def css(self):
        # Not sure why this doesn't work
        return """
            .mpld3-xaxis .tick text {transform: "rotate(90)"}
            """

    JAVASCRIPT = """
    mpld3.register_plugin("linkedview", LinkedViewPlugin);
    LinkedViewPlugin.prototype = Object.create(mpld3.Plugin.prototype);
    LinkedViewPlugin.prototype.constructor = LinkedViewPlugin;
    LinkedViewPlugin.prototype.requiredProps = [
        "idpts", "idlab", "idline", "idmean", "histdata", "meandata", "order", "labels"];
    LinkedViewPlugin.prototype.defaultProps = {}
    function LinkedViewPlugin(fig, props){
        mpld3.Plugin.call(this, fig, props);
    };

    LinkedViewPlugin.prototype.draw = function(){
      var pts = mpld3.get_element(this.props.idpts);
      var histlabel = mpld3.get_element(this.props.idlab);
      var histline = mpld3.get_element(this.props.idline);
      var meanline = mpld3.get_element(this.props.idmean);
      var histdata = this.props.histdata;
      var meandata = this.props.meandata;
      var order = this.props.order;
      var labels = this.props.labels;
      var n_cols = order.length;

      function mouseover(xy_pos, point_index){
        var col = point_index % n_cols;
        var row =  (point_index - col) / n_cols;
        // There's probably a better way to set text, but at least this works
        histlabel.obj._groups[0][0].innerHTML = labels[col] + " - " + labels[row];
        /* convert to order used in original data */
        col = order[col]
        row = order[row]
        /* convert to upper triangular index */
        if (row > col) {
            [row, col] = [col, row]; 
        }
        var tri_index = (n_cols*(n_cols+1)/2) - (n_cols-row)*((n_cols-row)+1)/2 + col - row
        for (var i=0; i < histline.data.length-2; i++) {
            histline.data[i+1][1] = histdata[tri_index][i];        
        }
        histline.elements().transition()
            .attr("d", histline.datafunc(histline.data))
            .style("stroke", this.style.fill);
        meanline.data[0][0] = meandata[tri_index];
        meanline.data[1][0] = meandata[tri_index];
        meanline.elements().transition()
            .attr("d", meanline.datafunc(meanline.data))
            .style("stroke", this.style.fill);

      }
      pts.elements().on("mouseover", mouseover);
    };
    """

    def __init__(self, points, histlabel, histline, meanline, histdata, meandata, order):
        if isinstance(points, matplotlib.lines.Line2D):
            suffix = "pts"
        else:
            suffix = None
        self.dict_ = {
            "type": "linkedview",
            "idpts": utils.get_id(points, suffix),
            "idlab": utils.get_id(histlabel),
            "idline": utils.get_id(histline),
            "idmean": utils.get_id(meanline),
            "histdata": histdata,
            "meandata": meandata,
            "order": list(order.keys()),
            "labels": list(order.values()),
        }
        
result = sns.clustermap(
    mean_tmrca_df,
    method="average",
    linewidths = 0,
    rasterized=True,
    cmap=plt.cm.inferno_r,
    xticklabels=1,
    yticklabels=0,
)
heatmap_plot = result.ax_heatmap
result.cax.tick_params(labelsize=8)
result.cax.set_xlabel("Average TMRCA (generations)", size=10)
heatmap_plot.set_xticklabels(result.ax_heatmap.get_xmajorticklabels(), fontsize=7)
heatmap_plot.invert_xaxis()
heatmap_plot.invert_yaxis()
result.ax_row_dendrogram.invert_yaxis()

# Clustermap reorders the cols & rows, so we must use the new order for the source data
row_order = getattr(result.dendrogram_row, 'reordered_ind', np.arange(mean_tmrca_df.shape[0]))
col_order = getattr(result.dendrogram_col, 'reordered_ind', np.arange(mean_tmrca_df.shape[1]))
assert row_order==col_order
# Require python 3.6 so that dict retains order
order = {o:mean_tmrca_df.columns[o] for o in col_order}

mesh_obj = heatmap_plot.collections[0]
histogram_plot = result.ax_col_dendrogram  # put the histogram where the column dendrogram normally goes
histogram_plot.clear()

# Change histogram axis position to make room for X axis
pos1 = histogram_plot.get_position() # get the original position 
pos2 = [pos1.x0, pos1.y0+0.05,  pos1.width, pos1.height/1.3]
histogram_plot.set_position(pos2) # set a new position
histogram_plot.set_xlabel("tMRCA (generations)")
histogram_plot.set_xticks(np.log([1e3, 2e3, 5e3, 1e4, 2e4, 5e4, 1e5]))
histogram_plot.set_xticklabels([1e3, 2e3, 5e3, 1e4, 2e4, 5e4, 1e5])

x, y = np.append(bins, bins[-1]), np.pad(np.zeros(dist_tmrcas.shape[1]), 1) # initial hist is all at 0
# create the histogram as a line object (easier to manipulate)
histogram_plot.step(x, y, '-')
# Add a line at the mean
histogram_plot.plot([0, 0], [0, 1], linewidth=4, color='w')  # Hide the mean line marker by making it white
histogram_plot.text(bins[1], 1.5, " ", verticalalignment='top')
hist_line = histogram_plot.lines[0]
mean_line = histogram_plot.lines[1]
hist_label = histogram_plot.texts[0]

histogram_plot.set_ylim(0, 1.5)
histogram_plot.set_xlim(bins[0], bins[-1])

plugins.connect(
    result.fig,
    LinkedView(
        mesh_obj,
        hist_label,
        hist_line,
        mean_line,
        # round the histogram values to save file space
        np.round(dist_tmrcas, 4),
        np.round(mean_tmrcas, 5), order))
with warnings.catch_warnings():
    # mpld3 triggers DeprecationWarning: np.asscalar
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    mpld3.save_html(result.fig, base_filename + ".html")
    # mpld3.show() # For notebooks
