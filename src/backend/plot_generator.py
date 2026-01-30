"""
Plot Generator - Creates Plotly visualizations for data analysis
Enhanced with aggregations, Y-axis scaling, categorical heatmaps, and fixed 3-var plots
Updated: limit 100 categories, show_all option, dynamic width for scrolling
"""
import pandas as pd
import numpy as np
from typing import Any, Optional
from collections import Counter
import re


class PlotGenerator:
    """Service for generating Plotly charts"""
    
    # Color palette
    COLORS = [
        "#6366f1",  # Indigo
        "#10b981",  # Emerald
        "#f59e0b",  # Amber
        "#ef4444",  # Red
        "#8b5cf6",  # Violet
        "#06b6d4",  # Cyan
        "#ec4899",  # Pink
        "#84cc16",  # Lime
        "#f97316",  # Orange
        "#14b8a6",  # Teal
    ]
    
    # Threshold for treating numeric as categorical
    LOW_CARDINALITY_THRESHOLD = 5
    
    # Default category limit (can be overridden with show_all)
    DEFAULT_CATEGORY_LIMIT = 100
    
    # Minimum bar width in pixels
    MIN_BAR_WIDTH = 25
    
    # Supported aggregation functions
    AGGREGATIONS = {
        'count': 'count',
        'sum': 'sum',
        'avg': 'mean',
        'mean': 'mean',
        'min': 'min',
        'max': 'max',
        'median': 'median',
        'std': 'std',
        'var': 'var',
        'first': 'first',
        'last': 'last',
        'nunique': 'nunique'
    }
    
    def __init__(self):
        pass
    
    def _is_low_cardinality(self, series: pd.Series) -> bool:
        """Check if a numeric column has low cardinality"""
        unique_count = series.dropna().nunique()
        return unique_count <= self.LOW_CARDINALITY_THRESHOLD
    
    def _apply_aggregation(self, df: pd.DataFrame, group_cols: list, value_col: str, agg_func: str) -> pd.DataFrame:
        """Apply aggregation function to grouped data"""
        agg_method = self.AGGREGATIONS.get(agg_func, 'mean')
        
        if agg_func == 'count':
            # Count doesn't need a value column
            result = df.groupby(group_cols).size().reset_index(name='count')
            return result
        else:
            # Ensure value column is numeric
            df_copy = df.copy()
            df_copy[value_col] = pd.to_numeric(df_copy[value_col], errors='coerce')
            
            result = df_copy.groupby(group_cols)[value_col].agg(agg_method).reset_index()
            result.columns = list(group_cols) + [f'{agg_func}_{value_col}']
            return result
    
    def _apply_y_axis_options(self, layout: dict, options: dict) -> dict:
        """Apply Y-axis scaling and range options to layout"""
        if not options:
            return layout
        
        # Y-axis range
        y_min = options.get('yRange', {}).get('min')
        y_max = options.get('yRange', {}).get('max')
        
        if y_min is not None or y_max is not None:
            if 'yaxis' not in layout:
                layout['yaxis'] = {}
            
            current_range = layout['yaxis'].get('range', [None, None])
            layout['yaxis']['range'] = [
                y_min if y_min is not None else current_range[0],
                y_max if y_max is not None else current_range[1]
            ]
        
        # Y-axis tick interval
        y_tick = options.get('yTick')
        if y_tick:
            if 'yaxis' not in layout:
                layout['yaxis'] = {}
            layout['yaxis']['dtick'] = y_tick
        
        # Scale type (linear/log)
        scale_type = options.get('scaleType', 'linear')
        if scale_type == 'log':
            if 'yaxis' not in layout:
                layout['yaxis'] = {}
            layout['yaxis']['type'] = 'log'
        
        return layout
    
    def _get_category_limit(self, options: dict) -> int:
        """Get category limit from options, or default"""
        if options and options.get('showAll'):
            return 99999  # Effectively no limit
        return self.DEFAULT_CATEGORY_LIMIT
    
    def _calculate_dynamic_width(self, num_categories: int, num_groups: int = 1) -> Optional[int]:
        """Calculate dynamic width for charts with many categories"""
        # Calculate minimum width needed
        min_width = num_categories * self.MIN_BAR_WIDTH * max(num_groups, 1)
        
        # Only return custom width if it exceeds standard chart width
        if min_width > 800:
            return max(min_width, 800)
        return None
    
    def generate_auto_plots(self, df: pd.DataFrame, columns: list[dict]) -> dict[str, list]:
        """Generate automatic plots for all suitable columns"""
        plots = []
        
        for col_info in columns:
            col_name = col_info["name"]
            dtype = col_info["dtype"]
            
            # Skip flagged columns
            flags = col_info.get("quality_flags", [])
            if "single_value" in flags or "potential_id" in flags:
                continue
            
            if dtype in ["numeric_int", "numeric_float"]:
                series = df[col_name]
                
                # Check if low cardinality
                if self._is_low_cardinality(series):
                    bar_plot = self._create_bar_chart(df, col_name)
                    if bar_plot:
                        plots.append({
                            "column": col_name,
                            "type": "bar",
                            "data": bar_plot
                        })
                else:
                    # Regular numeric - histogram + boxplot
                    hist_plot = self._create_histogram(df, col_name)
                    if hist_plot:
                        plots.append({
                            "column": col_name,
                            "type": "histogram",
                            "data": hist_plot
                        })
                    
                    box_plot = self._create_boxplot(df, col_name)
                    if box_plot:
                        plots.append({
                            "column": col_name,
                            "type": "boxplot",
                            "data": box_plot
                        })
            
            elif dtype == "categorical":
                unique_count = col_info.get("stats", {}).get("unique_count", 0)
                if unique_count and unique_count <= 15:
                    bar_plot = self._create_bar_chart(df, col_name)
                    if bar_plot:
                        plots.append({
                            "column": col_name,
                            "type": "bar",
                            "data": bar_plot
                        })
            
            # No auto-plots for text columns - they get LLM summaries instead
        
        return {"auto": plots}
    
    def generate_custom_plot(
        self, 
        df: pd.DataFrame, 
        plot_type: str, 
        x_column: str, 
        y_column: Optional[str] = None,
        color_column: Optional[str] = None,
        columns: Optional[list] = None,
        options: Optional[dict] = None
    ) -> dict[str, Any]:
        """Generate a custom plot based on user selection"""
        
        if options is None:
            options = {}
        
        plot = None
        
        # Extract options
        bin_count = options.get('binCount', 20)
        agg_func = options.get('aggregation', None)
        
        # Route to appropriate plot generator
        if plot_type == "histogram":
            plot = self._create_histogram(df, x_column, bin_count, options)
        elif plot_type == "boxplot":
            plot = self._create_boxplot(df, x_column, options)
        elif plot_type == "violin":
            plot = self._create_violin(df, x_column, options)
        elif plot_type == "bar":
            plot = self._create_bar_chart(df, x_column, y_column, agg_func, options)
        elif plot_type == "scatter" and y_column:
            plot = self._create_scatter(df, x_column, y_column, options)
        elif plot_type == "scatter_color" and y_column:
            # Color column is optional for scatter_color
            plot = self._create_scatter_colored(df, x_column, y_column, color_column, options)
        elif plot_type == "line" and y_column:
            plot = self._create_line_chart(df, x_column, y_column, agg_func, options)
        elif plot_type == "grouped_bar" and y_column:
            plot = self._create_grouped_bar(df, x_column, y_column, color_column, agg_func, options)
        elif plot_type == "stacked_bar" and y_column:
            plot = self._create_stacked_bar(df, x_column, y_column, color_column, agg_func, options)
        elif plot_type == "bubble" and columns and len(columns) >= 3:
            plot = self._create_bubble_chart(df, columns[0], columns[1], columns[2], options)
        elif plot_type == "heatmap_cat" and y_column:
            # Heatmap for two categorical variables
            plot = self._create_categorical_heatmap(df, x_column, y_column, agg_func, color_column, options)
        elif plot_type == "correlation":
            plot = self._create_correlation_matrix(df, columns)
        elif plot_type == "pairplot" and columns:
            plot = self._create_pairplot(df, columns[:4])  # Limit to 4 columns
        elif plot_type == "wordcloud":
            plot = self._create_wordcloud(df, x_column)
        elif plot_type == "pie":
            plot = self._create_pie_chart(df, x_column, y_column, agg_func, options)
        elif plot_type == "box_grouped" and y_column:
            plot = self._create_grouped_boxplot(df, x_column, y_column, color_column, options)
        else:
            # Default fallback
            plot = self._create_histogram(df, x_column, bin_count, options)
        
        if plot is None:
            plot = {"data": [], "layout": {"title": "Could not generate plot"}}
        
        # If it's an image-based plot, pass through directly
        if plot.get("is_image"):
            return plot
        
        return {
            "plot_data": plot.get("data", []),
            "plot_layout": plot.get("layout", {}),
            "total_categories": plot.get("total_categories"),
            "showing_categories": plot.get("showing_categories")
        }
    
    def _create_histogram(self, df: pd.DataFrame, column: str, bin_count: int = 20, options: dict = None) -> dict[str, Any]:
        """Create a histogram for numeric data"""
        try:
            data = pd.to_numeric(df[column], errors='coerce').dropna()
            
            if len(data) == 0:
                return None
            
            # Use provided bin count or calculate optimal
            n_bins = min(bin_count, 100)
            
            layout = {
                "title": f"Distribution of {column}",
                "xaxis": {"title": column},
                "yaxis": {"title": "Frequency"},
                "bargap": 0.05
            }
            
            # Apply Y-axis options
            if options:
                layout = self._apply_y_axis_options(layout, options)
            
            return {
                "data": [{
                    "type": "histogram",
                    "x": data.tolist(),
                    "nbinsx": n_bins,
                    "marker": {"color": self.COLORS[0]},
                    "opacity": 0.8
                }],
                "layout": layout
            }
        except Exception as e:
            print(f"Histogram error for {column}: {e}")
            return None
    
    def _create_boxplot(self, df: pd.DataFrame, column: str, options: dict = None) -> dict[str, Any]:
        """Create a boxplot for numeric data"""
        try:
            data = pd.to_numeric(df[column], errors='coerce').dropna()
            
            if len(data) == 0:
                return None
            
            layout = {
                "title": f"Boxplot of {column}",
                "yaxis": {"title": column}
            }
            
            if options:
                layout = self._apply_y_axis_options(layout, options)
            
            return {
                "data": [{
                    "type": "box",
                    "y": data.tolist(),
                    "name": column,
                    "marker": {"color": self.COLORS[1]},
                    "boxpoints": "outliers"
                }],
                "layout": layout
            }
        except Exception as e:
            print(f"Boxplot error for {column}: {e}")
            return None
    
    def _create_grouped_boxplot(self, df: pd.DataFrame, cat_col: str, num_col: str, color_col: str = None, options: dict = None) -> dict[str, Any]:
        """Create boxplots grouped by categorical variable"""
        try:
            df_clean = df[[cat_col, num_col]].dropna()
            df_clean[num_col] = pd.to_numeric(df_clean[num_col], errors='coerce')
            df_clean = df_clean.dropna()
            
            if len(df_clean) == 0:
                return None
            
            limit = self._get_category_limit(options)
            all_categories = df_clean[cat_col].unique()
            total_categories = len(all_categories)
            categories = all_categories[:limit]
            
            traces = []
            for i, cat in enumerate(categories):
                cat_data = df_clean[df_clean[cat_col] == cat][num_col]
                traces.append({
                    "type": "box",
                    "y": cat_data.tolist(),
                    "name": str(cat),
                    "marker": {"color": self.COLORS[i % len(self.COLORS)]},
                    "boxpoints": "outliers"
                })
            
            # Calculate dynamic width
            dynamic_width = self._calculate_dynamic_width(len(categories))
            
            layout = {
                "title": f"Distribution of {num_col} by {cat_col}",
                "yaxis": {"title": num_col},
                "xaxis": {"title": cat_col}
            }
            
            if dynamic_width:
                layout["width"] = dynamic_width
                layout["xaxis"]["tickangle"] = -45
            
            if options:
                layout = self._apply_y_axis_options(layout, options)
            
            return {
                "data": traces,
                "layout": layout,
                "total_categories": total_categories,
                "showing_categories": len(categories)
            }
        except Exception as e:
            print(f"Grouped boxplot error: {e}")
            return None
    
    def _create_violin(self, df: pd.DataFrame, column: str, options: dict = None) -> dict[str, Any]:
        """Create a violin plot for numeric data"""
        try:
            data = pd.to_numeric(df[column], errors='coerce').dropna()
            
            if len(data) == 0:
                return None
            
            layout = {
                "title": f"Violin Plot of {column}",
                "yaxis": {"title": column}
            }
            
            if options:
                layout = self._apply_y_axis_options(layout, options)
            
            return {
                "data": [{
                    "type": "violin",
                    "y": data.tolist(),
                    "name": column,
                    "box": {"visible": True},
                    "meanline": {"visible": True},
                    "fillcolor": self.COLORS[4],
                    "line": {"color": self.COLORS[0]},
                    "opacity": 0.7
                }],
                "layout": layout
            }
        except Exception as e:
            print(f"Violin error for {column}: {e}")
            return None
    
    def _create_bar_chart(self, df: pd.DataFrame, column: str, value_col: str = None, agg_func: str = None, options: dict = None) -> dict[str, Any]:
        """Create a bar chart for categorical data with optional aggregation"""
        try:
            limit = self._get_category_limit(options)
            
            if value_col and agg_func:
                # Aggregated bar chart
                agg_data = self._apply_aggregation(df, [column], value_col, agg_func)
                agg_col = agg_data.columns[-1]  # The aggregated column
                
                # Sort by aggregated value
                agg_data = agg_data.sort_values(agg_col, ascending=False)
                total_categories = len(agg_data)
                agg_data = agg_data.head(limit)
                
                x_vals = [str(x) for x in agg_data[column].tolist()]
                y_vals = agg_data[agg_col].tolist()
                y_title = f"{agg_func.upper()}({value_col})"
                title = f"{agg_func.upper()} of {value_col} by {column}"
            else:
                # Simple frequency count
                value_counts = df[column].value_counts()
                total_categories = len(value_counts)
                value_counts = value_counts.head(limit)
                
                if len(value_counts) == 0:
                    return None
                
                x_vals = [str(x) for x in value_counts.index.tolist()]
                y_vals = value_counts.values.tolist()
                y_title = "Count"
                title = f"Frequency of {column}"
            
            # Calculate dynamic width
            dynamic_width = self._calculate_dynamic_width(len(x_vals))
            
            layout = {
                "title": title,
                "xaxis": {"title": column, "tickangle": -45},
                "yaxis": {"title": y_title}
            }
            
            if dynamic_width:
                layout["width"] = dynamic_width
            
            if options:
                layout = self._apply_y_axis_options(layout, options)
            
            return {
                "data": [{
                    "type": "bar",
                    "x": x_vals,
                    "y": y_vals,
                    "marker": {"color": self.COLORS[2]}
                }],
                "layout": layout,
                "total_categories": total_categories,
                "showing_categories": len(x_vals)
            }
        except Exception as e:
            print(f"Bar chart error for {column}: {e}")
            return None
    
    def _create_pie_chart(self, df: pd.DataFrame, column: str, value_col: str = None, agg_func: str = None, options: dict = None) -> dict[str, Any]:
        """Create a pie chart for categorical data"""
        try:
            # Pie charts should have limited categories for readability
            pie_limit = min(self._get_category_limit(options), 20)
            
            if value_col and agg_func:
                agg_data = self._apply_aggregation(df, [column], value_col, agg_func)
                agg_col = agg_data.columns[-1]
                total_categories = len(agg_data)
                agg_data = agg_data.nlargest(pie_limit, agg_col)
                
                labels = [str(x) for x in agg_data[column].tolist()]
                values = agg_data[agg_col].tolist()
                title = f"{agg_func.upper()} of {value_col} by {column}"
            else:
                value_counts = df[column].value_counts()
                total_categories = len(value_counts)
                value_counts = value_counts.head(pie_limit)
                labels = [str(x) for x in value_counts.index.tolist()]
                values = value_counts.values.tolist()
                title = f"Distribution of {column}"
            
            return {
                "data": [{
                    "type": "pie",
                    "labels": labels,
                    "values": values,
                    "marker": {"colors": self.COLORS[:len(labels)]},
                    "hole": 0.3
                }],
                "layout": {
                    "title": title
                },
                "total_categories": total_categories,
                "showing_categories": len(labels)
            }
        except Exception as e:
            print(f"Pie chart error: {e}")
            return None
    
    def _create_scatter(self, df: pd.DataFrame, x_col: str, y_col: str, options: dict = None) -> dict[str, Any]:
        """Create a scatter plot for two numeric columns"""
        try:
            x_data = pd.to_numeric(df[x_col], errors='coerce')
            y_data = pd.to_numeric(df[y_col], errors='coerce')
            
            mask = ~(x_data.isna() | y_data.isna())
            x_clean = x_data[mask].tolist()
            y_clean = y_data[mask].tolist()
            
            if len(x_clean) == 0:
                return None
            
            marker_size = options.get('markerSize', 8) if options else 8
            opacity = options.get('opacity', 0.6) if options else 0.6
            
            layout = {
                "title": f"{x_col} vs {y_col}",
                "xaxis": {"title": x_col},
                "yaxis": {"title": y_col}
            }
            
            if options:
                layout = self._apply_y_axis_options(layout, options)
            
            return {
                "data": [{
                    "type": "scatter",
                    "mode": "markers",
                    "x": x_clean,
                    "y": y_clean,
                    "marker": {
                        "color": self.COLORS[3],
                        "size": marker_size,
                        "opacity": opacity
                    }
                }],
                "layout": layout
            }
        except Exception as e:
            print(f"Scatter plot error: {e}")
            return None
    
    def _create_scatter_colored(self, df: pd.DataFrame, x_col: str, y_col: str, color_col: str = None, options: dict = None) -> dict[str, Any]:
        """Create a scatter plot with color by third variable"""
        try:
            x_data = pd.to_numeric(df[x_col], errors='coerce')
            y_data = pd.to_numeric(df[y_col], errors='coerce')
            
            mask = ~(x_data.isna() | y_data.isna())
            
            marker_size = options.get('markerSize', 8) if options else 8
            opacity = options.get('opacity', 0.7) if options else 0.7
            
            layout = {
                "xaxis": {"title": x_col},
                "yaxis": {"title": y_col}
            }
            
            if options:
                layout = self._apply_y_axis_options(layout, options)
            
            # If no color column, just return regular scatter
            if not color_col or color_col not in df.columns:
                layout["title"] = f"{x_col} vs {y_col}"
                return {
                    "data": [{
                        "type": "scatter",
                        "mode": "markers",
                        "x": x_data[mask].tolist(),
                        "y": y_data[mask].tolist(),
                        "marker": {
                            "color": self.COLORS[3],
                            "size": marker_size,
                            "opacity": opacity
                        }
                    }],
                    "layout": layout
                }
            
            color_data = df[color_col]
            
            # Check if color column is numeric or categorical
            is_numeric_color = pd.api.types.is_numeric_dtype(color_data)
            
            # Also try to convert to numeric
            if not is_numeric_color:
                try:
                    numeric_test = pd.to_numeric(color_data, errors='coerce')
                    if numeric_test.notna().sum() / len(numeric_test) > 0.8:
                        color_data = numeric_test
                        is_numeric_color = True
                except:
                    pass
            
            if is_numeric_color:
                color_clean = pd.to_numeric(color_data, errors='coerce')[mask].tolist()
                layout["title"] = f"{x_col} vs {y_col} (colored by {color_col})"
                return {
                    "data": [{
                        "type": "scatter",
                        "mode": "markers",
                        "x": x_data[mask].tolist(),
                        "y": y_data[mask].tolist(),
                        "marker": {
                            "color": color_clean,
                            "colorscale": "Viridis",
                            "showscale": True,
                            "colorbar": {"title": color_col},
                            "size": marker_size,
                            "opacity": opacity
                        }
                    }],
                    "layout": layout
                }
            else:
                # Categorical color
                traces = []
                limit = self._get_category_limit(options)
                all_colors = color_data[mask].unique()
                total_categories = len(all_colors)
                unique_colors = all_colors[:limit]
                
                for i, cat in enumerate(unique_colors):
                    cat_mask = (color_data == cat) & mask
                    if cat_mask.sum() > 0:
                        traces.append({
                            "type": "scatter",
                            "mode": "markers",
                            "name": str(cat),
                            "x": x_data[cat_mask].tolist(),
                            "y": y_data[cat_mask].tolist(),
                            "marker": {
                                "color": self.COLORS[i % len(self.COLORS)],
                                "size": marker_size,
                                "opacity": opacity
                            }
                        })
                
                layout["title"] = f"{x_col} vs {y_col} (by {color_col})"
                return {
                    "data": traces,
                    "layout": layout,
                    "total_categories": total_categories,
                    "showing_categories": len(unique_colors)
                }
        except Exception as e:
            print(f"Colored scatter error: {e}")
            return None
    
    def _create_bubble_chart(self, df: pd.DataFrame, x_col: str, y_col: str, size_col: str, options: dict = None) -> dict[str, Any]:
        """Create a bubble chart (scatter with size by 3rd variable)"""
        try:
            x_data = pd.to_numeric(df[x_col], errors='coerce')
            y_data = pd.to_numeric(df[y_col], errors='coerce')
            size_data = pd.to_numeric(df[size_col], errors='coerce')
            
            mask = ~(x_data.isna() | y_data.isna() | size_data.isna())
            
            if mask.sum() == 0:
                return None
            
            # Normalize sizes
            sizes = size_data[mask]
            size_range = sizes.max() - sizes.min()
            if size_range > 0:
                size_normalized = ((sizes - sizes.min()) / size_range * 40 + 5).tolist()
            else:
                size_normalized = [20] * len(sizes)
            
            opacity = options.get('opacity', 0.6) if options else 0.6
            
            layout = {
                "title": f"Bubble Chart: {x_col} vs {y_col} (size: {size_col})",
                "xaxis": {"title": x_col},
                "yaxis": {"title": y_col}
            }
            
            if options:
                layout = self._apply_y_axis_options(layout, options)
            
            return {
                "data": [{
                    "type": "scatter",
                    "mode": "markers",
                    "x": x_data[mask].tolist(),
                    "y": y_data[mask].tolist(),
                    "marker": {
                        "size": size_normalized,
                        "color": self.COLORS[5],
                        "opacity": opacity,
                        "sizemode": "diameter"
                    },
                    "text": [f"{size_col}: {v:.2f}" for v in sizes.tolist()],
                    "hovertemplate": f"{x_col}: %{{x}}<br>{y_col}: %{{y}}<br>%{{text}}<extra></extra>"
                }],
                "layout": layout
            }
        except Exception as e:
            print(f"Bubble chart error: {e}")
            return None
    
    def _create_grouped_bar(self, df: pd.DataFrame, cat_col: str, num_col: str, group_col: str = None, agg_func: str = None, options: dict = None) -> dict[str, Any]:
        """Create a grouped bar chart with optional secondary grouping"""
        try:
            agg_method = agg_func if agg_func else 'mean'
            limit = self._get_category_limit(options)
            
            if group_col and group_col in df.columns:
                # Two-level grouping
                agg_data = self._apply_aggregation(df, [cat_col, group_col], num_col, agg_method)
                agg_col = agg_data.columns[-1]
                
                all_categories = agg_data[cat_col].unique()
                total_categories = len(all_categories)
                categories = all_categories[:limit]
                
                groups = agg_data[group_col].unique()[:20]  # Limit groups for legend
                
                traces = []
                for i, grp in enumerate(groups):
                    grp_data = agg_data[agg_data[group_col] == grp]
                    # Ensure all categories are present
                    y_vals = []
                    for cat in categories:
                        val = grp_data[grp_data[cat_col] == cat][agg_col].values
                        y_vals.append(float(val[0]) if len(val) > 0 else 0)
                    
                    traces.append({
                        "type": "bar",
                        "name": str(grp),
                        "x": [str(c) for c in categories],
                        "y": y_vals,
                        "marker": {"color": self.COLORS[i % len(self.COLORS)]}
                    })
                
                title = f"{agg_method.upper()} of {num_col} by {cat_col} and {group_col}"
                num_groups = len(groups)
            else:
                # Single grouping
                agg_data = self._apply_aggregation(df, [cat_col], num_col, agg_method)
                agg_col = agg_data.columns[-1]
                
                total_categories = len(agg_data)
                agg_data = agg_data.head(limit)
                categories = agg_data[cat_col].tolist()
                
                traces = [{
                    "type": "bar",
                    "x": [str(x) for x in categories],
                    "y": agg_data[agg_col].tolist(),
                    "marker": {"color": self.COLORS[4]}
                }]
                
                title = f"{agg_method.upper()} of {num_col} by {cat_col}"
                num_groups = 1
            
            # Calculate dynamic width
            dynamic_width = self._calculate_dynamic_width(len(categories), num_groups)
            
            layout = {
                "title": title,
                "xaxis": {"title": cat_col, "tickangle": -45},
                "yaxis": {"title": f"{agg_method.upper()}({num_col})"},
                "barmode": "group"
            }
            
            if dynamic_width:
                layout["width"] = dynamic_width
            
            if options:
                layout = self._apply_y_axis_options(layout, options)
            
            return {
                "data": traces,
                "layout": layout,
                "total_categories": total_categories,
                "showing_categories": len(categories)
            }
        except Exception as e:
            print(f"Grouped bar error: {e}")
            return None
    
    def _create_stacked_bar(self, df: pd.DataFrame, cat_col: str, num_col: str, stack_col: str = None, agg_func: str = None, options: dict = None) -> dict[str, Any]:
        """Create a stacked bar chart"""
        try:
            agg_method = agg_func if agg_func else 'sum'
            limit = self._get_category_limit(options)
            
            if stack_col and stack_col in df.columns:
                agg_data = self._apply_aggregation(df, [cat_col, stack_col], num_col, agg_method)
                agg_col = agg_data.columns[-1]
                
                all_categories = agg_data[cat_col].unique()
                total_categories = len(all_categories)
                categories = all_categories[:limit]
                
                stacks = agg_data[stack_col].unique()[:20]  # Limit stacks for legend
                
                traces = []
                for i, stk in enumerate(stacks):
                    stk_data = agg_data[agg_data[stack_col] == stk]
                    y_vals = []
                    for cat in categories:
                        val = stk_data[stk_data[cat_col] == cat][agg_col].values
                        y_vals.append(float(val[0]) if len(val) > 0 else 0)
                    
                    traces.append({
                        "type": "bar",
                        "name": str(stk),
                        "x": [str(c) for c in categories],
                        "y": y_vals,
                        "marker": {"color": self.COLORS[i % len(self.COLORS)]}
                    })
                
                title = f"Stacked {agg_method.upper()} of {num_col} by {cat_col}"
                num_groups = len(stacks)
            else:
                return self._create_bar_chart(df, cat_col, num_col, agg_method, options)
            
            # Calculate dynamic width
            dynamic_width = self._calculate_dynamic_width(len(categories), 1)
            
            layout = {
                "title": title,
                "xaxis": {"title": cat_col, "tickangle": -45},
                "yaxis": {"title": f"{agg_method.upper()}({num_col})"},
                "barmode": "stack"
            }
            
            if dynamic_width:
                layout["width"] = dynamic_width
            
            if options:
                layout = self._apply_y_axis_options(layout, options)
            
            return {
                "data": traces,
                "layout": layout,
                "total_categories": total_categories,
                "showing_categories": len(categories)
            }
        except Exception as e:
            print(f"Stacked bar error: {e}")
            return None
    
    def _create_categorical_heatmap(self, df: pd.DataFrame, x_col: str, y_col: str, agg_func: str = None, value_col: str = None, options: dict = None) -> dict[str, Any]:
        """Create a heatmap for two categorical variables"""
        try:
            limit = self._get_category_limit(options)
            
            if value_col and agg_func and value_col in df.columns:
                # Aggregated heatmap
                pivot = df.pivot_table(
                    index=y_col, 
                    columns=x_col, 
                    values=value_col, 
                    aggfunc=self.AGGREGATIONS.get(agg_func, 'mean')
                ).fillna(0)
                title = f"{agg_func.upper()} of {value_col} by {x_col} and {y_col}"
                colorbar_title = f"{agg_func}({value_col})"
            else:
                # Count heatmap
                pivot = pd.crosstab(df[y_col], df[x_col])
                title = f"Frequency Heatmap: {x_col} vs {y_col}"
                colorbar_title = "Count"
            
            # Track total before limiting
            total_x = len(pivot.columns)
            total_y = len(pivot.index)
            
            # Limit dimensions
            pivot = pivot.iloc[:limit, :limit]
            
            # Calculate dynamic width based on number of x categories
            dynamic_width = self._calculate_dynamic_width(len(pivot.columns))
            
            layout = {
                "title": title,
                "xaxis": {"title": x_col, "tickangle": -45},
                "yaxis": {"title": y_col, "autorange": "reversed"}
            }
            
            if dynamic_width:
                layout["width"] = dynamic_width
            
            return {
                "data": [{
                    "type": "heatmap",
                    "z": pivot.values.tolist(),
                    "x": [str(x) for x in pivot.columns.tolist()],
                    "y": [str(y) for y in pivot.index.tolist()],
                    "colorscale": "Blues",
                    "text": [[f"{v:.2f}" if isinstance(v, float) else str(v) for v in row] for row in pivot.values],
                    "texttemplate": "%{text}",
                    "textfont": {"size": 10},
                    "hoverongaps": False,
                    "colorbar": {"title": colorbar_title}
                }],
                "layout": layout,
                "total_categories": max(total_x, total_y),
                "showing_categories": max(len(pivot.columns), len(pivot.index))
            }
        except Exception as e:
            print(f"Categorical heatmap error: {e}")
            return None
    
    def _create_line_chart(self, df: pd.DataFrame, x_col: str, y_col: str, agg_func: str = None, options: dict = None) -> dict[str, Any]:
        """Create a line chart with optional aggregation"""
        try:
            limit = self._get_category_limit(options)
            
            if agg_func:
                # Aggregate by x column
                agg_data = self._apply_aggregation(df, [x_col], y_col, agg_func)
                agg_col = agg_data.columns[-1]
                sorted_df = agg_data.sort_values(x_col)
                
                total_points = len(sorted_df)
                sorted_df = sorted_df.head(limit)
                
                x_vals = sorted_df[x_col].tolist()
                y_vals = sorted_df[agg_col].tolist()
                y_title = f"{agg_func.upper()}({y_col})"
            else:
                sorted_df = df[[x_col, y_col]].dropna().sort_values(x_col)
                
                if len(sorted_df) == 0:
                    return None
                
                total_points = len(sorted_df)
                sorted_df = sorted_df.head(limit)
                
                x_vals = sorted_df[x_col].tolist()
                y_vals = sorted_df[y_col].tolist()
                y_title = y_col
            
            # Calculate dynamic width
            dynamic_width = self._calculate_dynamic_width(len(x_vals))
            
            layout = {
                "title": f"{y_title} over {x_col}",
                "xaxis": {"title": x_col},
                "yaxis": {"title": y_title}
            }
            
            if dynamic_width:
                layout["width"] = dynamic_width
            
            if options:
                layout = self._apply_y_axis_options(layout, options)
            
            return {
                "data": [{
                    "type": "scatter",
                    "mode": "lines+markers",
                    "x": x_vals,
                    "y": y_vals,
                    "line": {"color": self.COLORS[5]},
                    "marker": {"size": 4}
                }],
                "layout": layout,
                "total_categories": total_points,
                "showing_categories": len(x_vals)
            }
        except Exception as e:
            print(f"Line chart error: {e}")
            return None
    
    def _create_correlation_matrix(self, df: pd.DataFrame, columns: Optional[list] = None) -> dict[str, Any]:
        """Create a correlation matrix heatmap"""
        try:
            # Select numeric columns
            if columns:
                numeric_cols = [c for c in columns if c in df.columns]
                numeric_df = df[numeric_cols].select_dtypes(include=[np.number])
            else:
                numeric_df = df.select_dtypes(include=[np.number])
            
            if numeric_df.shape[1] < 2:
                return None
            
            # Limit to 15 columns
            numeric_df = numeric_df.iloc[:, :15]
            
            corr_matrix = numeric_df.corr()
            
            return {
                "data": [{
                    "type": "heatmap",
                    "z": corr_matrix.values.tolist(),
                    "x": corr_matrix.columns.tolist(),
                    "y": corr_matrix.columns.tolist(),
                    "colorscale": "RdBu",
                    "zmid": 0,
                    "text": [[f"{v:.2f}" for v in row] for row in corr_matrix.values],
                    "texttemplate": "%{text}",
                    "textfont": {"size": 10},
                    "hoverongaps": False
                }],
                "layout": {
                    "title": "Correlation Matrix",
                    "xaxis": {"tickangle": -45},
                    "yaxis": {"autorange": "reversed"}
                }
            }
        except Exception as e:
            print(f"Correlation matrix error: {e}")
            return None
    
    def _create_pairplot(self, df: pd.DataFrame, columns: list) -> dict[str, Any]:
        """Create a simplified pair plot (scatter matrix)"""
        try:
            # Filter to numeric columns
            numeric_cols = [c for c in columns if c in df.columns]
            if len(numeric_cols) < 2:
                return None
            
            # Limit to 4 columns
            numeric_cols = numeric_cols[:4]
            plot_df = df[numeric_cols].dropna()
            
            if len(plot_df) == 0:
                return None
            
            # Create scatter matrix using Plotly's splom
            return {
                "data": [{
                    "type": "splom",
                    "dimensions": [
                        {"label": col, "values": plot_df[col].tolist()}
                        for col in numeric_cols
                    ],
                    "marker": {
                        "color": self.COLORS[0],
                        "size": 4,
                        "opacity": 0.5
                    },
                    "diagonal": {"visible": True}
                }],
                "layout": {
                    "title": "Pair Plot",
                    "height": 600,
                    "width": 600
                }
            }
        except Exception as e:
            print(f"Pairplot error: {e}")
            return None
    
    def _create_word_frequency(self, df: pd.DataFrame, column: str) -> dict[str, Any]:
        """Create a word frequency chart for text columns"""
        try:
            # Combine all text
            text_data = df[column].dropna().astype(str)
            all_text = ' '.join(text_data)
            
            # Simple word extraction
            words = re.findall(r'\b[a-zA-Z]{3,}\b', all_text.lower())
            
            # Count and get top 20
            word_counts = Counter(words).most_common(20)
            
            if not word_counts:
                return None
            
            words, counts = zip(*word_counts)
            
            return {
                "data": [{
                    "type": "bar",
                    "x": list(counts)[::-1],
                    "y": list(words)[::-1],
                    "orientation": "h",
                    "marker": {"color": self.COLORS[6]}
                }],
                "layout": {
                    "title": f"Top Words in {column}",
                    "xaxis": {"title": "Frequency"},
                    "yaxis": {"title": "Word"},
                    "height": 500
                }
            }
        except Exception as e:
            print(f"Word frequency error for {column}: {e}")
            return None
    
    def _create_wordcloud(self, df: pd.DataFrame, column: str) -> dict[str, Any]:
        """Create a word cloud for text columns using matplotlib"""
        try:
            from wordcloud import WordCloud
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            import matplotlib.pyplot as plt
            import base64
            from io import BytesIO
            
            # Combine all text
            text_data = df[column].dropna().astype(str)
            all_text = ' '.join(text_data)
            
            if not all_text.strip():
                return None
            
            # Generate word cloud
            wordcloud = WordCloud(
                width=800,
                height=400,
                background_color='#161b22',
                colormap='Blues',
                max_words=100,
                min_font_size=10,
                max_font_size=80
            ).generate(all_text)
            
            # Create matplotlib figure
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            ax.set_title(f'Word Cloud: {column}', color='white', fontsize=14, pad=10)
            fig.patch.set_facecolor('#161b22')
            
            # Save to base64
            buffer = BytesIO()
            plt.savefig(buffer, format='png', facecolor='#161b22', edgecolor='none', bbox_inches='tight', dpi=100)
            buffer.seek(0)
            img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            plt.close(fig)
            
            # Return as image data
            return {
                "is_image": True,
                "image_base64": img_base64,
                "plot_data": [],
                "plot_layout": {}
            }
            
        except ImportError:
            print("wordcloud library not installed, falling back to word frequency bar chart")
            return self._create_word_frequency(df, column)
        except Exception as e:
            print(f"Word cloud error for {column}: {e}")
            return self._create_word_frequency(df, column)