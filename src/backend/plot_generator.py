"""
Plot Generator - Creates Plotly visualizations for data analysis
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
    
    def __init__(self):
        pass
    
    def _is_low_cardinality(self, series: pd.Series) -> bool:
        """Check if a numeric column has low cardinality"""
        unique_count = series.dropna().nunique()
        return unique_count <= self.LOW_CARDINALITY_THRESHOLD
    
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
            
            elif dtype == "text":
                # Word frequency chart for text columns
                word_plot = self._create_word_frequency(df, col_name)
                if word_plot:
                    plots.append({
                        "column": col_name,
                        "type": "word_frequency",
                        "data": word_plot
                    })
        
        return {"auto": plots}
    
    def generate_custom_plot(
        self, 
        df: pd.DataFrame, 
        plot_type: str, 
        x_column: str, 
        y_column: Optional[str] = None,
        color_column: Optional[str] = None,
        columns: Optional[list] = None
    ) -> dict[str, Any]:
        """Generate a custom plot based on user selection"""
        
        plot = None
        
        if plot_type == "histogram":
            plot = self._create_histogram(df, x_column)
        elif plot_type == "boxplot":
            plot = self._create_boxplot(df, x_column)
        elif plot_type == "violin":
            plot = self._create_violin(df, x_column)
        elif plot_type == "bar":
            plot = self._create_bar_chart(df, x_column)
        elif plot_type == "scatter" and y_column:
            plot = self._create_scatter(df, x_column, y_column)
        elif plot_type == "scatter_color" and y_column and color_column:
            plot = self._create_scatter_colored(df, x_column, y_column, color_column)
        elif plot_type == "line" and y_column:
            plot = self._create_line_chart(df, x_column, y_column)
        elif plot_type == "grouped_bar" and y_column:
            plot = self._create_grouped_bar(df, x_column, y_column)
        elif plot_type == "bubble" and columns and len(columns) >= 3:
            plot = self._create_bubble_chart(df, columns[0], columns[1], columns[2])
        elif plot_type == "correlation":
            plot = self._create_correlation_matrix(df, columns)
        elif plot_type == "pairplot" and columns:
            plot = self._create_pairplot(df, columns[:4])  # Limit to 4 columns
        else:
            plot = self._create_histogram(df, x_column)
        
        if plot is None:
            plot = {"data": [], "layout": {"title": "Could not generate plot"}}
        
        return {
            "plot_data": plot.get("data", []),
            "plot_layout": plot.get("layout", {})
        }
    
    def _create_histogram(self, df: pd.DataFrame, column: str) -> dict[str, Any]:
        """Create a histogram for numeric data"""
        try:
            data = pd.to_numeric(df[column], errors='coerce').dropna()
            
            if len(data) == 0:
                return None
            
            n_bins = min(int(np.sqrt(len(data))), 50)
            
            return {
                "data": [{
                    "type": "histogram",
                    "x": data.tolist(),
                    "nbinsx": n_bins,
                    "marker": {"color": self.COLORS[0]},
                    "opacity": 0.8
                }],
                "layout": {
                    "title": f"Distribution of {column}",
                    "xaxis": {"title": column},
                    "yaxis": {"title": "Frequency"},
                    "bargap": 0.05
                }
            }
        except Exception as e:
            print(f"Histogram error for {column}: {e}")
            return None
    
    def _create_boxplot(self, df: pd.DataFrame, column: str) -> dict[str, Any]:
        """Create a boxplot for numeric data"""
        try:
            data = pd.to_numeric(df[column], errors='coerce').dropna()
            
            if len(data) == 0:
                return None
            
            return {
                "data": [{
                    "type": "box",
                    "y": data.tolist(),
                    "name": column,
                    "marker": {"color": self.COLORS[1]},
                    "boxpoints": "outliers"
                }],
                "layout": {
                    "title": f"Boxplot of {column}",
                    "yaxis": {"title": column}
                }
            }
        except Exception as e:
            print(f"Boxplot error for {column}: {e}")
            return None
    
    def _create_violin(self, df: pd.DataFrame, column: str) -> dict[str, Any]:
        """Create a violin plot for numeric data"""
        try:
            data = pd.to_numeric(df[column], errors='coerce').dropna()
            
            if len(data) == 0:
                return None
            
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
                "layout": {
                    "title": f"Violin Plot of {column}",
                    "yaxis": {"title": column}
                }
            }
        except Exception as e:
            print(f"Violin error for {column}: {e}")
            return None
    
    def _create_bar_chart(self, df: pd.DataFrame, column: str) -> dict[str, Any]:
        """Create a bar chart for categorical data"""
        try:
            value_counts = df[column].value_counts().head(15)
            
            if len(value_counts) == 0:
                return None
            
            return {
                "data": [{
                    "type": "bar",
                    "x": [str(x) for x in value_counts.index.tolist()],
                    "y": value_counts.values.tolist(),
                    "marker": {"color": self.COLORS[2]}
                }],
                "layout": {
                    "title": f"Frequency of {column}",
                    "xaxis": {"title": column, "tickangle": -45},
                    "yaxis": {"title": "Count"}
                }
            }
        except Exception as e:
            print(f"Bar chart error for {column}: {e}")
            return None
    
    def _create_scatter(self, df: pd.DataFrame, x_col: str, y_col: str) -> dict[str, Any]:
        """Create a scatter plot for two numeric columns"""
        try:
            x_data = pd.to_numeric(df[x_col], errors='coerce')
            y_data = pd.to_numeric(df[y_col], errors='coerce')
            
            mask = ~(x_data.isna() | y_data.isna())
            x_clean = x_data[mask].tolist()
            y_clean = y_data[mask].tolist()
            
            if len(x_clean) == 0:
                return None
            
            return {
                "data": [{
                    "type": "scatter",
                    "mode": "markers",
                    "x": x_clean,
                    "y": y_clean,
                    "marker": {
                        "color": self.COLORS[3],
                        "size": 8,
                        "opacity": 0.6
                    }
                }],
                "layout": {
                    "title": f"{x_col} vs {y_col}",
                    "xaxis": {"title": x_col},
                    "yaxis": {"title": y_col}
                }
            }
        except Exception as e:
            print(f"Scatter plot error: {e}")
            return None
    
    def _create_scatter_colored(self, df: pd.DataFrame, x_col: str, y_col: str, color_col: str) -> dict[str, Any]:
        """Create a scatter plot with color by third variable"""
        try:
            x_data = pd.to_numeric(df[x_col], errors='coerce')
            y_data = pd.to_numeric(df[y_col], errors='coerce')
            color_data = df[color_col]
            
            mask = ~(x_data.isna() | y_data.isna())
            
            # Check if color column is numeric or categorical
            is_numeric_color = pd.api.types.is_numeric_dtype(color_data)
            
            if is_numeric_color:
                color_clean = pd.to_numeric(color_data, errors='coerce')[mask].tolist()
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
                            "size": 8,
                            "opacity": 0.7
                        }
                    }],
                    "layout": {
                        "title": f"{x_col} vs {y_col} (colored by {color_col})",
                        "xaxis": {"title": x_col},
                        "yaxis": {"title": y_col}
                    }
                }
            else:
                # Categorical color
                traces = []
                unique_colors = color_data[mask].unique()[:10]  # Limit to 10 categories
                
                for i, cat in enumerate(unique_colors):
                    cat_mask = (color_data == cat) & mask
                    traces.append({
                        "type": "scatter",
                        "mode": "markers",
                        "name": str(cat),
                        "x": x_data[cat_mask].tolist(),
                        "y": y_data[cat_mask].tolist(),
                        "marker": {
                            "color": self.COLORS[i % len(self.COLORS)],
                            "size": 8,
                            "opacity": 0.7
                        }
                    })
                
                return {
                    "data": traces,
                    "layout": {
                        "title": f"{x_col} vs {y_col} (by {color_col})",
                        "xaxis": {"title": x_col},
                        "yaxis": {"title": y_col}
                    }
                }
        except Exception as e:
            print(f"Colored scatter error: {e}")
            return None
    
    def _create_bubble_chart(self, df: pd.DataFrame, x_col: str, y_col: str, size_col: str) -> dict[str, Any]:
        """Create a bubble chart (scatter with size by 3rd variable)"""
        try:
            x_data = pd.to_numeric(df[x_col], errors='coerce')
            y_data = pd.to_numeric(df[y_col], errors='coerce')
            size_data = pd.to_numeric(df[size_col], errors='coerce')
            
            mask = ~(x_data.isna() | y_data.isna() | size_data.isna())
            
            # Normalize sizes
            sizes = size_data[mask]
            size_normalized = ((sizes - sizes.min()) / (sizes.max() - sizes.min()) * 40 + 5).tolist()
            
            return {
                "data": [{
                    "type": "scatter",
                    "mode": "markers",
                    "x": x_data[mask].tolist(),
                    "y": y_data[mask].tolist(),
                    "marker": {
                        "size": size_normalized,
                        "color": self.COLORS[5],
                        "opacity": 0.6,
                        "sizemode": "diameter"
                    },
                    "text": [f"{size_col}: {v:.2f}" for v in sizes.tolist()],
                    "hovertemplate": f"{x_col}: %{{x}}<br>{y_col}: %{{y}}<br>%{{text}}<extra></extra>"
                }],
                "layout": {
                    "title": f"Bubble Chart: {x_col} vs {y_col} (size: {size_col})",
                    "xaxis": {"title": x_col},
                    "yaxis": {"title": y_col}
                }
            }
        except Exception as e:
            print(f"Bubble chart error: {e}")
            return None
    
    def _create_grouped_bar(self, df: pd.DataFrame, cat_col: str, num_col: str) -> dict[str, Any]:
        """Create a grouped bar chart"""
        try:
            grouped = df.groupby(cat_col)[num_col].mean().head(15)
            
            if len(grouped) == 0:
                return None
            
            return {
                "data": [{
                    "type": "bar",
                    "x": [str(x) for x in grouped.index.tolist()],
                    "y": grouped.values.tolist(),
                    "marker": {"color": self.COLORS[4]}
                }],
                "layout": {
                    "title": f"Average {num_col} by {cat_col}",
                    "xaxis": {"title": cat_col, "tickangle": -45},
                    "yaxis": {"title": f"Mean {num_col}"}
                }
            }
        except Exception as e:
            print(f"Grouped bar error: {e}")
            return None
    
    def _create_line_chart(self, df: pd.DataFrame, x_col: str, y_col: str) -> dict[str, Any]:
        """Create a line chart"""
        try:
            sorted_df = df[[x_col, y_col]].dropna().sort_values(x_col)
            
            if len(sorted_df) == 0:
                return None
            
            return {
                "data": [{
                    "type": "scatter",
                    "mode": "lines+markers",
                    "x": sorted_df[x_col].tolist(),
                    "y": sorted_df[y_col].tolist(),
                    "line": {"color": self.COLORS[5]},
                    "marker": {"size": 4}
                }],
                "layout": {
                    "title": f"{y_col} over {x_col}",
                    "xaxis": {"title": x_col},
                    "yaxis": {"title": y_col}
                }
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