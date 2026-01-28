"""
LLM Service - Claude Haiku integration for data analysis insights
"""
import os
import json
from anthropic import Anthropic
from typing import Any, Optional


class LLMService:
    """Service for generating LLM-powered analysis insights"""
    
    def __init__(self):
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            print("WARNING: ANTHROPIC_API_KEY not set. LLM features will be limited.")
            self.client = None
        else:
            self.client = Anthropic(api_key=api_key)
        
        self.model = "claude-3-haiku-20240307"
    
    async def analyze(
        self, 
        dataset_info: dict, 
        columns: list[dict],
        schema_content: Optional[str] = None
    ) -> dict[str, Any]:
        """Generate comprehensive analysis using LLM"""
        
        if not self.client:
            return self._fallback_analysis(dataset_info, columns)
        
        # Build detailed column summary for prompt
        column_summaries = []
        numeric_cols = []
        categorical_cols = []
        text_cols = []
        flagged_cols = []
        
        for col in columns:
            summary = f"- **{col['name']}** ({col['dtype']}): "
            summary += f"{col['missing']} missing ({col['missing_pct']}%)"
            
            if col.get('quality_flags'):
                summary += f" [FLAGS: {', '.join(col['quality_flags'])}]"
                flagged_cols.append(col['name'])
            
            if col.get('stats'):
                stats = col['stats']
                if 'mean' in stats and stats['mean'] is not None:
                    summary += f"\n  Statistics: min={stats.get('min')}, max={stats.get('max')}, mean={stats.get('mean'):.2f}, median={stats.get('median')}, std={stats.get('std'):.2f}"
                    if stats.get('outliers_low', 0) > 0 or stats.get('outliers_high', 0) > 0:
                        summary += f", outliers: {stats.get('outliers_low', 0)} low, {stats.get('outliers_high', 0)} high"
                    numeric_cols.append(col['name'])
                elif 'unique_count' in stats:
                    summary += f", {stats['unique_count']} unique values"
                    if 'top_values' in stats and stats['top_values']:
                        top_3 = stats['top_values'][:3]
                        top_str = ", ".join([f"{v['value']}({v['percent']}%)" for v in top_3])
                        summary += f"\n  Top values: {top_str}"
                    if col['dtype'] == 'categorical':
                        categorical_cols.append(col['name'])
                    elif col['dtype'] == 'text':
                        text_cols.append(col['name'])
            
            column_summaries.append(summary)
        
        columns_text = "\n".join(column_summaries)
        
        # Dataset characteristics summary
        characteristics = f"""
DATASET CHARACTERISTICS:
- Total records: {dataset_info['rows']}
- Total columns: {dataset_info['columns']}
- Numeric columns ({len(numeric_cols)}): {', '.join(numeric_cols[:10])}{'...' if len(numeric_cols) > 10 else ''}
- Categorical columns ({len(categorical_cols)}): {', '.join(categorical_cols[:10])}{'...' if len(categorical_cols) > 10 else ''}
- Text columns ({len(text_cols)}): {', '.join(text_cols[:5])}{'...' if len(text_cols) > 5 else ''}
- Columns with quality issues ({len(flagged_cols)}): {', '.join(flagged_cols[:10])}{'...' if len(flagged_cols) > 10 else ''}
"""
        
        schema_section = ""
        if schema_content:
            schema_section = f"\n\nSCHEMA/DESCRIPTION PROVIDED BY USER:\n{schema_content}\n"
        
        prompt = f"""You are an expert data analyst providing a comprehensive exploratory data analysis report. Analyze this dataset thoroughly:

FILENAME: {dataset_info.get('filename', 'unknown')}
{characteristics}

DETAILED COLUMN INFORMATION:
{columns_text}
{schema_section}

Provide your analysis in the following JSON format. Be COMPREHENSIVE and SPECIFIC:

{{
    "summary": "Write 2 full paragraphs (NOT bullet points, flowing prose):\\n\\nParagraph 1: Describe what this dataset contains - what kind of data is this? What does each record represent? What time period or context does it cover? What are the key variables and what do they measure? Explain the structure and scope of the data.\\n\\nParagraph 2: Describe the overall data quality and characteristics - how complete is the data? What are the notable distributions? Are there any interesting patterns visible in the basic statistics? What preprocessing might be needed? What is this data suitable for?",
    
    "insights": [
        "INSIGHT 1: [Specific finding about data distribution] - Reference actual numbers, percentages, or statistics. Explain what this means practically.",
        "INSIGHT 2: [Specific finding about relationships or patterns] - Connect multiple variables if relevant. Explain implications.",
        "INSIGHT 3: [Data quality observation] - Be specific about which columns and what issues. Suggest remediation.",
        "INSIGHT 4: [Outlier or anomaly finding] - Identify specific outliers with numbers. Explain potential causes.",
        "INSIGHT 5: [Business/domain insight] - What does this data tell us about the underlying phenomenon?",
        "INSIGHT 6: [Recommendation for analysis] - What specific analysis or modeling approach would be valuable?",
        "INSIGHT 7: [Notable statistic or finding] - Highlight something surprising or important in the data."
    ],
    
    "limitations": [
        "LIMITATION 1: [Specific data quality issue] - Which columns affected, how severe, what's the impact on analysis?",
        "LIMITATION 2: [Coverage or sampling limitation] - What's missing or potentially biased in this data?",
        "LIMITATION 3: [What cannot be concluded] - What questions can this data NOT answer?"
    ]
}}

IMPORTANT GUIDELINES:
- The summary MUST be 2 substantial paragraphs of flowing prose (4-6 sentences each), NOT bullet points
- Each insight MUST reference specific column names and actual statistics from the data
- Insights should be ACTIONABLE - tell the analyst what to DO with this information
- Be specific about numbers: "Age ranges from 0.17 to 80 with mean 29.5" not just "Age varies"
- Connect findings to potential use cases (prediction, segmentation, etc.)
- If you see patterns suggesting this is a well-known dataset (like Titanic), leverage that domain knowledge

Respond ONLY with valid JSON, no other text."""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}]
            )
            
            response_text = response.content[0].text.strip()
            
            # Clean up response if needed
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.startswith("```"):
                response_text = response_text[3:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            
            result = json.loads(response_text.strip())
            
            # Ensure summary has proper line breaks for display
            if 'summary' in result:
                result['summary'] = result['summary'].replace('\\n\\n', '\n\n').replace('\\n', '\n')
            
            return result
            
        except json.JSONDecodeError as e:
            print(f"JSON parse error: {e}")
            print(f"Response was: {response_text[:500]}")
            return self._fallback_analysis(dataset_info, columns)
        except Exception as e:
            print(f"LLM error: {e}")
            return self._fallback_analysis(dataset_info, columns)
    
    async def suggest_plot(self, columns: list[dict]) -> dict[str, Any]:
        """Suggest appropriate plot type for given columns"""
        
        if not self.client:
            return self._fallback_plot_suggestion(columns)
        
        col_details = []
        for col in columns:
            detail = f"- {col['name']}: type={col['dtype']}"
            if col.get('stats'):
                if 'unique_count' in col['stats']:
                    detail += f", {col['stats']['unique_count']} unique values"
                if 'mean' in col['stats']:
                    detail += f", mean={col['stats']['mean']}, std={col['stats'].get('std', 'N/A')}"
            col_details.append(detail)
        
        num_cols = len(columns)
        
        if num_cols == 1:
            prompt = f"""The user selected 1 column from their dataset:
{chr(10).join(col_details)}

Suggest which additional columns would create an interesting visualization. Also recommend the best plot type.

Respond in JSON:
{{
    "plot_type": "histogram|boxplot|violin|bar|scatter|line|scatter_color|bubble|correlation",
    "x_column": "{columns[0]['name']}",
    "y_column": null,
    "reasoning": "For this single column, I recommend [plot type]. To create more interesting visualizations, consider adding [suggest 2-3 column types that would pair well, e.g., 'a categorical column for grouping' or 'another numeric column for correlation analysis']."
}}"""
        elif num_cols == 2:
            prompt = f"""The user selected 2 columns:
{chr(10).join(col_details)}

Suggest the best visualization for these two columns.

Respond in JSON:
{{
    "plot_type": "scatter|line|grouped_bar|scatter_color",
    "x_column": "first_column",
    "y_column": "second_column",
    "reasoning": "Brief explanation of why this plot type is best for these columns"
}}"""
        else:
            prompt = f"""The user selected {num_cols} columns:
{chr(10).join(col_details)}

Suggest the best multi-variable visualization.

Respond in JSON:
{{
    "plot_type": "scatter_color|bubble|correlation|pairplot",
    "x_column": "column_for_x",
    "y_column": "column_for_y", 
    "reasoning": "Explanation of how to best visualize these multiple variables"
}}"""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=400,
                messages=[{"role": "user", "content": prompt}]
            )
            
            response_text = response.content[0].text.strip()
            
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.startswith("```"):
                response_text = response_text[3:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            
            return json.loads(response_text.strip())
            
        except Exception as e:
            print(f"Plot suggestion error: {e}")
            return self._fallback_plot_suggestion(columns)
    
    def _fallback_analysis(self, dataset_info: dict, columns: list[dict]) -> dict[str, Any]:
        """Fallback analysis when LLM is unavailable"""
        
        numeric_cols = [c for c in columns if c['dtype'].startswith('numeric')]
        cat_cols = [c for c in columns if c['dtype'] == 'categorical']
        text_cols = [c for c in columns if c['dtype'] == 'text']
        missing_cols = [c for c in columns if c['missing'] > 0]
        flagged_cols = [c for c in columns if c.get('quality_flags')]
        
        # Build comprehensive summary
        summary_p1 = f"This dataset contains {dataset_info['rows']:,} records across {dataset_info['columns']} columns. "
        summary_p1 += f"The data includes {len(numeric_cols)} numeric variables suitable for statistical analysis, "
        summary_p1 += f"{len(cat_cols)} categorical variables for grouping and segmentation"
        if text_cols:
            summary_p1 += f", and {len(text_cols)} text fields containing unstructured data"
        summary_p1 += ". Each record appears to represent a distinct observation or entity in the dataset."
        
        summary_p2 = f"Data quality assessment reveals that {len(missing_cols)} columns contain missing values that may require imputation or careful handling. "
        if flagged_cols:
            flags = []
            for c in flagged_cols:
                flags.extend(c.get('quality_flags', []))
            summary_p2 += f"Quality flags have been raised for {len(flagged_cols)} columns, including issues such as: {', '.join(set(flags))}. "
        summary_p2 += "The numeric variables show varying distributions that should be explored through visualization before modeling. "
        summary_p2 += "This dataset appears suitable for exploratory analysis, statistical testing, and potentially predictive modeling depending on the target variable."
        
        summary = f"{summary_p1}\n\n{summary_p2}"
        
        # Build meaningful insights
        insights = []
        
        if numeric_cols:
            col = numeric_cols[0]
            if col.get('stats') and 'mean' in col['stats']:
                stats = col['stats']
                insights.append(f"The column '{col['name']}' ranges from {stats.get('min')} to {stats.get('max')} with a mean of {stats.get('mean'):.2f} and standard deviation of {stats.get('std'):.2f}, indicating {'high' if stats.get('std', 0) > stats.get('mean', 1) else 'moderate'} variability in the data.")
        
        if cat_cols:
            insights.append(f"Found {len(cat_cols)} categorical columns ({', '.join([c['name'] for c in cat_cols[:3]])}) that can be used for grouping, stratification, or as features in classification models.")
        
        if missing_cols:
            worst = max(missing_cols, key=lambda x: x['missing_pct'])
            insights.append(f"Missing data is present in {len(missing_cols)} columns. The most affected is '{worst['name']}' with {worst['missing_pct']}% missing values - consider imputation strategies or exclusion depending on analysis goals.")
        
        if flagged_cols:
            for c in flagged_cols[:2]:
                if 'single_value' in c.get('quality_flags', []):
                    insights.append(f"Column '{c['name']}' contains only a single unique value and provides no discriminative information - recommend removing from analysis.")
                if 'potential_id' in c.get('quality_flags', []):
                    insights.append(f"Column '{c['name']}' appears to be an identifier column with high cardinality - should be excluded from statistical modeling but may be useful for record tracking.")
        
        insights.append(f"Dataset has {dataset_info['rows']:,} observations which is {'sufficient' if dataset_info['rows'] > 100 else 'limited'} for most statistical analyses and machine learning approaches.")
        
        # Pad to minimum 5 insights
        while len(insights) < 5:
            insights.append("Further domain-specific analysis recommended to uncover deeper patterns and relationships in the data.")
        
        limitations = [
            f"Automated analysis may miss domain-specific patterns - recommend consulting with subject matter experts familiar with this type of data.",
            f"Statistical relationships identified do not imply causation - experimental design or causal inference methods needed for causal claims.",
            f"Results should be validated on held-out data before deployment, and any derived insights verified against external sources when possible."
        ]
        
        return {
            "summary": summary,
            "insights": insights[:7],
            "limitations": limitations
        }
    
    def _fallback_plot_suggestion(self, columns: list[dict]) -> dict[str, Any]:
        """Fallback plot suggestion when LLM is unavailable"""
        
        if len(columns) == 1:
            col = columns[0]
            if col['dtype'].startswith('numeric'):
                return {
                    "plot_type": "histogram",
                    "x_column": col['name'],
                    "y_column": None,
                    "reasoning": f"For the single numeric column '{col['name']}', a histogram shows the distribution. Consider adding another numeric column to create a scatter plot, or a categorical column for grouped comparisons."
                }
            else:
                return {
                    "plot_type": "bar",
                    "x_column": col['name'],
                    "y_column": None,
                    "reasoning": f"For the categorical column '{col['name']}', a bar chart shows frequency counts. Consider adding a numeric column to see averages by category."
                }
        
        elif len(columns) == 2:
            numeric_cols = [c for c in columns if c['dtype'].startswith('numeric')]
            cat_cols = [c for c in columns if c['dtype'] == 'categorical']
            
            if len(numeric_cols) == 2:
                return {
                    "plot_type": "scatter",
                    "x_column": numeric_cols[0]['name'],
                    "y_column": numeric_cols[1]['name'],
                    "reasoning": f"A scatter plot reveals the relationship between '{numeric_cols[0]['name']}' and '{numeric_cols[1]['name']}'. Add a third categorical column for color-coding by groups."
                }
            elif len(numeric_cols) == 1 and len(cat_cols) == 1:
                return {
                    "plot_type": "grouped_bar",
                    "x_column": cat_cols[0]['name'],
                    "y_column": numeric_cols[0]['name'],
                    "reasoning": f"A grouped bar chart shows average '{numeric_cols[0]['name']}' across categories of '{cat_cols[0]['name']}'."
                }
        
        elif len(columns) >= 3:
            numeric_cols = [c for c in columns if c['dtype'].startswith('numeric')]
            
            if len(numeric_cols) >= 3:
                return {
                    "plot_type": "scatter_color",
                    "x_column": numeric_cols[0]['name'],
                    "y_column": numeric_cols[1]['name'],
                    "reasoning": f"With 3+ variables, a colored scatter plot shows '{numeric_cols[0]['name']}' vs '{numeric_cols[1]['name']}' with color representing '{numeric_cols[2]['name']}'. For an overview of all relationships, try the correlation matrix."
                }
            elif len(numeric_cols) >= 2:
                return {
                    "plot_type": "correlation",
                    "x_column": columns[0]['name'],
                    "y_column": columns[1]['name'],
                    "reasoning": "A correlation matrix shows relationships between all numeric variables at once."
                }
        
        # Default fallback
        return {
            "plot_type": "histogram",
            "x_column": columns[0]['name'],
            "y_column": None,
            "reasoning": "Starting with a histogram to explore the distribution of the first selected variable."
        }